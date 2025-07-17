# Launch Template for Master Node
resource "aws_launch_template" "mpi_master" {
  name_prefix   = "${var.cluster_name}-master-"
  image_id      = data.aws_ami.ubuntu.id
  instance_type = var.master_instance_type
  key_name      = aws_key_pair.mpi_cluster_key.key_name

  vpc_security_group_ids = [aws_security_group.mpi_master.id]

  iam_instance_profile {
    name = aws_iam_instance_profile.mpi_instance_profile.name
  }

  block_device_mappings {
    device_name = "/dev/sda1"
    ebs {
      volume_size           = var.root_volume_size
      volume_type           = "gp3"
      encrypted             = true
      delete_on_termination = true
    }
  }

  monitoring {
    enabled = var.enable_monitoring
  }

  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }

  user_data = base64encode(templatefile("${path.module}/../scripts/user-data-master.sh", {
    cluster_name        = var.cluster_name
    efs_id              = aws_efs_file_system.mpi_shared_storage.id
    aws_region          = var.aws_region
    mpi_version         = var.mpi_version
    s3_bucket           = aws_s3_bucket.mpi_artifacts.bucket
    web_port            = var.web_port
    enable_web_interface = var.enable_web_interface
    auto_shutdown_enabled = var.auto_shutdown_enabled
    idle_timeout_minutes = var.idle_timeout_minutes
  }))

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name = "${var.cluster_name}-master"
      Role = "master"
      Type = "mpi-node"
    }
  }

  tags = {
    Name = "${var.cluster_name}-master-template"
  }
}

# Launch Template for Worker Nodes
resource "aws_launch_template" "mpi_worker" {
  name_prefix   = "${var.cluster_name}-worker-"
  image_id      = data.aws_ami.ubuntu.id
  instance_type = var.worker_instance_type
  key_name      = aws_key_pair.mpi_cluster_key.key_name

  vpc_security_group_ids = [aws_security_group.mpi_worker.id]

  iam_instance_profile {
    name = aws_iam_instance_profile.mpi_instance_profile.name
  }

  block_device_mappings {
    device_name = "/dev/sda1"
    ebs {
      volume_size           = var.root_volume_size
      volume_type           = "gp3"
      encrypted             = true
      delete_on_termination = true
    }
  }

  monitoring {
    enabled = var.enable_monitoring
  }

  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }

  user_data = base64encode(templatefile("${path.module}/../scripts/user-data-worker.sh", {
    cluster_name        = var.cluster_name
    efs_id              = aws_efs_file_system.mpi_shared_storage.id
    aws_region          = var.aws_region
    mpi_version         = var.mpi_version
    s3_bucket           = aws_s3_bucket.mpi_artifacts.bucket
    mpi_slots_per_node  = var.mpi_slots_per_node
    auto_shutdown_enabled = var.auto_shutdown_enabled
    idle_timeout_minutes = var.idle_timeout_minutes
  }))

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name = "${var.cluster_name}-worker"
      Role = "worker"
      Type = "mpi-node"
    }
  }

  tags = {
    Name = "${var.cluster_name}-worker-template"
  }
}

# Master Node Instance
resource "aws_instance" "mpi_master" {
  launch_template {
    id      = aws_launch_template.mpi_master.id
    version = "$Latest"
  }

  subnet_id = aws_subnet.public[0].id

  tags = {
    Name = "${var.cluster_name}-master"
    Role = "master"
    Type = "mpi-node"
  }
}

# Worker Node Instances
resource "aws_instance" "mpi_workers" {
  count = var.worker_count

  launch_template {
    id      = aws_launch_template.mpi_worker.id
    version = "$Latest"
  }

  subnet_id = aws_subnet.private[count.index % length(aws_subnet.private)].id

  tags = {
    Name = "${var.cluster_name}-worker-${count.index + 1}"
    Role = "worker"
    Type = "mpi-node"
    WorkerIndex = count.index + 1
  }
}

# Auto Scaling Group for Worker Nodes (if enabled)
resource "aws_autoscaling_group" "mpi_workers" {
  count = var.enable_autoscaling ? 1 : 0

  name                = "${var.cluster_name}-workers-asg"
  vpc_zone_identifier = aws_subnet.private[*].id
  target_group_arns   = var.enable_web_interface ? [aws_lb_target_group.mpi_web.arn] : []
  health_check_type   = "EC2"
  health_check_grace_period = 300

  min_size         = var.min_workers
  max_size         = var.max_workers
  desired_capacity = var.worker_count

  launch_template {
    id      = aws_launch_template.mpi_worker.id
    version = "$Latest"
  }

  tag {
    key                 = "Name"
    value               = "${var.cluster_name}-worker-asg"
    propagate_at_launch = true
  }

  tag {
    key                 = "Role"
    value               = "worker"
    propagate_at_launch = true
  }

  tag {
    key                 = "Type"
    value               = "mpi-node"
    propagate_at_launch = true
  }
}

# Auto Scaling Policies
resource "aws_autoscaling_policy" "scale_up" {
  count = var.enable_autoscaling ? 1 : 0

  name                   = "${var.cluster_name}-scale-up"
  scaling_adjustment     = 1
  adjustment_type        = "ChangeInCapacity"
  cooldown               = 300
  autoscaling_group_name = aws_autoscaling_group.mpi_workers[0].name
}

resource "aws_autoscaling_policy" "scale_down" {
  count = var.enable_autoscaling ? 1 : 0

  name                   = "${var.cluster_name}-scale-down"
  scaling_adjustment     = -1
  adjustment_type        = "ChangeInCapacity"
  cooldown               = 300
  autoscaling_group_name = aws_autoscaling_group.mpi_workers[0].name
}

# CloudWatch Alarms for Auto Scaling
resource "aws_cloudwatch_metric_alarm" "cpu_high" {
  count = var.enable_autoscaling ? 1 : 0

  alarm_name          = "${var.cluster_name}-cpu-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = "120"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "This metric monitors ec2 cpu utilization"
  alarm_actions       = [aws_autoscaling_policy.scale_up[0].arn]

  dimensions = {
    AutoScalingGroupName = aws_autoscaling_group.mpi_workers[0].name
  }
}

resource "aws_cloudwatch_metric_alarm" "cpu_low" {
  count = var.enable_autoscaling ? 1 : 0

  alarm_name          = "${var.cluster_name}-cpu-low"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = "120"
  statistic           = "Average"
  threshold           = "20"
  alarm_description   = "This metric monitors ec2 cpu utilization"
  alarm_actions       = [aws_autoscaling_policy.scale_down[0].arn]

  dimensions = {
    AutoScalingGroupName = aws_autoscaling_group.mpi_workers[0].name
  }
}

# Spot Fleet Request (alternative to ASG for spot instances)
resource "aws_spot_fleet_request" "mpi_workers_spot" {
  count = var.use_spot_instances && !var.enable_autoscaling ? 1 : 0

  iam_fleet_role      = aws_iam_role.spot_fleet_role[0].arn
  allocation_strategy = "diversified"
  target_capacity     = var.worker_count
  spot_price          = var.spot_price

  launch_template_config {
    launch_template_specification {
      id      = aws_launch_template.mpi_worker.id
      version = aws_launch_template.mpi_worker.latest_version
    }

    overrides {
      instance_type     = var.worker_instance_type
      subnet_id         = aws_subnet.private[0].id
      availability_zone = aws_subnet.private[0].availability_zone
    }

    overrides {
      instance_type     = var.worker_instance_type
      subnet_id         = aws_subnet.private[1].id
      availability_zone = aws_subnet.private[1].availability_zone
    }
  }

  tags = {
    Name = "${var.cluster_name}-spot-fleet"
  }
}

# IAM Role for Spot Fleet
resource "aws_iam_role" "spot_fleet_role" {
  count = var.use_spot_instances && !var.enable_autoscaling ? 1 : 0

  name = "${var.cluster_name}-spot-fleet-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "spotfleet.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "spot_fleet_role_policy" {
  count = var.use_spot_instances && !var.enable_autoscaling ? 1 : 0

  role       = aws_iam_role.spot_fleet_role[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2SpotFleetTaggingRole"
}
