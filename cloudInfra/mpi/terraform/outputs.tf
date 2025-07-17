# Output Values
output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.mpi_vpc.id
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = aws_subnet.private[*].id
}

output "master_instance_id" {
  description = "ID of the master instance"
  value       = aws_instance.mpi_master.id
}

output "master_public_ip" {
  description = "Public IP of the master instance"
  value       = aws_instance.mpi_master.public_ip
}

output "master_private_ip" {
  description = "Private IP of the master instance"
  value       = aws_instance.mpi_master.private_ip
}

output "worker_instance_ids" {
  description = "IDs of the worker instances"
  value       = aws_instance.mpi_workers[*].id
}

output "worker_private_ips" {
  description = "Private IPs of the worker instances"
  value       = aws_instance.mpi_workers[*].private_ip
}

output "efs_id" {
  description = "ID of the EFS file system"
  value       = aws_efs_file_system.mpi_shared_storage.id
}

output "efs_dns_name" {
  description = "DNS name of the EFS file system"
  value       = aws_efs_file_system.mpi_shared_storage.dns_name
}

output "s3_bucket_name" {
  description = "Name of the S3 bucket for artifacts"
  value       = aws_s3_bucket.mpi_artifacts.bucket
}

output "load_balancer_dns" {
  description = "DNS name of the load balancer"
  value       = aws_lb.mpi_alb.dns_name
}

output "load_balancer_url" {
  description = "URL of the load balancer"
  value       = "http://${aws_lb.mpi_alb.dns_name}"
}

output "ssh_command" {
  description = "SSH command to connect to the master node"
  value       = "ssh -i ~/.ssh/mpi-cluster-key ubuntu@${aws_instance.mpi_master.public_ip}"
}

output "cluster_info" {
  description = "Complete cluster information"
  value = {
    cluster_name    = var.cluster_name
    master_ip       = aws_instance.mpi_master.public_ip
    worker_ips      = aws_instance.mpi_workers[*].private_ip
    total_nodes     = 1 + var.worker_count
    total_slots     = (1 + var.worker_count) * var.mpi_slots_per_node
    efs_mount_point = aws_efs_file_system.mpi_shared_storage.dns_name
    s3_bucket       = aws_s3_bucket.mpi_artifacts.bucket
  }
}

output "security_group_ids" {
  description = "Security group IDs"
  value = {
    master = aws_security_group.mpi_master.id
    worker = aws_security_group.mpi_worker.id
    efs    = aws_security_group.efs.id
    alb    = aws_security_group.alb.id
  }
}

output "key_pair_name" {
  description = "Name of the EC2 key pair"
  value       = aws_key_pair.mpi_cluster_key.key_name
}

output "cloudwatch_log_group" {
  description = "CloudWatch log group name"
  value       = aws_cloudwatch_log_group.mpi_logs.name
}

output "sns_topic_arn" {
  description = "SNS topic ARN for notifications"
  value       = aws_sns_topic.mpi_notifications.arn
}

# Output for Ansible inventory
output "ansible_inventory" {
  description = "Ansible inventory in INI format"
  value = templatefile("${path.module}/../config/inventory.tpl", {
    master_ip   = aws_instance.mpi_master.public_ip
    worker_ips  = aws_instance.mpi_workers[*].private_ip
    ssh_key     = aws_key_pair.mpi_cluster_key.key_name
  })
}

# Output hostfile for MPI
output "mpi_hostfile" {
  description = "MPI hostfile content"
  value = templatefile("${path.module}/../config/hostfile.tpl", {
    master_ip      = aws_instance.mpi_master.private_ip
    worker_ips     = aws_instance.mpi_workers[*].private_ip
    slots_per_node = var.mpi_slots_per_node
  })
}

# Performance metrics
output "performance_info" {
  description = "Expected performance characteristics"
  value = {
    master_vcpus    = data.aws_ec2_instance_type.master.default_vcpus
    worker_vcpus    = data.aws_ec2_instance_type.worker.default_vcpus
    total_vcpus     = data.aws_ec2_instance_type.master.default_vcpus + (var.worker_count * data.aws_ec2_instance_type.worker.default_vcpus)
    master_memory   = data.aws_ec2_instance_type.master.memory_size
    worker_memory   = data.aws_ec2_instance_type.worker.memory_size
    total_memory    = data.aws_ec2_instance_type.master.memory_size + (var.worker_count * data.aws_ec2_instance_type.worker.memory_size)
    network_performance = data.aws_ec2_instance_type.master.network_performance
  }
}

# Data sources for instance type information
data "aws_ec2_instance_type" "master" {
  instance_type = var.master_instance_type
}

data "aws_ec2_instance_type" "worker" {
  instance_type = var.worker_instance_type
}

# Cost estimation
output "estimated_hourly_cost" {
  description = "Estimated hourly cost (approximate)"
  value = {
    master_cost     = "~$0.20/hour"  # Approximate for c5n.xlarge
    worker_cost     = "~$0.10/hour"  # Approximate for c5n.large
    total_cost      = "~$${0.20 + (var.worker_count * 0.10)}/hour"
    monthly_cost    = "~$${(0.20 + (var.worker_count * 0.10)) * 24 * 30}/month"
    spot_savings    = var.use_spot_instances ? "50-70% savings with spot instances" : "N/A"
  }
}

# Monitoring endpoints
output "monitoring_endpoints" {
  description = "Monitoring and logging endpoints"
  value = {
    cloudwatch_logs = "https://${var.aws_region}.console.aws.amazon.com/cloudwatch/home?region=${var.aws_region}#logsV2:log-groups/log-group/${replace(aws_cloudwatch_log_group.mpi_logs.name, "/", "%2F")}"
    ec2_dashboard   = "https://${var.aws_region}.console.aws.amazon.com/ec2/v2/home?region=${var.aws_region}#Instances:"
    efs_dashboard   = "https://${var.aws_region}.console.aws.amazon.com/efs/home?region=${var.aws_region}#/file-systems/${aws_efs_file_system.mpi_shared_storage.id}"
  }
}
