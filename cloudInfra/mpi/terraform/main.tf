terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "MPI-Search-Engine"
      Environment = var.environment
      Owner       = var.owner
      CreatedBy   = "Terraform"
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# Key pair for EC2 instances
resource "aws_key_pair" "mpi_cluster_key" {
  key_name   = "${var.cluster_name}-key"
  public_key = file(var.public_key_path)

  tags = {
    Name = "${var.cluster_name}-key"
  }
}

# EFS for shared storage
resource "aws_efs_file_system" "mpi_shared_storage" {
  creation_token = "${var.cluster_name}-efs"
  
  performance_mode = "generalPurpose"
  encrypted        = true
  
  tags = {
    Name = "${var.cluster_name}-shared-storage"
  }
}

resource "aws_efs_mount_target" "mpi_efs_mount" {
  count           = length(aws_subnet.private)
  file_system_id  = aws_efs_file_system.mpi_shared_storage.id
  subnet_id       = aws_subnet.private[count.index].id
  security_groups = [aws_security_group.efs.id]
}

# Application Load Balancer for web interface (optional)
resource "aws_lb" "mpi_alb" {
  name               = "${var.cluster_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id

  enable_deletion_protection = false

  tags = {
    Name = "${var.cluster_name}-alb"
  }
}

resource "aws_lb_target_group" "mpi_web" {
  name     = "${var.cluster_name}-web-tg"
  port     = 8080
  protocol = "HTTP"
  vpc_id   = aws_vpc.mpi_vpc.id

  health_check {
    enabled             = true
    healthy_threshold   = 2
    interval            = 30
    matcher             = "200"
    path                = "/health"
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 2
  }

  tags = {
    Name = "${var.cluster_name}-web-tg"
  }
}

resource "aws_lb_listener" "mpi_web" {
  load_balancer_arn = aws_lb.mpi_alb.arn
  port              = "80"
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.mpi_web.arn
  }
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "mpi_logs" {
  name              = "/aws/ec2/${var.cluster_name}"
  retention_in_days = var.log_retention_days

  tags = {
    Name = "${var.cluster_name}-logs"
  }
}

# SNS Topic for notifications
resource "aws_sns_topic" "mpi_notifications" {
  name = "${var.cluster_name}-notifications"

  tags = {
    Name = "${var.cluster_name}-notifications"
  }
}

# IAM role for EC2 instances
resource "aws_iam_role" "mpi_instance_role" {
  name = "${var.cluster_name}-instance-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "mpi_instance_policy" {
  name = "${var.cluster_name}-instance-policy"
  role = aws_iam_role.mpi_instance_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams",
          "cloudwatch:PutMetricData",
          "ec2:DescribeInstances",
          "ec2:DescribeTags",
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_instance_profile" "mpi_instance_profile" {
  name = "${var.cluster_name}-instance-profile"
  role = aws_iam_role.mpi_instance_role.name
}

# S3 bucket for artifacts and results
resource "aws_s3_bucket" "mpi_artifacts" {
  bucket = "${var.cluster_name}-artifacts-${random_string.bucket_suffix.result}"

  tags = {
    Name = "${var.cluster_name}-artifacts"
  }
}

resource "aws_s3_bucket_versioning" "mpi_artifacts_versioning" {
  bucket = aws_s3_bucket.mpi_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "mpi_artifacts_encryption" {
  bucket = aws_s3_bucket.mpi_artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}
