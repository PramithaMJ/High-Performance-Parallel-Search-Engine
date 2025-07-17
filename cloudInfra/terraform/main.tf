# Terraform configuration for HPC Search Engine Cluster
# Provider configuration
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
}

# Variables
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t2.medium"
}

variable "cluster_size" {
  description = "Number of compute nodes"
  type        = number
  default     = 3
}

variable "key_name" {
  description = "AWS key pair name"
  type        = string
}

variable "project_name" {
  description = "Project name for tagging"
  type        = string
  default     = "hpc-search-engine"
}

# Data sources
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

data "aws_availability_zones" "available" {
  state = "available"
}

# VPC Configuration
resource "aws_vpc" "hpc_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name    = "${var.project_name}-vpc"
    Project = var.project_name
  }
}

# Internet Gateway
resource "aws_internet_gateway" "hpc_igw" {
  vpc_id = aws_vpc.hpc_vpc.id

  tags = {
    Name    = "${var.project_name}-igw"
    Project = var.project_name
  }
}

# Public Subnet
resource "aws_subnet" "hpc_subnet" {
  vpc_id                  = aws_vpc.hpc_vpc.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = data.aws_availability_zones.available.names[0]
  map_public_ip_on_launch = true

  tags = {
    Name    = "${var.project_name}-subnet"
    Project = var.project_name
  }
}

# Route Table
resource "aws_route_table" "hpc_rt" {
  vpc_id = aws_vpc.hpc_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.hpc_igw.id
  }

  tags = {
    Name    = "${var.project_name}-rt"
    Project = var.project_name
  }
}

# Route Table Association
resource "aws_route_table_association" "hpc_rta" {
  subnet_id      = aws_subnet.hpc_subnet.id
  route_table_id = aws_route_table.hpc_rt.id
}

# Security Group
resource "aws_security_group" "hpc_sg" {
  name_prefix = "${var.project_name}-sg"
  vpc_id      = aws_vpc.hpc_vpc.id

  # SSH access
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # MPI communication within cluster
  ingress {
    from_port = 1024
    to_port   = 65535
    protocol  = "tcp"
    self      = true
  }

  # NFS
  ingress {
    from_port = 2049
    to_port   = 2049
    protocol  = "tcp"
    self      = true
  }

  # HTTP for monitoring
  ingress {
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # All outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "${var.project_name}-sg"
    Project = var.project_name
  }
}

# Placement Group for low latency
resource "aws_placement_group" "hpc_pg" {
  name     = "${var.project_name}-pg"
  strategy = "cluster"

  tags = {
    Name    = "${var.project_name}-pg"
    Project = var.project_name
  }
}

# Master Node
resource "aws_instance" "hpc_master" {
  ami                     = data.aws_ami.ubuntu.id
  instance_type           = var.instance_type
  key_name                = var.key_name
  vpc_security_group_ids  = [aws_security_group.hpc_sg.id]
  subnet_id               = aws_subnet.hpc_subnet.id
  placement_group         = aws_placement_group.hpc_pg.id
  disable_api_termination = false

  root_block_device {
    volume_type = "gp3"
    volume_size = 20
    encrypted   = true
  }

  user_data = base64encode(templatefile("${path.module}/user_data_master.sh", {
    node_type = "master"
  }))

  tags = {
    Name    = "${var.project_name}-master"
    Type    = "master"
    Project = var.project_name
  }
}

# Worker Nodes
resource "aws_instance" "hpc_workers" {
  count                   = var.cluster_size - 1
  ami                     = data.aws_ami.ubuntu.id
  instance_type           = var.instance_type
  key_name                = var.key_name
  vpc_security_group_ids  = [aws_security_group.hpc_sg.id]
  subnet_id               = aws_subnet.hpc_subnet.id
  placement_group         = aws_placement_group.hpc_pg.id
  disable_api_termination = false

  root_block_device {
    volume_type = "gp3"
    volume_size = 20
    encrypted   = true
  }

  user_data = base64encode(templatefile("${path.module}/user_data_worker.sh", {
    node_type    = "worker"
    master_ip    = aws_instance.hpc_master.private_ip
    worker_index = count.index + 1
  }))

  depends_on = [aws_instance.hpc_master]

  tags = {
    Name    = "${var.project_name}-worker-${count.index + 1}"
    Type    = "worker"
    Project = var.project_name
  }
}

# Application Load Balancer for monitoring
resource "aws_lb" "hpc_alb" {
  name               = "${var.project_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.hpc_sg.id]
  subnets            = [aws_subnet.hpc_subnet.id]

  enable_deletion_protection = false

  tags = {
    Name    = "${var.project_name}-alb"
    Project = var.project_name
  }
}

# Target Group
resource "aws_lb_target_group" "hpc_tg" {
  name     = "${var.project_name}-tg"
  port     = 8080
  protocol = "HTTP"
  vpc_id   = aws_vpc.hpc_vpc.id

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
    Name    = "${var.project_name}-tg"
    Project = var.project_name
  }
}

# ALB Listener
resource "aws_lb_listener" "hpc_listener" {
  load_balancer_arn = aws_lb.hpc_alb.arn
  port              = "80"
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.hpc_tg.arn
  }
}

# Target Group Attachments
resource "aws_lb_target_group_attachment" "hpc_master_tg" {
  target_group_arn = aws_lb_target_group.hpc_tg.arn
  target_id        = aws_instance.hpc_master.id
  port             = 8080
}
