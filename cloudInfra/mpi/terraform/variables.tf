# AWS Configuration
variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"
}

variable "availability_zones" {
  description = "List of availability zones to use"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b", "us-east-1c"]
}

# Cluster Configuration
variable "cluster_name" {
  description = "Name of the MPI cluster"
  type        = string
  default     = "mpi-search-engine"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "owner" {
  description = "Owner of the infrastructure"
  type        = string
  default     = "mpi-team"
}

# EC2 Configuration
variable "master_instance_type" {
  description = "EC2 instance type for master node"
  type        = string
  default     = "c5n.xlarge"  # 4 vCPUs, 10.5 GB RAM, Enhanced Networking
}

variable "worker_instance_type" {
  description = "EC2 instance type for worker nodes"
  type        = string
  default     = "c5n.large"   # 2 vCPUs, 5.25 GB RAM, Enhanced Networking
}

variable "worker_count" {
  description = "Number of worker nodes"
  type        = number
  default     = 4
}

variable "use_spot_instances" {
  description = "Use spot instances for cost optimization"
  type        = bool
  default     = true
}

variable "spot_price" {
  description = "Maximum spot price per hour"
  type        = string
  default     = "0.10"
}

# Storage Configuration
variable "root_volume_size" {
  description = "Size of root EBS volume in GB"
  type        = number
  default     = 50
}

variable "efs_performance_mode" {
  description = "EFS performance mode (generalPurpose or maxIO)"
  type        = string
  default     = "generalPurpose"
}

# Network Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.10.0/24", "10.0.20.0/24", "10.0.30.0/24"]
}

# Security Configuration
variable "public_key_path" {
  description = "Path to SSH public key file"
  type        = string
  default     = "~/.ssh/id_rsa.pub"
}

variable "allowed_ssh_cidrs" {
  description = "CIDR blocks allowed for SSH access"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # Change this to your IP for better security
}

variable "allowed_web_cidrs" {
  description = "CIDR blocks allowed for web access"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

# Monitoring and Logging
variable "log_retention_days" {
  description = "CloudWatch log retention period in days"
  type        = number
  default     = 14
}

variable "enable_monitoring" {
  description = "Enable detailed CloudWatch monitoring"
  type        = bool
  default     = true
}

# Auto-scaling Configuration
variable "enable_autoscaling" {
  description = "Enable auto-scaling for worker nodes"
  type        = bool
  default     = false
}

variable "min_workers" {
  description = "Minimum number of worker nodes when auto-scaling"
  type        = number
  default     = 2
}

variable "max_workers" {
  description = "Maximum number of worker nodes when auto-scaling"
  type        = number
  default     = 10
}

# MPI Configuration
variable "mpi_slots_per_node" {
  description = "Number of MPI slots per node"
  type        = number
  default     = 4
}

variable "mpi_version" {
  description = "OpenMPI version to install"
  type        = string
  default     = "4.1.4"
}

# Application Configuration
variable "app_version" {
  description = "Version of the search engine application"
  type        = string
  default     = "latest"
}

variable "enable_web_interface" {
  description = "Enable web interface for the search engine"
  type        = bool
  default     = true
}

variable "web_port" {
  description = "Port for web interface"
  type        = number
  default     = 8080
}

# Cost Optimization
variable "auto_shutdown_enabled" {
  description = "Enable automatic shutdown of instances when idle"
  type        = bool
  default     = true
}

variable "idle_timeout_minutes" {
  description = "Minutes of inactivity before auto-shutdown"
  type        = number
  default     = 60
}

# Data Configuration
variable "dataset_s3_bucket" {
  description = "S3 bucket containing the dataset"
  type        = string
  default     = ""
}

variable "dataset_s3_prefix" {
  description = "S3 prefix for dataset files"
  type        = string
  default     = "dataset/"
}

# Notification Configuration
variable "notification_email" {
  description = "Email address for cluster notifications"
  type        = string
  default     = ""
}

# Backup Configuration
variable "enable_backup" {
  description = "Enable automated backups"
  type        = bool
  default     = true
}

variable "backup_retention_days" {
  description = "Number of days to retain backups"
  type        = number
  default     = 7
}
