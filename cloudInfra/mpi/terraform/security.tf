# Security Group for Master Node
resource "aws_security_group" "mpi_master" {
  name_prefix = "${var.cluster_name}-master-"
  vpc_id      = aws_vpc.mpi_vpc.id

  # SSH access
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_ssh_cidrs
  }

  # Web interface access
  ingress {
    from_port   = var.web_port
    to_port     = var.web_port
    protocol    = "tcp"
    cidr_blocks = var.allowed_web_cidrs
  }

  # MPI communication with workers
  ingress {
    from_port       = 0
    to_port         = 65535
    protocol        = "tcp"
    security_groups = [aws_security_group.mpi_worker.id]
  }

  # Allow all outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.cluster_name}-master-sg"
  }
}

# Security Group for Worker Nodes
resource "aws_security_group" "mpi_worker" {
  name_prefix = "${var.cluster_name}-worker-"
  vpc_id      = aws_vpc.mpi_vpc.id

  # SSH access from master
  ingress {
    from_port       = 22
    to_port         = 22
    protocol        = "tcp"
    security_groups = [aws_security_group.mpi_master.id]
  }

  # MPI communication from master
  ingress {
    from_port       = 0
    to_port         = 65535
    protocol        = "tcp"
    security_groups = [aws_security_group.mpi_master.id]
  }

  # Inter-worker communication
  ingress {
    from_port = 0
    to_port   = 65535
    protocol  = "tcp"
    self      = true
  }

  # Allow all outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.cluster_name}-worker-sg"
  }
}

# Security Group for EFS
resource "aws_security_group" "efs" {
  name_prefix = "${var.cluster_name}-efs-"
  vpc_id      = aws_vpc.mpi_vpc.id

  # NFS access from master and workers
  ingress {
    from_port       = 2049
    to_port         = 2049
    protocol        = "tcp"
    security_groups = [aws_security_group.mpi_master.id, aws_security_group.mpi_worker.id]
  }

  # Allow all outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.cluster_name}-efs-sg"
  }
}

# Security Group for Application Load Balancer
resource "aws_security_group" "alb" {
  name_prefix = "${var.cluster_name}-alb-"
  vpc_id      = aws_vpc.mpi_vpc.id

  # HTTP access
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = var.allowed_web_cidrs
  }

  # HTTPS access
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = var.allowed_web_cidrs
  }

  # Allow all outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.cluster_name}-alb-sg"
  }
}

# Network ACL for additional security (optional)
resource "aws_network_acl" "mpi_nacl" {
  vpc_id = aws_vpc.mpi_vpc.id

  # Allow inbound HTTP/HTTPS
  ingress {
    protocol   = "tcp"
    rule_no    = 100
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 80
    to_port    = 80
  }

  ingress {
    protocol   = "tcp"
    rule_no    = 110
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 443
    to_port    = 443
  }

  # Allow inbound SSH
  ingress {
    protocol   = "tcp"
    rule_no    = 120
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 22
    to_port    = 22
  }

  # Allow inbound ephemeral ports
  ingress {
    protocol   = "tcp"
    rule_no    = 130
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 1024
    to_port    = 65535
  }

  # Allow all outbound traffic
  egress {
    protocol   = "-1"
    rule_no    = 100
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 0
    to_port    = 0
  }

  tags = {
    Name = "${var.cluster_name}-nacl"
  }
}

# WAF for additional web application security
resource "aws_wafv2_web_acl" "mpi_waf" {
  count = var.enable_web_interface ? 1 : 0
  
  name  = "${var.cluster_name}-waf"
  scope = "REGIONAL"

  default_action {
    allow {}
  }

  # Rate limiting rule
  rule {
    name     = "RateLimitRule"
    priority = 1

    action {
      block {}
    }

    statement {
      rate_based_statement {
        limit              = 2000
        aggregate_key_type = "IP"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                 = "RateLimitRule"
      sampled_requests_enabled    = true
    }
  }

  # AWS Managed Rules
  rule {
    name     = "AWSManagedRulesCommonRuleSet"
    priority = 2

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesCommonRuleSet"
        vendor_name = "AWS"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                 = "CommonRuleSetMetric"
      sampled_requests_enabled    = true
    }
  }

  tags = {
    Name = "${var.cluster_name}-waf"
  }

  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                 = "MPISearchEngineWAF"
    sampled_requests_enabled    = true
  }
}

# Associate WAF with ALB
resource "aws_wafv2_web_acl_association" "mpi_waf_association" {
  count = var.enable_web_interface ? 1 : 0

  resource_arn = aws_lb.mpi_alb.arn
  web_acl_arn  = aws_wafv2_web_acl.mpi_waf[0].arn
}
