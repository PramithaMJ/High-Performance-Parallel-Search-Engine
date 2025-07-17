# VPC
resource "aws_vpc" "mpi_vpc" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "${var.cluster_name}-vpc"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "mpi_igw" {
  vpc_id = aws_vpc.mpi_vpc.id

  tags = {
    Name = "${var.cluster_name}-igw"
  }
}

# Public Subnets
resource "aws_subnet" "public" {
  count = length(var.public_subnet_cidrs)

  vpc_id                  = aws_vpc.mpi_vpc.id
  cidr_block              = var.public_subnet_cidrs[count.index]
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "${var.cluster_name}-public-subnet-${count.index + 1}"
    Type = "Public"
  }
}

# Private Subnets
resource "aws_subnet" "private" {
  count = length(var.private_subnet_cidrs)

  vpc_id            = aws_vpc.mpi_vpc.id
  cidr_block        = var.private_subnet_cidrs[count.index]
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = {
    Name = "${var.cluster_name}-private-subnet-${count.index + 1}"
    Type = "Private"
  }
}

# Elastic IPs for NAT Gateways
resource "aws_eip" "nat_eip" {
  count = length(aws_subnet.public)

  domain = "vpc"
  depends_on = [aws_internet_gateway.mpi_igw]

  tags = {
    Name = "${var.cluster_name}-nat-eip-${count.index + 1}"
  }
}

# NAT Gateways
resource "aws_nat_gateway" "mpi_nat" {
  count = length(aws_subnet.public)

  allocation_id = aws_eip.nat_eip[count.index].id
  subnet_id     = aws_subnet.public[count.index].id

  tags = {
    Name = "${var.cluster_name}-nat-gateway-${count.index + 1}"
  }

  depends_on = [aws_internet_gateway.mpi_igw]
}

# Route Table for Public Subnets
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.mpi_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.mpi_igw.id
  }

  tags = {
    Name = "${var.cluster_name}-public-rt"
  }
}

# Route Tables for Private Subnets
resource "aws_route_table" "private" {
  count = length(aws_subnet.private)

  vpc_id = aws_vpc.mpi_vpc.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.mpi_nat[count.index].id
  }

  tags = {
    Name = "${var.cluster_name}-private-rt-${count.index + 1}"
  }
}

# Route Table Associations - Public
resource "aws_route_table_association" "public" {
  count = length(aws_subnet.public)

  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# Route Table Associations - Private
resource "aws_route_table_association" "private" {
  count = length(aws_subnet.private)

  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[count.index].id
}

# VPC Endpoint for S3 (for cost optimization)
resource "aws_vpc_endpoint" "s3" {
  vpc_id       = aws_vpc.mpi_vpc.id
  service_name = "com.amazonaws.${var.aws_region}.s3"

  tags = {
    Name = "${var.cluster_name}-s3-endpoint"
  }
}

# VPC Endpoint Route Table Associations
resource "aws_vpc_endpoint_route_table_association" "s3_private" {
  count = length(aws_route_table.private)

  vpc_endpoint_id = aws_vpc_endpoint.s3.id
  route_table_id  = aws_route_table.private[count.index].id
}

resource "aws_vpc_endpoint_route_table_association" "s3_public" {
  vpc_endpoint_id = aws_vpc_endpoint.s3.id
  route_table_id  = aws_route_table.public.id
}

# VPC Flow Logs for monitoring
resource "aws_flow_log" "mpi_vpc_flow_log" {
  iam_role_arn    = aws_iam_role.flow_log_role.arn
  log_destination = aws_cloudwatch_log_group.vpc_flow_log.arn
  traffic_type    = "ALL"
  vpc_id          = aws_vpc.mpi_vpc.id

  tags = {
    Name = "${var.cluster_name}-vpc-flow-log"
  }
}

resource "aws_cloudwatch_log_group" "vpc_flow_log" {
  name              = "/aws/vpc/${var.cluster_name}/flowlogs"
  retention_in_days = var.log_retention_days

  tags = {
    Name = "${var.cluster_name}-vpc-flow-log"
  }
}

resource "aws_iam_role" "flow_log_role" {
  name = "${var.cluster_name}-flow-log-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "vpc-flow-logs.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "flow_log_policy" {
  name = "${var.cluster_name}-flow-log-policy"
  role = aws_iam_role.flow_log_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogGroups",
          "logs:DescribeLogStreams"
        ]
        Effect   = "Allow"
        Resource = "*"
      }
    ]
  })
}
