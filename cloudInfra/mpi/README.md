# AWS EC2 MPI Cluster Infrastructure

This directory contains all the necessary infrastructure code and scripts to deploy and run the MPI-based parallel search engine on AWS EC2 instances.

## Directory Structure

```
cloudInfra/mpi/
├── terraform/              # Infrastructure as Code
│   ├── main.tf             # Main Terraform configuration
│   ├── variables.tf        # Variable definitions
│   ├── outputs.tf          # Output definitions
│   ├── ec2.tf              # EC2 instances configuration
│   ├── vpc.tf              # VPC and networking
│   └── security.tf         # Security groups and keys
├── ansible/                # Configuration Management
│   ├── inventory.ini       # Ansible inventory
│   ├── mpi-cluster.yml     # Main playbook
│   ├── roles/              # Ansible roles
│   │   ├── common/         # Common setup
│   │   ├── mpi/            # MPI installation
│   │   └── search-engine/  # Application deployment
├── scripts/                # Deployment and management scripts
│   ├── deploy.sh           # Main deployment script
│   ├── setup-cluster.sh    # Cluster setup
│   ├── run-mpi.sh          # MPI execution script
│   ├── cleanup.sh          # Resource cleanup
│   └── monitoring.sh       # Cluster monitoring
├── config/                 # Configuration files
│   ├── aws-config.ini      # AWS configuration
│   ├── mpi-hostfile        # MPI hostfile template
│   └── cluster-config.yml  # Cluster configuration
└── monitoring/             # Monitoring and logging
    ├── cloudwatch.tf       # CloudWatch configuration
    └── grafana-config.yml  # Grafana dashboard
```

## Quick Start

1. **Prerequisites Setup**
   ```bash
   # Install required tools
   ./scripts/setup-prerequisites.sh
   ```

2. **Deploy Infrastructure**
   ```bash
   # Deploy AWS infrastructure
   ./scripts/deploy.sh
   ```

3. **Run MPI Search Engine**
   ```bash
   # Execute parallel search
   ./scripts/run-mpi.sh -n 16 -q "your search query"
   ```

4. **Cleanup Resources**
   ```bash
   # Clean up AWS resources
   ./scripts/cleanup.sh
   ```

## Features

- **Auto-scaling EC2 cluster** with optimal instance types for MPI workloads
- **Network-optimized instances** with enhanced networking for low-latency communication
- **Shared EFS storage** for distributed file access
- **Load balancing** across multiple availability zones
- **Cost optimization** with spot instances and auto-shutdown
- **Monitoring and logging** with CloudWatch and custom metrics
- **Security** with VPC, security groups, and key management

## Configuration

Edit `config/aws-config.ini` to customize:
- Instance types and counts
- AWS regions and availability zones
- Storage configuration
- Network settings
- Security parameters

## Monitoring

The infrastructure includes:
- CloudWatch metrics and alarms
- Custom MPI performance monitoring
- Grafana dashboards for visualization
- Log aggregation and analysis

## Cost Optimization

- Spot instances for cost savings
- Auto-shutdown after idle periods
- Resource tagging for cost tracking
- Optimized instance sizing based on workload
