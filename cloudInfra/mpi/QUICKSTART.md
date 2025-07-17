# AWS EC2 MPI Cluster Quick Start Guide

This guide will help you deploy and run the MPI Search Engine on AWS EC2 instances using the provided infrastructure automation.

## Prerequisites

Before starting, ensure you have:

1. **AWS Account** with appropriate permissions
2. **AWS CLI** installed and configured
3. **Terraform** (version 1.0+)
4. **SSH key pair** for EC2 access
5. **jq** for JSON processing
6. **Basic understanding** of AWS, MPI, and command line

## Quick Deployment (5 minutes)

### 1. Setup Environment
```bash
# Navigate to the cloud infrastructure directory
cd cloudInfra/mpi

# Run the setup script
./scripts/setup.sh
```

### 2. Configure AWS Credentials
```bash
aws configure
# Enter your AWS Access Key ID, Secret Access Key, Region, and output format
```

### 3. Deploy the Cluster
```bash
# Deploy with default settings (4 worker nodes, c5n.large instances)
./scripts/deploy.sh

# Or customize the deployment
./scripts/deploy.sh -n my-cluster -w 8 -t c5n.xlarge -r us-west-2
```

### 4. Run Your First Search
```bash
# Run a distributed search
./scripts/run-mpi.sh search "machine learning"

# Monitor cluster performance
./scripts/run-mpi.sh monitor
```

### 5. Access Web Interface
The deployment will provide a URL for the web interface where you can:
- Submit search queries through a web UI
- Monitor cluster status
- View performance metrics

### 6. Cleanup Resources
```bash
# Clean up all AWS resources (with backup)
./scripts/cleanup.sh

# Quick cleanup without backup
./scripts/cleanup.sh --force --no-backup
```

## Deployment Options

### Basic Deployment
```bash
./scripts/deploy.sh
```
- 1 master node (c5n.xlarge)
- 4 worker nodes (c5n.large)
- Spot instances for cost savings
- Auto-shutdown after 60 minutes of inactivity

### Custom Deployment
```bash
./scripts/deploy.sh \
  --cluster-name production-search \
  --workers 16 \
  --instance-type c5n.2xlarge \
  --region us-west-2 \
  --environment prod
```

### Development Deployment
```bash
./scripts/deploy.sh \
  --cluster-name dev-test \
  --workers 2 \
  --instance-type c5n.large \
  --skip-test
```

## Running Searches

### Basic Search
```bash
./scripts/run-mpi.sh search "your search query"
```

### Advanced Search Options
```bash
# Use specific number of processes
./scripts/run-mpi.sh -n 16 search "distributed computing"

# JSON output format
./scripts/run-mpi.sh -f json search "parallel algorithms"

# Verbose output with timing
./scripts/run-mpi.sh -v -b search "high performance computing"
```

### Upload Custom Dataset
```bash
# Upload local dataset
./scripts/run-mpi.sh upload ./my-documents/

# Upload single file
./scripts/run-mpi.sh upload ./large-document.txt
```

### Performance Benchmarking
```bash
# Run comprehensive benchmark
./scripts/run-mpi.sh benchmark

# Results will be saved to the cluster and displayed
```

## Monitoring and Management

### Cluster Status
```bash
# Check cluster health
./scripts/run-mpi.sh status

# Real-time monitoring
./scripts/run-mpi.sh monitor
```

### Log Access
```bash
# View master node logs
./scripts/run-mpi.sh logs master

# View worker node logs
./scripts/run-mpi.sh logs workers

# View search execution logs
./scripts/run-mpi.sh logs search
```

### Direct SSH Access
```bash
# Get SSH command from deployment output
terraform output ssh_command

# Or manually connect
ssh -i ~/.ssh/mpi-cluster-key ubuntu@<master-ip>
```

## Configuration Options

### Instance Types Recommendation
| Use Case | Master | Workers | Total vCPUs | Est. Cost/Hour |
|----------|--------|---------|-------------|----------------|
| Development | c5n.large | c5n.large | 8 | $0.30 |
| Testing | c5n.xlarge | c5n.large | 12 | $0.60 |
| Production | c5n.2xlarge | c5n.xlarge | 32 | $2.40 |
| High-Performance | c5n.4xlarge | c5n.2xlarge | 64 | $9.60 |

### Cost Optimization Features
- **Spot Instances**: 50-70% cost savings
- **Auto-shutdown**: Stops idle instances automatically
- **Flexible scaling**: Scale workers based on workload
- **Regional pricing**: Choose cost-effective regions

### Network-Optimized Instances
The infrastructure uses c5n instances which provide:
- Up to 100 Gbps network performance
- Enhanced networking with SR-IOV
- Low latency for MPI communication
- Optimized for HPC workloads

## Troubleshooting

### Common Issues

1. **Deployment Fails**
   ```bash
   # Check AWS credentials
   aws sts get-caller-identity
   
   # Check Terraform state
   cd terraform && terraform show
   ```

2. **Cannot Connect to Cluster**
   ```bash
   # Check security group rules
   terraform output security_group_ids
   
   # Verify SSH key
   ssh-add ~/.ssh/mpi-cluster-key
   ```

3. **MPI Search Fails**
   ```bash
   # Check cluster status
   ./scripts/run-mpi.sh status
   
   # View detailed logs
   ./scripts/run-mpi.sh logs master
   ```

4. **High Costs**
   ```bash
   # Enable auto-shutdown
   # Edit terraform/terraform.tfvars:
   auto_shutdown_enabled = true
   idle_timeout_minutes = 30
   
   # Redeploy
   terraform apply
   ```

### Emergency Procedures

1. **Force Cleanup**
   ```bash
   ./scripts/cleanup.sh --emergency cluster-name
   ```

2. **Manual Resource Cleanup**
   ```bash
   # List all resources
   aws ec2 describe-instances --filters "Name=tag:Project,Values=MPI-Search-Engine"
   
   # Terminate instances
   aws ec2 terminate-instances --instance-ids i-1234567890abcdef0
   ```

3. **State Recovery**
   ```bash
   cd terraform
   terraform import aws_instance.mpi_master i-1234567890abcdef0
   ```

## Advanced Usage

### Custom MPI Applications
1. Upload your MPI application to `/shared/mpi-search-engine/`
2. Modify the search script to use your binary
3. Update the hostfile if needed

### Integration with CI/CD
```yaml
# Example GitHub Actions workflow
name: Deploy MPI Cluster
on: [push]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v1
      - name: Deploy Cluster
        run: ./cloudInfra/mpi/scripts/deploy.sh
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
```

### Custom Monitoring
The infrastructure includes CloudWatch integration for:
- CPU and memory metrics
- Network performance
- Custom MPI job metrics
- Cost tracking

### Scaling Operations
```bash
# Scale workers up
terraform apply -var="worker_count=8"

# Scale workers down
terraform apply -var="worker_count=2"

# Enable auto-scaling
terraform apply -var="enable_autoscaling=true"
```

## Security Best Practices

1. **Restrict SSH Access**
   ```bash
   # Edit terraform/terraform.tfvars
   allowed_ssh_cidrs = ["your.ip.address/32"]
   ```

2. **Use IAM Roles**
   - Instances use IAM roles instead of access keys
   - Minimal required permissions

3. **Network Security**
   - Private subnets for workers
   - Security groups with minimal access
   - VPC flow logs enabled

4. **Encryption**
   - EBS volumes encrypted at rest
   - EFS encrypted in transit and at rest
   - S3 bucket with server-side encryption

## Cost Monitoring

### Estimated Costs (US East 1)
- **c5n.large**: ~$0.108/hour ($77/month)
- **c5n.xlarge**: ~$0.216/hour ($154/month)
- **c5n.2xlarge**: ~$0.432/hour ($309/month)
- **c5n.4xlarge**: ~$0.864/hour ($618/month)

### Cost Alerts
The infrastructure can be configured with CloudWatch billing alerts:
```bash
# Enable billing alerts
aws budgets create-budget --account-id 123456789012 \
  --budget file://budget.json
```

### Savings Strategies
1. Use spot instances (default enabled)
2. Enable auto-shutdown (default enabled)
3. Use appropriate instance sizes
4. Deploy in cost-effective regions
5. Monitor usage with AWS Cost Explorer

## Support and Documentation

### Additional Resources
- [AWS EC2 Documentation](https://docs.aws.amazon.com/ec2/)
- [OpenMPI Documentation](https://www.open-mpi.org/doc/)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)

### Community
- GitHub Issues for bug reports
- Discussions for questions and improvements
- Wiki for additional examples and tutorials

### Professional Support
For production deployments, consider:
- AWS Enterprise Support
- Professional services for optimization
- Custom implementation consulting
