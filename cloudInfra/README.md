# HPC Search Engine Cloud Infrastructure

This directory contains Terraform and Ansible configurations to deploy your HPC Search Engine on AWS using t2.medium instances.

##  Quick Start

1. **Prerequisites:**
   ```bash
   # Install required tools
   # Terraform: https://www.terraform.io/downloads.html
   # AWS CLI: https://aws.amazon.com/cli/
   # Ansible: pip install ansible
   ```

2. **Configure AWS:**
   ```bash
   aws configure
   # Enter your AWS Access Key ID, Secret Access Key, and region
   ```

3. **Update Configuration:**
   ```bash
   cd terraform
   # Edit terraform.tfvars with your settings
   key_name = "your-aws-key-pair-name"  # Your AWS key pair
   ```

4. **Deploy:**
   ```bash
   chmod +x deploy.sh
   ./deploy.sh deploy
   ```

##  Directory Structure

```
cloudInfra/
├── terraform/              # Infrastructure as Code
│   ├── main.tf             # Main Terraform configuration
│   ├── outputs.tf          # Output definitions
│   ├── terraform.tfvars    # Variables (customize this)
│   ├── user_data_master.sh # Master node initialization
│   └── user_data_worker.sh # Worker node initialization
├── ansible/                # Configuration Management
│   ├── site.yml            # Main playbook
│   ├── inventory.yml       # Hosts inventory
│   └── templates/          # Configuration templates
│       ├── run_cluster.sh.j2
│       ├── monitor_cluster.sh.j2
│       ├── dashboard.py.j2
│       └── worker_health.sh.j2
├── config.ini              # Global configuration
├── deploy.sh               # Deployment script
└── README.md              # This file
```

##  Infrastructure Components

### AWS Resources Created:
- **VPC** with public subnet
- **3 x t2.medium instances** (1 master + 2 workers)
- **Security Group** for MPI communication
- **Load Balancer** for dashboard access
- **Placement Group** for low latency
- **NFS** for shared storage

### Instance Configuration:
- **Instance Type:** t2.medium (2 vCPUs, 4GB RAM)
- **Storage:** 20GB encrypted GP3
- **Network:** Enhanced networking enabled
- **Cost:** ~$0.14/hour for 3 nodes

##  Optimal Configuration

For t2.medium instances:
- **MPI Processes:** 3 (one per node)
- **OpenMP Threads:** 2 (per process)
- **Total Cores:** 6
- **Total Memory:** 12GB

##  Monitoring

### Web Dashboard
Access the cluster dashboard at: `http://<load-balancer-dns>`

Features:
- Real-time cluster status
- Node health monitoring
- Resource utilization
- Quick actions

### Command Line
```bash
# SSH to master node
ssh -i your-key.pem ubuntu@<master-ip>

# Monitor cluster
/shared/monitor_cluster.sh

# Check worker health
/shared/worker_health.sh
```

##  Running the Search Engine

```bash
# Connect to master node
ssh -i your-key.pem ubuntu@<master-ip>

# Run search engine
/shared/run_cluster.sh "search terms here"

# Example with parameters
/shared/run_cluster.sh -i input.txt -o results.txt
```

##  Configuration Files

### terraform.tfvars
```hcl
aws_region    = "us-east-1"
instance_type = "t2.medium"
cluster_size  = 3
key_name      = "your-key-pair-name"
project_name  = "hpc-search-engine"
```

### config.ini
Global configuration for deployment and runtime settings.

##  Customization

### Scaling the Cluster
```bash
# Edit terraform.tfvars
cluster_size = 5  # Increase to 5 nodes

# Redeploy
./deploy.sh deploy
```

### Different Instance Types
```bash
# Edit terraform.tfvars for more power
instance_type = "c5.large"  # 2 vCPUs, 4GB RAM, better network
# or
instance_type = "c5.xlarge" # 4 vCPUs, 8GB RAM
```

### Different Regions
```bash
# Edit terraform.tfvars
aws_region = "us-west-2"  # Change region
```

##  Cost Optimization

### Current Cost (t2.medium)
- **3 nodes:** ~$0.14/hour
- **24 hours:** ~$3.36/day
- **Monthly:** ~$100/month

### Cost-Saving Tips
1. **Use Spot Instances:** Up to 90% savings
2. **Auto-shutdown:** Implement idle detection
3. **Schedule:** Only run when needed
4. **Right-size:** Use smaller instances for development

### Spot Instance Configuration
```bash
# Add to terraform.tfvars
use_spot_instances = true
spot_price = "0.02"  # ~50% savings
```

##  Troubleshooting

### Common Issues

1. **SSH Connection Failed**
   ```bash
   # Check security group rules
   # Ensure your IP is allowed
   # Verify key pair name
   ```

2. **Instances Not Communicating**
   ```bash
   # Check placement group
   # Verify NFS mounts
   # Test network connectivity
   ```

3. **Build Failures**
   ```bash
   # Check dependencies
   # Verify shared directory access
   # Review build logs
   ```

### Debug Commands
```bash
# Check cluster status
./deploy.sh status

# SSH to master and check logs
ssh -i key.pem ubuntu@<master-ip>
tail -f /var/log/hpc-setup.log

# Check Ansible output
ansible-playbook -i inventory.yml site.yml -v
```

##  Cleanup

```bash
# Destroy all resources
./deploy.sh cleanup

# This will:
# - Terminate all EC2 instances
# - Delete VPC and networking
# - Remove security groups
# - Clean up all AWS resources
```

## 📈 Performance Tuning

### Network Optimization
- **Placement Groups:** Enabled for low latency
- **Enhanced Networking:** SR-IOV enabled
- **TCP Settings:** Optimized for MPI

### Memory Optimization
- **OpenMP Stack:** Limited to 512K
- **Memory Allocation:** Optimized for 4GB nodes
- **Swap:** Minimal usage

### CPU Optimization
- **Process Binding:** Enabled
- **NUMA Awareness:** Configured
- **Thread Affinity:** Optimized

##  Security

### Features Enabled
- **Encryption:** EBS volumes encrypted
- **Security Groups:** Minimal required ports
- **SSH Keys:** Secure key-based authentication
- **Updates:** Automatic security updates
- **Monitoring:** Health checks and logging

### Best Practices
- Use IAM roles instead of access keys
- Rotate SSH keys regularly
- Monitor resource usage
- Enable CloudTrail for auditing

##  Additional Resources

- [AWS HPC Guide](https://aws.amazon.com/hpc/)
- [MPI Best Practices](https://www.open-mpi.org/faq/)
- [OpenMP Guide](https://www.openmp.org/specifications/)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
