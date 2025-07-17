#  Complete AWS Deployment Guide for HPC Search Engine

## Overview
Deploy your Hybrid MPI+OpenMP search engine on AWS using 3 t2.medium instances for parallel processing.

**Cluster Configuration:**
- **3 x t2.medium instances** (2 vCPUs, 4GB RAM each)
- **Total: 6 cores, 12GB RAM**
- **Cost: ~$0.14/hour (~$3.36/day)**
- **MPI Processes: 3** (one per node)
- **OpenMP Threads: 2** (per process)

---

## 📋 Prerequisites

### 1. Install Required Tools

**Windows (PowerShell):**
```powershell
# Install AWS CLI
winget install Amazon.AWSCLI

# Install Terraform
winget install Hashicorp.Terraform

# Install Ansible (requires Python)
pip install ansible

# Install jq for JSON processing
winget install jqlang.jq
```

**Linux/macOS:**
```bash
# AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip && sudo ./aws/install

# Terraform
wget https://releases.hashicorp.com/terraform/1.6.0/terraform_1.6.0_linux_amd64.zip
unzip terraform_1.6.0_linux_amd64.zip && sudo mv terraform /usr/local/bin/

# Ansible
pip install ansible

# jq
sudo apt-get install jq  # Ubuntu/Debian
brew install jq          # macOS
```

### 2. Configure AWS Credentials

```bash
aws configure
# Enter your:
# - AWS Access Key ID
# - AWS Secret Access Key  
# - Default region: us-east-1
# - Default output format: json
```

### 3. Create AWS Key Pair

```bash
# Create a new key pair
aws ec2 create-key-pair \
    --key-name hpc-search-key \
    --query 'KeyMaterial' \
    --output text > ~/.ssh/hpc-search-key.pem

# Set correct permissions
chmod 400 ~/.ssh/hpc-search-key.pem
```

---

##  Step-by-Step Deployment

### Step 1: Prepare Your Code

```bash
# Navigate to your project
cd "c:\Users\chand\Desktop\HPC\High-Performance-Parallel-Search-Engine\Hybrid Version"

# On Linux/macOS, make deployment script executable
# chmod +x deploy_aws.sh
# On Windows, scripts are executable by default

# Update Terraform configuration
cd ../cloudInfra/terraform
```

Edit `terraform.tfvars`:
```hcl
aws_region    = "us-east-1"
instance_type = "t2.medium"
cluster_size  = 3
key_name      = "hpc-search-key"  # Your actual key name
project_name  = "hpc-search-engine"
```

### Step 2: Deploy the Cluster

```bash
# Go back to Hybrid Version directory
cd "../../Hybrid Version"

# Deploy everything (infrastructure + application)
./deploy_aws.sh deploy
```

This will:
1.  Build AWS-optimized search engine
2.  Create AWS infrastructure (VPC, instances, security groups)
3.  Configure cluster with Ansible
4.  Deploy your application
5.  Set up monitoring dashboard

### Step 3: Connect to Your Cluster

```bash
# SSH to master node (IP provided in deployment output)
ssh -i ~/.ssh/hpc-search-key.pem ubuntu@<MASTER_IP>

# Verify cluster is ready
/shared/run_aws_cluster.sh monitor
```

---

##  Running Your Search Engine

### Basic Commands (on master node)

```bash
# 1. Crawl and Index a Website
/shared/run_aws_cluster.sh -c "https://medium.com/@lpramithamj" -d 2 -p 50

# 2. Crawl Medium Profile
/shared/run_aws_cluster.sh -m "@lpramithamj"

# 3. Search Query
/shared/run_aws_cluster.sh -q "artificial intelligence machine learning"

# 4. Advanced Crawling with Custom Parameters
/shared/run_aws_cluster.sh -c "https://example.com" -d 3 -p 100 -v

# 5. Performance Monitoring
/shared/run_aws_cluster.sh monitor
```

### Example Workflows

**🌐 Medium Profile Analysis:**
```bash
# Crawl medium profile and analyze content
/shared/run_aws_cluster.sh -m "@lpramithamj" -d 2 -p 30

# Search for specific topics
/shared/run_aws_cluster.sh -q "machine learning algorithms"
/shared/run_aws_cluster.sh -q "data science python"
```

** Website Analysis:**
```bash
# Crawl company website
/shared/run_aws_cluster.sh -c "https://company.com" -d 2 -p 50

# Search for specific information
/shared/run_aws_cluster.sh -q "products services solutions"
```

** Research and Discovery:**
```bash
# Multi-step analysis
/shared/run_aws_cluster.sh -c "https://research-site.com" -d 3 -p 100
/shared/run_aws_cluster.sh -q "recent developments innovations"
```

---

##  Monitoring and Performance

### Web Dashboard
Access the cluster dashboard:
```
http://<LOAD_BALANCER_DNS>
```
Features:
- Real-time cluster status
- Node health monitoring  
- Resource utilization
- Performance metrics

### Command Line Monitoring

```bash
# Real-time cluster monitor
/shared/monitor_cluster.sh

# Check individual node health
/shared/worker_health.sh

# View performance logs
tail -f /shared/logs/aws_run_*.log

# Check metrics
cat /shared/aws_hybrid_metrics.csv
```

### Performance Tuning

```bash
# Run with verbose output
/shared/run_aws_cluster.sh -c "https://example.com" -v

# Custom thread count (if needed)
/shared/run_aws_cluster.sh -c "https://example.com" -t 1

# Monitor resource usage during execution
htop  # On master node
```

---

##  Cost Management

### Current Costs
- **3 x t2.medium:** $0.0464/hour each = $0.139/hour total
- **Load Balancer:** ~$0.0225/hour
- **EBS Storage:** ~$0.002/hour
- **Total:** ~$0.164/hour (~$3.94/day)

### Cost Optimization Tips

1. **Use Spot Instances (up to 70% savings):**
   ```bash
   # Edit terraform.tfvars
   use_spot_instances = true
   spot_price = "0.02"
   ```

2. **Auto-shutdown when idle:**
   ```bash
   # On master node
   sudo crontab -e
   # Add: 0 2 * * * /usr/local/bin/auto-shutdown.sh
   ```

3. **Destroy cluster when not needed:**
   ```bash
   ./deploy_aws.sh cleanup
   ```

---

##  Troubleshooting

### Common Issues

**1. SSH Connection Failed**
```bash
# Check security group rules
aws ec2 describe-security-groups --group-names hpc-search-sg

# Verify key permissions
ls -la ~/.ssh/hpc-search-key.pem  # Should be -r--------
```

**2. MPI Not Working**
```bash
# On master node, test MPI
mpirun -np 3 --hostfile /shared/hostfile hostname

# Check node connectivity
ping hpc-worker-1
ping hpc-worker-2
```

**3. Out of Memory Errors**
```bash
# Check memory usage
free -h

# Reduce page count
/shared/run_aws_cluster.sh -c "https://example.com" -p 25

# Monitor during execution
watch free -h
```

**4. Slow Performance**
```bash
# Check network connectivity
ping -c 10 8.8.8.8

# Monitor CPU usage
htop

# Check disk space
df -h /shared
```

### Debug Commands

```bash
# Check cluster status
./deploy_aws.sh info

# View deployment logs
tail -f /var/log/hpc-setup.log

# Test MPI functionality
mpirun -np 3 --hostfile /shared/hostfile echo "MPI Test from $(hostname)"

# Check OpenMP
echo | gcc -fopenmp -E -dM - | grep _OPENMP

# Verify shared storage
ls -la /shared/
```

---

##  Performance Benchmarks

### Expected Performance (t2.medium cluster)

| Workload | Pages | Time | Throughput |
|----------|-------|------|------------|
| Medium Profile | 30 | ~2-3 min | 10-15 pages/min |
| Website Crawl | 50 | ~3-5 min | 10-17 pages/min |
| Search Query | - | ~1-2 sec | Sub-second response |

### Optimization Results

**Serial vs Parallel Performance:**
- **Serial (1 core):** ~5 pages/min
- **Parallel (6 cores):** ~15 pages/min
- **Speedup:** ~3x (accounting for network overhead)

---

##  Cleanup

### Temporary Cleanup
```bash
# Stop applications but keep infrastructure
ssh -i ~/.ssh/hpc-search-key.pem ubuntu@<MASTER_IP>
sudo systemctl stop hpc-dashboard
```

### Complete Cleanup
```bash
# Destroy all AWS resources
./deploy_aws.sh cleanup

# This removes:
# - All EC2 instances
# - VPC and networking
# - Security groups  
# - Load balancer
# - All associated costs stop
```

---

##  Advanced Usage

### Custom Configurations

**1. Scale the Cluster:**
```bash
# Edit terraform.tfvars
cluster_size = 5  # Scale to 5 nodes

# Redeploy
./deploy_aws.sh deploy
```

**2. Use Larger Instances:**
```bash
# Edit terraform.tfvars
instance_type = "c5.large"  # 2 vCPUs, 4GB, better network

# Redeploy
./deploy_aws.sh deploy
```

**3. Different Regions:**
```bash
# Edit terraform.tfvars
aws_region = "us-west-2"

# Redeploy
./deploy_aws.sh deploy
```

### Integration with CI/CD

```bash
# Automated deployment script
#!/bin/bash
./deploy_aws.sh deploy
/shared/run_aws_cluster.sh -c "$CRAWL_URL" -q "$SEARCH_TERMS" > results.txt
./deploy_aws.sh cleanup
```

---

##  Additional Resources

- **AWS HPC Guide:** https://aws.amazon.com/hpc/
- **MPI Tutorial:** https://mpitutorial.com/
- **OpenMP Guide:** https://www.openmp.org/
- **Terraform AWS Provider:** https://registry.terraform.io/providers/hashicorp/aws/latest

---

## 🆘 Support

If you encounter issues:

1. **Check the logs:** `/shared/logs/` and `/var/log/hpc-setup.log`
2. **Verify prerequisites:** All tools installed and AWS configured
3. **Test connectivity:** SSH, ping between nodes
4. **Monitor resources:** Memory, CPU, disk space
5. **Review configuration:** Terraform and Ansible files

**Quick Health Check:**
```bash
# Run this on master node
/shared/run_aws_cluster.sh -q "test" -v
```

This setup provides a robust, cost-effective HPC environment for your search engine with full parallel processing capabilities!
