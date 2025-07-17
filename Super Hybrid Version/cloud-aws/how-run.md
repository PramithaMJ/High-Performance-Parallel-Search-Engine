Based on your existing Hybrid Version code, here's a complete guide to deploy and run your search engine on AWS with multiple nodes in parallel:

##  Complete AWS Deployment Guide

### 1. Update Your Configuration Files

First, update your existing configuration to work optimally in AWS:

````ini
[runtime]
# AWS-optimized settings for t2.medium (2 vCPUs, 4GB RAM per node)
default_mpi_processes = 3
default_omp_threads = 2
max_pages_per_process = 100
max_depth = 2
queue_size = 1000

[aws]
# AWS deployment settings
region = us-east-1
instance_type = t2.medium
cluster_size = 3
key_name = your-aws-key-pair-name

[performance]
# Optimized for limited memory
memory_limit_mb = 3500
stack_size_kb = 512
network_timeout = 30
````

### 2. Create AWS-Optimized Makefile

````makefile
# AWS-optimized Makefile for t2.medium instances
CC = mpicc
CXX = mpicxx

# Conservative optimization for t2.medium
CFLAGS = -O2 -fopenmp -Wall -std=c99 -pipe
CXXFLAGS = -O2 -fopenmp -Wall -std=c++11 -pipe

# Memory-conscious settings
CFLAGS += -DMAX_URLS=1000 -DMAX_PAGES=100 -DAWS_OPTIMIZED
CXXFLAGS += -DMAX_URLS=1000 -DMAX_PAGES=100 -DAWS_OPTIMIZED

# Libraries
LDFLAGS = -fopenmp -lcurl -lm

# Directories
SRCDIR = src
OBJDIR = obj
BINDIR = bin
INCDIR = include

# Include path
INCLUDES = -I$(INCDIR)

# Source files
SOURCES = $(wildcard $(SRCDIR)/*.c)
OBJECTS = $(SOURCES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)
TARGET = $(BINDIR)/search_engine

# Create directories
$(shell mkdir -p $(OBJDIR) $(BINDIR))

# Main target
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

# Compile objects
$(OBJDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# AWS-specific optimizations
aws-optimize:
	@echo "Applying AWS optimizations..."
	@sed -i 's/MAX_URLS 10000/MAX_URLS 1000/g' $(INCDIR)/*.h
	@sed -i 's/MAX_PAGES 1000/MAX_PAGES 100/g' $(INCDIR)/*.h

clean:
	rm -rf $(OBJDIR) $(BINDIR)

.PHONY: all clean aws-optimize
````

### 3. Update Main Code for AWS

````c
// ...existing code...

int main(int argc, char* argv[]) {
    // AWS-specific initialization
    #ifdef AWS_OPTIMIZED
    // Reduce memory footprint for t2.medium
    setenv("OMP_STACKSIZE", "512K", 1);
    setenv("MALLOC_TRIM_THRESHOLD", "100000", 1);
    #endif
    
    // Initialize MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI does not support threading\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    // AWS cluster info display
    if (mpi_rank == 0) {
        printf("\n╔══════════════════════════════════════════════╗\n");
        printf("║        AWS HPC Search Engine Cluster          ║\n");
        printf("╠══════════════════════════════════════════════╣\n");
        printf("║ AWS Region: us-east-1                        ║\n");
        printf("║ Instance Type: t2.medium                     ║\n");
        printf("║ MPI Processes: %-3d                           ║\n", mpi_size);
        printf("║ Cores per Node: 2                           ║\n");
        printf("║ Total Cores: %-3d                             ║\n", mpi_size * 2);
        printf("╚══════════════════════════════════════════════╝\n");
    }
    
    // ...existing code continues...
}
````

### 4. Create AWS Deployment Scripts

````bash
#!/bin/bash

echo " Setting up HPC Search Engine on AWS..."

# Step 1: Create infrastructure with Terraform
cd terraform
echo "📋 Creating AWS infrastructure..."
terraform init
terraform plan -var-file="terraform.tfvars"
terraform apply -auto-approve -var-file="terraform.tfvars"

# Get outputs
MASTER_IP=$(terraform output -raw master_public_ip)
WORKER_IPS=$(terraform output -json worker_public_ips | jq -r '.[]')
LB_DNS=$(terraform output -raw load_balancer_dns)

echo " Infrastructure created!"
echo "Master IP: $MASTER_IP"
echo "Load Balancer: $LB_DNS"

# Step 2: Wait for instances to be ready
echo "⏳ Waiting for instances to be ready..."
sleep 60

# Step 3: Create Ansible inventory
cd ../ansible
cat > inventory.yml << EOF
all:
  children:
    master:
      hosts:
        master:
          ansible_host: $MASTER_IP
          ansible_user: ubuntu
          ansible_ssh_private_key_file: ~/.ssh/your-key.pem
    workers:
      hosts:
EOF

i=1
for worker_ip in $WORKER_IPS; do
cat >> inventory.yml << EOF
        worker$i:
          ansible_host: $worker_ip
          ansible_user: ubuntu
          ansible_ssh_private_key_file: ~/.ssh/your-key.pem
EOF
((i++))
done

# Step 4: Run Ansible configuration
echo "⚙️ Configuring cluster with Ansible..."
ansible-playbook -i inventory.yml site.yml

echo " AWS HPC Cluster is ready!"
echo "🌐 Dashboard: http://$LB_DNS"
echo "🔗 SSH to master: ssh -i ~/.ssh/your-key.pem ubuntu@$MASTER_IP"
````

### 5. Create AWS-Optimized Run Script

````bash
#!/bin/bash

# AWS-optimized run script for t2.medium cluster
echo " Running HPC Search Engine on AWS Cluster..."

# AWS cluster configuration
AWS_NODES=3
CORES_PER_NODE=2
MPI_PROCESSES=$AWS_NODES
OMP_THREADS=$CORES_PER_NODE

echo " AWS Cluster Configuration:"
echo "  - Instance Type: t2.medium"
echo "  - Nodes: $AWS_NODES"
echo "  - Cores per Node: $CORES_PER_NODE"
echo "  - MPI Processes: $MPI_PROCESSES"
echo "  - OpenMP Threads: $OMP_THREADS"
echo "  - Total Cores: $((AWS_NODES * CORES_PER_NODE))"

# Create hostfile for AWS cluster
cat > /shared/hostfile << EOF
master slots=1
worker1 slots=1
worker2 slots=1
EOF

# Set AWS-optimized environment
export OMP_NUM_THREADS=$OMP_THREADS
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export OMP_STACKSIZE=512K

# AWS network optimization
export OMPI_MCA_btl_tcp_if_include=eth0
export OMPI_MCA_oob_tcp_if_include=eth0

# Build with AWS optimizations
cd /shared/search-engine
make -f Makefile.aws clean
make -f Makefile.aws aws-optimize
make -f Makefile.aws

echo "🏁 Starting parallel search..."

# Run with optimal AWS configuration
mpirun -np $MPI_PROCESSES \
       --hostfile /shared/hostfile \
       --map-by node \
       --bind-to core \
       -x OMP_NUM_THREADS \
       -x OMP_PROC_BIND \
       -x OMP_PLACES \
       -x OMP_STACKSIZE \
       ./bin/search_engine "$@"

echo " Search completed!"

# Display performance metrics
if [ -f "hybrid_metrics.csv" ]; then
    echo "📈 Performance Summary:"
    tail -5 hybrid_metrics.csv
fi
````

### 6. Update Crawler for AWS

````c
// ...existing code...

void crawl_parallel(const char* start_url, int depth, int max_pages, int num_threads) {
    // AWS-specific optimizations
    #ifdef AWS_OPTIMIZED
    // Reduce queue size for t2.medium memory constraints
    if (max_pages > 100) max_pages = 100;
    if (num_threads > 2) num_threads = 2;
    
    // AWS network timeout optimization
    curl_global_init(CURL_GLOBAL_DEFAULT);
    #endif
    
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    if (mpi_rank == 0) {
        printf("\n🌐 AWS Cluster Crawling Configuration:\n");
        printf("   - Max pages per node: %d\n", max_pages);
        printf("   - OpenMP threads: %d\n", num_threads);
        printf("   - Network timeout: 30s\n");
        printf("   - Memory limit: 3.5GB per node\n");
    }
    
    // ...existing crawling code continues with AWS optimizations...
    
    #pragma omp parallel num_threads(num_threads) shared(queue, depth, front, rear, queue_lock, \
                                                         total_pages_crawled, total_failed_downloads, \
                                                         thread_active, active_threads, thread_pages, mpi_rank, max_pages)
    {
        int thread_id = omp_get_thread_num();
        char aws_thread_id[64];
        snprintf(aws_thread_id, sizeof(aws_thread_id), "AWS-Node%d-Thread%d", mpi_rank, thread_id);
        
        // AWS-specific thread configuration
        #ifdef AWS_OPTIMIZED
        // Set conservative timeouts for t2.medium network
        CURL *curl = curl_easy_init();
        if (curl) {
            curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
            curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);
            curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME, 20L);
            curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT, 1000L);
        }
        #endif
        
        // ...existing parallel crawling logic...
    }
}
````

### 7. Complete Deployment Instructions

**Step 1: Prepare Your Local Environment**

```bash
# Clone and prepare your code
cd "c:\Users\chand\Desktop\HPC\High-Performance-Parallel-Search-Engine"
cd "Hybrid Version"

# Install AWS CLI and configure
aws configure
# Enter your AWS credentials

# Install Terraform
# Download from: https://www.terraform.io/downloads.html

# Install Ansible
pip install ansible
```

**Step 2: Create AWS Key Pair**

```bash
# Create a new key pair in AWS
aws ec2 create-key-pair \
    --key-name hpc-search-key \
    --query 'KeyMaterial' \
    --output text > ~/.ssh/hpc-search-key.pem

chmod 400 ~/.ssh/hpc-search-key.pem
```

**Step 3: Update Configuration**

```bash
# Edit the terraform variables
cd cloudInfra/terraform
nano terraform.tfvars

# Update with your key name:
key_name = "hpc-search-key"
```

**Step 4: Deploy to AWS**

```bash
# Make deployment script executable
chmod +x ../setup_aws.sh

# Deploy the cluster
../setup_aws.sh
```

**Step 5: Connect and Run**

```bash
# SSH to master node (use the IP from deployment output)
ssh -i ~/.ssh/hpc-search-key.pem ubuntu@<MASTER_IP>

# Run your search engine
/shared/run_aws_cluster.sh -c "https://medium.com/@lpramithamj" -q "artificial intelligence"

# Monitor the cluster
/shared/monitor_cluster.sh
```

### 8. Running Different Workloads

**Crawling and Indexing:**
```bash
# On master node
/shared/run_aws_cluster.sh -c "https://medium.com/@lpramithamj" -d 2 -p 50
```

**Search Queries:**
```bash
# Search with optimized parallel processing
/shared/run_aws_cluster.sh -q "machine learning algorithms"
```

**Benchmarking:**
```bash
# Run performance tests
cd /shared/search-engine/scripts
./performance_benchmark.sh > aws_benchmark_results.csv
```

### 9. Cost and Performance Monitoring

**Monitor Costs:**
```bash
# Check current AWS costs
aws ce get-cost-and-usage \
    --time-period Start=2025-07-17,End=2025-07-18 \
    --granularity DAILY \
    --metrics BlendedCost
```

**Performance Monitoring:**
- Access web dashboard at load balancer URL
- Check `/shared/monitor_cluster.sh` for real-time metrics
- Review logs in `/var/log/hpc-*.log`

### 10. Cleanup

```bash
# When finished, destroy the cluster to save costs
cd cloudInfra/terraform
terraform destroy -auto-approve
```

## Key Changes Made to Your Code:

1. **Memory Optimization**: Reduced MAX_URLS and MAX_PAGES for t2.medium
2. **AWS Compilation Flags**: Added AWS_OPTIMIZED preprocessor directive
3. **Network Timeouts**: Optimized for AWS networking
4. **Thread Configuration**: Limited to 2 threads per node (t2.medium has 2 vCPUs)
5. **MPI Hostfile**: Configured for 3-node cluster
6. **Environment Variables**: Set optimal OpenMP settings for AWS

This setup gives you a fully functional HPC search engine running on AWS with optimal configuration for t2.medium instances at approximately **$0.14/hour** for the entire 3-node cluster.
