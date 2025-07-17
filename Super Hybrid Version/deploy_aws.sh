#!/bin/bash

# AWS HPC Search Engine Deployment Script
# Deploys your hybrid MPI+OpenMP search engine on AWS t2.medium cluster

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CLOUD_INFRA_DIR="$PROJECT_ROOT/cloudInfra"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

info() {
    echo -e "${PURPLE}[INFO]${NC} $1"
}

# Banner
print_banner() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║                                                          ║"
    echo "║         AWS HPC Search Engine Deployment             ║"
    echo "║                                                          ║"
    echo "║  Hybrid MPI+OpenMP Parallel Search Engine               ║"
    echo "║  Target: 3x t2.medium instances                         ║"
    echo "║  Total: 6 cores, 12GB RAM                               ║"
    echo "║  Cost: ~$0.14/hour                                      ║"
    echo "║                                                          ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    local missing_tools=()
    
    # Check Terraform
    if ! command -v terraform &> /dev/null; then
        missing_tools+=("terraform")
    fi
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        missing_tools+=("aws-cli")
    fi
    
    # Check Ansible
    if ! command -v ansible &> /dev/null; then
        missing_tools+=("ansible")
    fi
    
    # Check jq
    if ! command -v jq &> /dev/null; then
        missing_tools+=("jq")
    fi
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        error "Missing required tools: ${missing_tools[*]}"
        echo ""
        echo "Installation instructions:"
        echo "  - Terraform: https://www.terraform.io/downloads.html"
        echo "  - AWS CLI: https://aws.amazon.com/cli/"
        echo "  - Ansible: pip install ansible"
        echo "  - jq: apt-get install jq (Linux) or brew install jq (macOS)"
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        error "AWS credentials not configured"
        echo "Please run: aws configure"
        exit 1
    fi
    
    # Check if cloud infrastructure directory exists
    if [ ! -d "$CLOUD_INFRA_DIR" ]; then
        error "Cloud infrastructure directory not found: $CLOUD_INFRA_DIR"
        echo "Please ensure you have the cloudInfra directory with Terraform and Ansible configs"
        exit 1
    fi
    
    success "All prerequisites met!"
}

# Build the search engine
build_search_engine() {
    log "Building AWS-optimized search engine..."
    
    cd "$SCRIPT_DIR"
    
    # Clean previous builds
    if [ -f "Makefile.aws" ]; then
        make -f Makefile.aws clean 2>/dev/null || true
    fi
    
    # Build with AWS optimizations
    log "Compiling with AWS optimizations..."
    make -f Makefile.aws production
    
    if [ ! -f "bin/search_engine" ]; then
        error "Failed to build search engine"
        exit 1
    fi
    
    success "Search engine built successfully!"
    
    # Display binary info
    info "Binary information:"
    ls -lh bin/search_engine
    file bin/search_engine
}

# Deploy infrastructure
deploy_infrastructure() {
    log "Deploying AWS infrastructure..."
    
    cd "$CLOUD_INFRA_DIR/terraform"
    
    # Check terraform.tfvars
    if [ ! -f "terraform.tfvars" ]; then
        error "terraform.tfvars not found"
        echo "Creating template terraform.tfvars..."
        cat > terraform.tfvars << EOF
aws_region    = "us-east-1"
instance_type = "t2.medium"
cluster_size  = 3
key_name      = "your-aws-key-pair-name"  # CHANGE THIS!
project_name  = "hpc-search-engine"
EOF
        error "Please edit terraform.tfvars with your AWS key pair name"
        exit 1
    fi
    
    # Initialize and apply Terraform
    log "Initializing Terraform..."
    terraform init
    
    log "Planning deployment..."
    terraform plan -var-file="terraform.tfvars"
    
    log "Applying infrastructure..."
    terraform apply -auto-approve -var-file="terraform.tfvars"
    
    # Get outputs
    MASTER_IP=$(terraform output -raw master_public_ip)
    WORKER_IPS=$(terraform output -json worker_public_ips)
    LB_DNS=$(terraform output -raw load_balancer_dns)
    
    success "Infrastructure deployed!"
    info "Master IP: $MASTER_IP"
    info "Load Balancer: http://$LB_DNS"
    
    # Save outputs
    cat > ../aws_outputs.json << EOF
{
    "master_ip": "$MASTER_IP",
    "worker_ips": $WORKER_IPS,
    "load_balancer_dns": "$LB_DNS",
    "deployment_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
    
    cd - > /dev/null
}

# Configure cluster with Ansible
configure_cluster() {
    log "Configuring cluster with Ansible..."
    
    cd "$CLOUD_INFRA_DIR/ansible"
    
    # Get deployment info
    if [ ! -f "../aws_outputs.json" ]; then
        error "AWS outputs not found. Deploy infrastructure first."
        exit 1
    fi
    
    MASTER_IP=$(jq -r '.master_ip' ../aws_outputs.json)
    WORKER_IPS=$(jq -r '.worker_ips[]' ../aws_outputs.json)
    
    # Wait for instances to be ready
    log "Waiting for instances to be SSH-ready..."
    for ip in $MASTER_IP $WORKER_IPS; do
        info "Checking $ip..."
        while ! nc -z "$ip" 22 2>/dev/null; do
            sleep 5
        done
        success "$ip is ready"
    done
    
    # Generate dynamic inventory
    log "Generating Ansible inventory..."
    cat > inventory.yml << EOF
all:
  children:
    hpc_cluster:
      children:
        master:
          hosts:
            hpc-master:
              ansible_host: $MASTER_IP
              ansible_user: ubuntu
              ansible_ssh_private_key_file: ~/.ssh/\${key_name}.pem
              node_type: master
        workers:
          hosts:
EOF
    
    i=1
    for worker_ip in $WORKER_IPS; do
        cat >> inventory.yml << EOF
            hpc-worker-$i:
              ansible_host: $worker_ip
              ansible_user: ubuntu
              ansible_ssh_private_key_file: ~/.ssh/\${key_name}.pem
              node_type: worker
              worker_index: $i
EOF
        ((i++))
    done
    
    cat >> inventory.yml << EOF
      vars:
        ansible_ssh_common_args: '-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null'
        project_name: hpc-search-engine
        mpi_processes: 3
        omp_threads: 2
        shared_dir: /shared
        mpi_hostfile: /shared/hostfile
EOF
    
    # Run Ansible playbook
    log "Running Ansible configuration..."
    ansible-playbook -i inventory.yml site.yml -v
    
    cd - > /dev/null
    success "Cluster configured successfully!"
}

# Copy search engine to cluster
deploy_application() {
    log "Deploying search engine to cluster..."
    
    # Get master IP
    MASTER_IP=$(jq -r '.master_ip' "$CLOUD_INFRA_DIR/aws_outputs.json")
    KEY_NAME=$(grep 'key_name' "$CLOUD_INFRA_DIR/terraform/terraform.tfvars" | cut -d'"' -f2)
    KEY_FILE="$HOME/.ssh/${KEY_NAME}.pem"
    
    if [ ! -f "$KEY_FILE" ]; then
        error "SSH key not found: $KEY_FILE"
        exit 1
    fi
    
    # Create deployment package
    log "Creating deployment package..."
    cd "$SCRIPT_DIR"
    
    # Create temporary deployment directory
    DEPLOY_DIR="/tmp/hpc-search-deploy"
    rm -rf "$DEPLOY_DIR"
    mkdir -p "$DEPLOY_DIR"
    
    # Copy essential files
    cp -r bin "$DEPLOY_DIR/"
    cp -r data "$DEPLOY_DIR/" 2>/dev/null || mkdir "$DEPLOY_DIR/data"
    cp config.aws.ini "$DEPLOY_DIR/config.ini"
    cp Makefile.aws "$DEPLOY_DIR/Makefile"
    
    # Create run script
    cat > "$DEPLOY_DIR/run_aws.sh" << 'EOF'
#!/bin/bash

# AWS HPC Search Engine Run Script
echo " Running HPC Search Engine on AWS Cluster"
echo "============================================="

# Configuration
NODES=3
CORES_PER_NODE=2
MPI_PROCESSES=$NODES
OMP_THREADS=$CORES_PER_NODE

echo " Cluster Configuration:"
echo "  - Instance Type: t2.medium"
echo "  - Nodes: $NODES"
echo "  - Cores per Node: $CORES_PER_NODE"
echo "  - MPI Processes: $MPI_PROCESSES"
echo "  - OpenMP Threads: $OMP_THREADS"
echo "  - Total Cores: $((NODES * CORES_PER_NODE))"

# Set environment variables
export OMP_NUM_THREADS=$OMP_THREADS
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export OMP_STACKSIZE=512K

# AWS network optimizations
export OMPI_MCA_btl_tcp_if_include=eth0
export OMPI_MCA_oob_tcp_if_include=eth0

echo ""
echo "🏁 Starting search engine..."

# Run with MPI
mpirun -np $MPI_PROCESSES \
       --hostfile /shared/hostfile \
       --map-by node \
       --bind-to core \
       --report-bindings \
       -x OMP_NUM_THREADS \
       -x OMP_PROC_BIND \
       -x OMP_PLACES \
       -x OMP_STACKSIZE \
       /shared/bin/search_engine "$@"

echo ""
echo " Execution completed!"
EOF
    
    chmod +x "$DEPLOY_DIR/run_aws.sh"
    
    # Transfer to cluster
    log "Transferring files to master node..."
    scp -i "$KEY_FILE" -o StrictHostKeyChecking=no -r "$DEPLOY_DIR"/* "ubuntu@$MASTER_IP:/shared/"
    
    # Make sure binary is executable
    ssh -i "$KEY_FILE" -o StrictHostKeyChecking=no "ubuntu@$MASTER_IP" "chmod +x /shared/bin/search_engine"
    
    success "Application deployed to cluster!"
    
    # Cleanup
    rm -rf "$DEPLOY_DIR"
}

# Display cluster information
show_cluster_info() {
    log "Displaying cluster information..."
    
    # Get deployment info
    MASTER_IP=$(jq -r '.master_ip' "$CLOUD_INFRA_DIR/aws_outputs.json")
    LB_DNS=$(jq -r '.load_balancer_dns' "$CLOUD_INFRA_DIR/aws_outputs.json")
    KEY_NAME=$(grep 'key_name' "$CLOUD_INFRA_DIR/terraform/terraform.tfvars" | cut -d'"' -f2)
    
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                    DEPLOYMENT COMPLETE!               ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    echo -e "${BLUE} Cluster Information:${NC}"
    echo "   • Instance Type: t2.medium"
    echo "   • Cluster Size: 3 nodes"
    echo "   • Total Cores: 6"
    echo "   • Total Memory: 12GB"
    echo "   • Cost: ~$0.14/hour"
    echo ""
    
    echo -e "${BLUE} Access Information:${NC}"
    echo "   • Master SSH: ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${MASTER_IP}"
    echo "   • Dashboard: http://${LB_DNS}"
    echo ""
    
    echo -e "${BLUE} Quick Start Commands:${NC}"
    echo "   # Connect to cluster:"
    echo "   ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${MASTER_IP}"
    echo ""
    echo "   # Run search engine (on master node):"
    echo "   /shared/run_aws.sh -c 'https://medium.com/@lpramithamj' -q 'artificial intelligence'"
    echo ""
    echo "   # Monitor cluster:"
    echo "   /shared/monitor_cluster.sh"
    echo ""
    echo "   # Check cluster status:"
    echo "   /shared/dashboard/server.py"
    echo ""
    
    echo -e "${BLUE} Example Workloads:${NC}"
    echo "   # Crawl and index a website:"
    echo "   /shared/run_aws.sh -c 'https://medium.com/@lpramithamj' -d 2 -p 50"
    echo ""
    echo "   # Search with parallel processing:"
    echo "   /shared/run_aws.sh -q 'machine learning algorithms'"
    echo ""
    echo "   # Benchmark performance:"
    echo "   cd /shared && ./run_aws.sh -q 'test search' > benchmark_results.txt"
    echo ""
    
    echo -e "${YELLOW} Cost Management:${NC}"
    echo "   • Current cost: ~$0.14/hour"
    echo "   • Daily cost: ~$3.36"
    echo "   • Remember to destroy cluster when done: terraform destroy"
    echo ""
}

# Cleanup function
cleanup() {
    log "Cleaning up AWS resources..."
    
    cd "$CLOUD_INFRA_DIR/terraform"
    terraform destroy -auto-approve
    
    success "Cleanup completed!"
}

# Main deployment function
deploy() {
    print_banner
    check_prerequisites
    build_search_engine
    deploy_infrastructure
    configure_cluster
    deploy_application
    show_cluster_info
}

# Main script logic
case "${1:-deploy}" in
    "deploy")
        deploy
        ;;
    "cleanup")
        cleanup
        ;;
    "info")
        show_cluster_info
        ;;
    "build")
        build_search_engine
        ;;
    *)
        echo "Usage: $0 [deploy|cleanup|info|build]"
        echo ""
        echo "Commands:"
        echo "  deploy   - Deploy the complete HPC cluster (default)"
        echo "  cleanup  - Destroy the HPC cluster"
        echo "  info     - Show cluster information"
        echo "  build    - Build the search engine only"
        exit 1
        ;;
esac
