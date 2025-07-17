#!/bin/bash

# HPC Cluster Deployment Script for AWS
# This script deploys a t2.medium cluster using Terraform and configures it with Ansible

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TERRAFORM_DIR="$SCRIPT_DIR/terraform"
ANSIBLE_DIR="$SCRIPT_DIR/ansible"
CONFIG_FILE="$SCRIPT_DIR/config.ini"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if Terraform is installed
    if ! command -v terraform &> /dev/null; then
        error "Terraform is not installed. Please install Terraform first."
        error "Visit: https://www.terraform.io/downloads.html"
        exit 1
    fi
    
    # Check if AWS CLI is installed
    if ! command -v aws &> /dev/null; then
        error "AWS CLI is not installed. Please install AWS CLI first."
        error "Visit: https://aws.amazon.com/cli/"
        exit 1
    fi
    
    # Check if Ansible is installed
    if ! command -v ansible &> /dev/null; then
        error "Ansible is not installed. Please install Ansible first."
        error "Run: pip install ansible"
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        error "AWS credentials not configured. Please run 'aws configure'"
        exit 1
    fi
    
    success "All prerequisites met!"
}

# Deploy infrastructure with Terraform
deploy_infrastructure() {
    log "Deploying infrastructure with Terraform..."
    
    cd "$TERRAFORM_DIR"
    
    # Check if terraform.tfvars exists
    if [ ! -f "terraform.tfvars" ]; then
        error "terraform.tfvars not found. Please create it with your configuration."
        echo "Example content:"
        echo 'key_name = "your-aws-key-pair"'
        echo 'aws_region = "us-east-1"'
        echo 'instance_type = "t2.medium"'
        echo 'cluster_size = 3'
        exit 1
    fi
    
    # Initialize Terraform
    log "Initializing Terraform..."
    terraform init
    
    # Plan deployment
    log "Planning deployment..."
    terraform plan -out=tfplan
    
    # Apply deployment
    log "Applying deployment..."
    terraform apply tfplan
    
    # Get outputs
    log "Getting deployment outputs..."
    MASTER_IP=$(terraform output -raw master_public_ip)
    WORKER_IPS=$(terraform output -json worker_public_ips | jq -r '.[]')
    
    success "Infrastructure deployed successfully!"
    log "Master IP: $MASTER_IP"
    log "Worker IPs: $(echo $WORKER_IPS | tr '\n' ' ')"
    
    # Save outputs for Ansible
    cat > "$ANSIBLE_DIR/terraform_outputs.json" << EOF
{
    "master_public_ip": "$MASTER_IP",
    "worker_public_ips": $(terraform output -json worker_public_ips),
    "master_private_ip": "$(terraform output -raw master_private_ip)",
    "worker_private_ips": $(terraform output -json worker_private_ips)
}
EOF
    
    cd - > /dev/null
}

# Configure cluster with Ansible
configure_cluster() {
    log "Configuring cluster with Ansible..."
    
    cd "$ANSIBLE_DIR"
    
    # Check if private key exists
    KEY_NAME=$(grep 'key_name' "$TERRAFORM_DIR/terraform.tfvars" | cut -d'"' -f2)
    KEY_FILE="$HOME/.ssh/${KEY_NAME}.pem"
    
    if [ ! -f "$KEY_FILE" ]; then
        error "SSH key file not found: $KEY_FILE"
        error "Please ensure your AWS key pair is available in ~/.ssh/"
        exit 1
    fi
    
    # Generate dynamic inventory
    log "Generating Ansible inventory..."
    
    MASTER_IP=$(jq -r '.master_public_ip' terraform_outputs.json)
    WORKER_1_IP=$(jq -r '.worker_public_ips[0]' terraform_outputs.json)
    WORKER_2_IP=$(jq -r '.worker_public_ips[1]' terraform_outputs.json)
    
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
              ansible_ssh_private_key_file: $KEY_FILE
              node_type: master
        workers:
          hosts:
            hpc-worker-1:
              ansible_host: $WORKER_1_IP
              ansible_user: ubuntu
              ansible_ssh_private_key_file: $KEY_FILE
              node_type: worker
              worker_index: 1
            hpc-worker-2:
              ansible_host: $WORKER_2_IP
              ansible_user: ubuntu
              ansible_ssh_private_key_file: $KEY_FILE
              node_type: worker
              worker_index: 2
      vars:
        ansible_ssh_common_args: '-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null'
        project_name: hpc-search-engine
        mpi_processes: 3
        omp_threads: 2
        shared_dir: /shared
        mpi_hostfile: /shared/hostfile
EOF
    
    # Wait for instances to be ready
    log "Waiting for instances to be ready..."
    for ip in $MASTER_IP $WORKER_1_IP $WORKER_2_IP; do
        log "Waiting for $ip..."
        while ! ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -i "$KEY_FILE" ubuntu@$ip "echo 'Instance ready'" 2>/dev/null; do
            sleep 10
        done
        success "Instance $ip is ready"
    done
    
    # Run Ansible playbook
    log "Running Ansible playbook..."
    ansible-playbook -i inventory.yml site.yml -v
    
    cd - > /dev/null
    success "Cluster configuration completed!"
}

# Display cluster information
show_cluster_info() {
    log "Cluster deployment completed!"
    echo ""
    echo "=================== CLUSTER INFORMATION ==================="
    
    cd "$TERRAFORM_DIR"
    
    echo "Infrastructure Details:"
    echo "  - Cluster Size: $(terraform output -json cluster_info | jq -r '.cluster_size')"
    echo "  - Instance Type: $(terraform output -json cluster_info | jq -r '.instance_type')"
    echo "  - Total vCPUs: $(terraform output -json cluster_info | jq -r '.total_vcpus')"
    echo "  - Total Memory: $(terraform output -json cluster_info | jq -r '.total_memory_gb')GB"
    echo ""
    
    echo "Access Information:"
    echo "  - Master Node SSH: $(terraform output -raw ssh_command_master)"
    echo "  - Dashboard URL: http://$(terraform output -raw load_balancer_dns)"
    echo ""
    
    echo "Quick Start Commands:"
    echo "  # Connect to master node:"
    echo "  $(terraform output -raw ssh_command_master)"
    echo ""
    echo "  # Run search engine (on master node):"
    echo "  /shared/run_cluster.sh <search_terms>"
    echo ""
    echo "  # Monitor cluster:"
    echo "  /shared/monitor_cluster.sh"
    echo ""
    
    cd - > /dev/null
    
    echo "=========================================================="
    success "Deployment completed successfully!"
}

# Cleanup function
cleanup() {
    log "Cleaning up deployment..."
    
    cd "$TERRAFORM_DIR"
    terraform destroy -auto-approve
    
    success "Cleanup completed!"
}

# Main deployment function
deploy() {
    log "Starting HPC cluster deployment on AWS..."
    
    check_prerequisites
    deploy_infrastructure
    configure_cluster
    show_cluster_info
}

# Script usage
usage() {
    echo "Usage: $0 [deploy|cleanup|status]"
    echo ""
    echo "Commands:"
    echo "  deploy   - Deploy the HPC cluster"
    echo "  cleanup  - Destroy the HPC cluster"
    echo "  status   - Show cluster status"
    echo ""
    echo "Examples:"
    echo "  $0 deploy"
    echo "  $0 cleanup"
}

# Get cluster status
status() {
    log "Getting cluster status..."
    
    cd "$TERRAFORM_DIR"
    
    if [ ! -f "terraform.tfstate" ]; then
        warning "No terraform state found. Cluster may not be deployed."
        exit 1
    fi
    
    terraform show
}

# Main script logic
case "${1:-}" in
    "deploy")
        deploy
        ;;
    "cleanup")
        cleanup
        ;;
    "status")
        status
        ;;
    *)
        usage
        exit 1
        ;;
esac
