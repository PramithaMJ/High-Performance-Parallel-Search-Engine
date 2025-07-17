#!/bin/bash

# Main deployment script for MPI Search Engine on AWS
# This script orchestrates the entire deployment process

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TERRAFORM_DIR="$PROJECT_ROOT/terraform"
CONFIG_DIR="$PROJECT_ROOT/config"

# Default values
CLUSTER_NAME="mpi-search-engine"
AWS_REGION="us-east-1"
WORKER_COUNT="4"
INSTANCE_TYPE="c5n.large"
USE_SPOT="true"
ENVIRONMENT="dev"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if required tools are installed
    local tools=("terraform" "aws" "jq")
    for tool in "${tools[@]}"; do
        if ! command -v $tool &> /dev/null; then
            print_error "$tool is not installed. Please install it first."
            exit 1
        fi
    done
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        print_error "AWS credentials not configured. Please run 'aws configure' first."
        exit 1
    fi
    
    # Check if SSH key exists
    if [ ! -f ~/.ssh/id_rsa.pub ]; then
        print_warning "SSH key not found. Generating new key pair..."
        ssh-keygen -t rsa -b 2048 -f ~/.ssh/id_rsa -N ""
    fi
    
    print_success "Prerequisites check completed"
}

# Function to initialize Terraform
init_terraform() {
    print_status "Initializing Terraform..."
    
    cd "$TERRAFORM_DIR"
    terraform init
    
    print_success "Terraform initialized"
}

# Function to create terraform.tfvars file
create_tfvars() {
    print_status "Creating Terraform variables file..."
    
    cat > "$TERRAFORM_DIR/terraform.tfvars" << EOF
# AWS Configuration
aws_region = "$AWS_REGION"

# Cluster Configuration
cluster_name = "$CLUSTER_NAME"
environment = "$ENVIRONMENT"
owner = "$(whoami)"

# EC2 Configuration
master_instance_type = "c5n.xlarge"
worker_instance_type = "$INSTANCE_TYPE"
worker_count = $WORKER_COUNT
use_spot_instances = $USE_SPOT

# Security Configuration
public_key_path = "~/.ssh/id_rsa.pub"
allowed_ssh_cidrs = ["$(curl -s https://checkip.amazonaws.com)/32"]

# Monitoring
enable_monitoring = true
log_retention_days = 14

# Auto-shutdown (for cost optimization)
auto_shutdown_enabled = true
idle_timeout_minutes = 60

# Web interface
enable_web_interface = true
web_port = 8080
EOF

    print_success "Terraform variables created"
}

# Function to plan Terraform deployment
plan_deployment() {
    print_status "Planning Terraform deployment..."
    
    cd "$TERRAFORM_DIR"
    terraform plan -out=tfplan
    
    print_success "Terraform plan created"
}

# Function to apply Terraform deployment
apply_deployment() {
    print_status "Applying Terraform deployment..."
    
    cd "$TERRAFORM_DIR"
    terraform apply tfplan
    
    print_success "Infrastructure deployed successfully"
}

# Function to get deployment outputs
get_outputs() {
    print_status "Getting deployment outputs..."
    
    cd "$TERRAFORM_DIR"
    
    # Get outputs in JSON format
    terraform output -json > "$PROJECT_ROOT/outputs.json"
    
    # Extract key information
    MASTER_IP=$(terraform output -raw master_public_ip)
    LOAD_BALANCER_URL=$(terraform output -raw load_balancer_url)
    SSH_COMMAND=$(terraform output -raw ssh_command)
    
    print_success "Deployment information saved to outputs.json"
    
    # Display key information
    echo ""
    echo "================================================================"
    echo "                 DEPLOYMENT COMPLETED SUCCESSFULLY"
    echo "================================================================"
    echo ""
    echo "Master Node SSH Access:"
    echo "  $SSH_COMMAND"
    echo ""
    echo "Web Interface:"
    echo "  $LOAD_BALANCER_URL"
    echo ""
    echo "Cluster Information:"
    terraform output cluster_info
    echo ""
    echo "================================================================"
}

# Function to wait for cluster to be ready
wait_for_cluster() {
    print_status "Waiting for cluster to be ready..."
    
    local master_ip="$1"
    local max_attempts=60
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        print_status "Checking cluster status... attempt $attempt/$max_attempts"
        
        if ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no ubuntu@$master_ip "test -f /var/log/master-setup-complete" 2>/dev/null; then
            print_success "Master node is ready!"
            break
        fi
        
        if [ $attempt -eq $max_attempts ]; then
            print_error "Cluster setup timeout. Please check the instances manually."
            return 1
        fi
        
        sleep 30
        ((attempt++))
    done
    
    # Wait a bit more for worker nodes
    print_status "Waiting for worker nodes to join cluster..."
    sleep 60
    
    # Check cluster status
    ssh -o StrictHostKeyChecking=no ubuntu@$master_ip "cat /shared/hostfile" || true
}

# Function to setup cluster configuration
setup_cluster() {
    print_status "Setting up cluster configuration..."
    
    local master_ip="$1"
    
    # Copy SSH keys to shared storage
    ssh -o StrictHostKeyChecking=no ubuntu@$master_ip "sudo -u mpiuser mkdir -p /shared/.ssh"
    ssh -o StrictHostKeyChecking=no ubuntu@$master_ip "sudo -u mpiuser cp /home/mpiuser/.ssh/id_rsa.pub /shared/.ssh/"
    
    # Copy MPI application source code
    print_status "Uploading MPI application source code..."
    scp -o StrictHostKeyChecking=no -r "$PROJECT_ROOT/../../MPI Version/"* ubuntu@$master_ip:/tmp/mpi-source/
    ssh -o StrictHostKeyChecking=no ubuntu@$master_ip "sudo cp -r /tmp/mpi-source/* /shared/mpi-search-engine/ && sudo chown -R mpiuser:mpiuser /shared/mpi-search-engine/"
    
    # Build the application
    ssh -o StrictHostKeyChecking=no ubuntu@$master_ip "cd /shared/mpi-search-engine && sudo -u mpiuser make clean && sudo -u mpiuser make all"
    
    print_success "Cluster configuration completed"
}

# Function to run a test search
test_cluster() {
    print_status "Running test search on the cluster..."
    
    local master_ip="$1"
    
    # Create a simple test dataset if it doesn't exist
    ssh -o StrictHostKeyChecking=no ubuntu@$master_ip "
        if [ ! -f /shared/dataset/test.txt ]; then
            sudo -u mpiuser mkdir -p /shared/dataset
            echo 'This is a test document for the MPI search engine. It contains sample text for testing parallel search capabilities.' | sudo -u mpiuser tee /shared/dataset/test.txt
            echo 'Another test document with different content for comprehensive search testing.' | sudo -u mpiuser tee /shared/dataset/test2.txt
        fi
    "
    
    # Run test search
    print_status "Executing test search for 'test'..."
    ssh -o StrictHostKeyChecking=no ubuntu@$master_ip "cd /shared && sudo -u mpiuser ./run-search.sh 'test'" || {
        print_warning "Test search failed, but cluster should still be functional"
    }
    
    print_success "Test completed"
}

# Function to display help
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Deploy MPI Search Engine cluster on AWS EC2"
    echo ""
    echo "Options:"
    echo "  -n, --cluster-name NAME    Cluster name (default: mpi-search-engine)"
    echo "  -r, --region REGION        AWS region (default: us-east-1)"
    echo "  -w, --workers COUNT        Number of worker nodes (default: 4)"
    echo "  -t, --instance-type TYPE   Worker instance type (default: c5n.large)"
    echo "  -s, --spot                 Use spot instances (default: true)"
    echo "  -e, --environment ENV      Environment name (default: dev)"
    echo "  --skip-test               Skip cluster testing"
    echo "  --plan-only               Only create and show Terraform plan"
    echo "  -h, --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Deploy with defaults"
    echo "  $0 -n my-cluster -w 8 -t c5n.xlarge  # Custom configuration"
    echo "  $0 --plan-only                       # Just show the plan"
}

# Function to cleanup on script exit
cleanup() {
    if [ $? -ne 0 ]; then
        print_error "Deployment failed. You may need to clean up resources manually."
        print_warning "Run 'terraform destroy' in $TERRAFORM_DIR to clean up."
    fi
}

# Main deployment function
main() {
    # Set up cleanup trap
    trap cleanup EXIT
    
    # Parse command line arguments
    SKIP_TEST=false
    PLAN_ONLY=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -n|--cluster-name)
                CLUSTER_NAME="$2"
                shift 2
                ;;
            -r|--region)
                AWS_REGION="$2"
                shift 2
                ;;
            -w|--workers)
                WORKER_COUNT="$2"
                shift 2
                ;;
            -t|--instance-type)
                INSTANCE_TYPE="$2"
                shift 2
                ;;
            -s|--spot)
                USE_SPOT="true"
                shift
                ;;
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            --skip-test)
                SKIP_TEST=true
                shift
                ;;
            --plan-only)
                PLAN_ONLY=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    print_status "Starting MPI Search Engine cluster deployment..."
    print_status "Cluster Name: $CLUSTER_NAME"
    print_status "AWS Region: $AWS_REGION"
    print_status "Worker Count: $WORKER_COUNT"
    print_status "Instance Type: $INSTANCE_TYPE"
    print_status "Use Spot Instances: $USE_SPOT"
    print_status "Environment: $ENVIRONMENT"
    
    # Execute deployment steps
    check_prerequisites
    init_terraform
    create_tfvars
    plan_deployment
    
    if [ "$PLAN_ONLY" = true ]; then
        print_success "Plan created successfully. Review the plan in $TERRAFORM_DIR/tfplan"
        exit 0
    fi
    
    apply_deployment
    get_outputs
    
    # Get master IP for further operations
    cd "$TERRAFORM_DIR"
    MASTER_IP=$(terraform output -raw master_public_ip)
    
    wait_for_cluster "$MASTER_IP"
    setup_cluster "$MASTER_IP"
    
    if [ "$SKIP_TEST" = false ]; then
        test_cluster "$MASTER_IP"
    fi
    
    print_success "Deployment completed successfully!"
    print_status "Your MPI Search Engine cluster is ready to use."
    print_status "Web interface: $(terraform output -raw load_balancer_url)"
    print_status "SSH access: $(terraform output -raw ssh_command)"
    
    # Disable cleanup trap on successful completion
    trap - EXIT
}

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
