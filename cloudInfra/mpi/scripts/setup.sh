#!/bin/bash

# Setup script for MPI Cloud Infrastructure
# This script prepares the environment and makes all scripts executable

set -e

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

print_status "Setting up MPI Cloud Infrastructure..."

# Make all scripts executable
print_status "Making scripts executable..."
chmod +x "$PROJECT_ROOT/scripts"/*.sh
chmod +x "$PROJECT_ROOT/terraform"/*.tf 2>/dev/null || true

# Create necessary directories
print_status "Creating directories..."
mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$PROJECT_ROOT/backups"
mkdir -p "$PROJECT_ROOT/monitoring"

# Check prerequisites
print_status "Checking prerequisites..."

# Check if required tools are installed
tools=("terraform" "aws" "jq" "ssh" "scp")
missing_tools=()

for tool in "${tools[@]}"; do
    if ! command -v $tool &> /dev/null; then
        missing_tools+=("$tool")
    fi
done

if [ ${#missing_tools[@]} -ne 0 ]; then
    print_warning "Missing required tools: ${missing_tools[*]}"
    echo "Please install the missing tools:"
    echo ""
    echo "Ubuntu/Debian:"
    echo "  sudo apt update"
    echo "  sudo apt install awscli jq openssh-client"
    echo "  # Install Terraform from https://www.terraform.io/downloads.html"
    echo ""
    echo "macOS:"
    echo "  brew install awscli jq terraform"
    echo ""
    echo "Windows:"
    echo "  # Install AWS CLI from https://aws.amazon.com/cli/"
    echo "  # Install Terraform from https://www.terraform.io/downloads.html"
    echo "  # Install jq from https://stedolan.github.io/jq/download/"
    echo ""
else
    print_success "All required tools are available"
fi

# Check AWS configuration
if aws sts get-caller-identity &> /dev/null; then
    print_success "AWS credentials are configured"
    AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
    AWS_REGION=$(aws configure get region || echo "us-east-1")
    print_status "AWS Account: $AWS_ACCOUNT"
    print_status "AWS Region: $AWS_REGION"
else
    print_warning "AWS credentials not configured"
    echo "Please configure AWS credentials:"
    echo "  aws configure"
    echo ""
fi

# Check SSH key
if [ -f ~/.ssh/id_rsa.pub ]; then
    print_success "SSH public key found"
else
    print_warning "SSH public key not found"
    echo "Generate SSH key pair:"
    echo "  ssh-keygen -t rsa -b 2048 -f ~/.ssh/id_rsa"
    echo ""
fi

# Create sample configuration
print_status "Creating sample configuration..."
if [ ! -f "$PROJECT_ROOT/config/local-config.ini" ]; then
    cp "$PROJECT_ROOT/config/aws-config.ini" "$PROJECT_ROOT/config/local-config.ini"
    print_success "Sample configuration created: config/local-config.ini"
    print_warning "Please review and customize the configuration before deployment"
fi

# Display usage information
echo ""
echo "================================================================"
echo "          MPI Cloud Infrastructure Setup Complete"
echo "================================================================"
echo ""
echo "Quick Start Guide:"
echo ""
echo "1. Configure AWS credentials (if not done):"
echo "   aws configure"
echo ""
echo "2. Review and customize configuration:"
echo "   vim $PROJECT_ROOT/config/local-config.ini"
echo ""
echo "3. Deploy the cluster:"
echo "   cd $PROJECT_ROOT"
echo "   ./scripts/deploy.sh"
echo ""
echo "4. Run a search:"
echo "   ./scripts/run-mpi.sh search 'your query here'"
echo ""
echo "5. Monitor the cluster:"
echo "   ./scripts/run-mpi.sh monitor"
echo ""
echo "6. Cleanup resources:"
echo "   ./scripts/cleanup.sh"
echo ""
echo "Available Scripts:"
echo "  deploy.sh      - Deploy the complete infrastructure"
echo "  run-mpi.sh     - Run searches and manage the cluster"
echo "  cleanup.sh     - Clean up AWS resources"
echo ""
echo "Configuration Files:"
echo "  config/aws-config.ini      - Default configuration"
echo "  config/local-config.ini    - Your customized configuration"
echo "  config/cluster-config.yml  - Cluster-specific settings"
echo ""
echo "Terraform Files:"
echo "  terraform/main.tf       - Main infrastructure definition"
echo "  terraform/variables.tf  - Variable definitions"
echo "  terraform/outputs.tf    - Output definitions"
echo ""
echo "Documentation:"
echo "  README.md - Complete documentation and examples"
echo ""
echo "================================================================"

if [ ${#missing_tools[@]} -eq 0 ] && aws sts get-caller-identity &> /dev/null; then
    echo ""
    print_success "Environment is ready for deployment!"
    echo ""
    echo "To deploy now with default settings:"
    echo "  cd $PROJECT_ROOT && ./scripts/deploy.sh"
    echo ""
else
    echo ""
    print_warning "Please complete the setup steps above before deployment"
    echo ""
fi
