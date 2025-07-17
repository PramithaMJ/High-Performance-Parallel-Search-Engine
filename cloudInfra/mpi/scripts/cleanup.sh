#!/bin/bash

# Cleanup script for MPI cluster resources
# This script safely destroys the AWS infrastructure

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

# Default values
FORCE=false
BACKUP_DATA=true

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

# Function to check if cluster exists
check_cluster_exists() {
    if [ ! -f "$TERRAFORM_DIR/terraform.tfstate" ]; then
        print_warning "No Terraform state found. Cluster may not exist or was deployed elsewhere."
        return 1
    fi
    
    if [ ! -f "$TERRAFORM_DIR/outputs.json" ]; then
        print_warning "No cluster outputs found. Attempting to refresh state..."
        cd "$TERRAFORM_DIR"
        terraform refresh || {
            print_error "Failed to refresh Terraform state"
            return 1
        }
        terraform output -json > outputs.json || {
            print_error "Failed to get Terraform outputs"
            return 1
        }
    fi
    
    return 0
}

# Function to backup cluster data
backup_cluster_data() {
    if [ "$BACKUP_DATA" = false ]; then
        print_status "Skipping data backup as requested"
        return 0
    fi
    
    print_status "Backing up cluster data..."
    
    # Get cluster information
    if ! check_cluster_exists; then
        print_warning "Cannot backup data - cluster information not available"
        return 0
    fi
    
    local master_ip=$(jq -r '.master_public_ip.value' "$TERRAFORM_DIR/outputs.json" 2>/dev/null)
    local s3_bucket=$(jq -r '.s3_bucket_name.value' "$TERRAFORM_DIR/outputs.json" 2>/dev/null)
    local cluster_name=$(jq -r '.cluster_info.value.cluster_name' "$TERRAFORM_DIR/outputs.json" 2>/dev/null)
    
    if [ "$master_ip" = "null" ] || [ -z "$master_ip" ]; then
        print_warning "Cannot get master IP for backup"
        return 0
    fi
    
    # Check if master is accessible
    if ! ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no ubuntu@$master_ip "echo 'Connected'" >/dev/null 2>&1; then
        print_warning "Cannot connect to master node for backup"
        return 0
    fi
    
    # Create backup directory
    local backup_dir="/tmp/mpi-cluster-backup-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$backup_dir"
    
    print_status "Downloading data to $backup_dir..."
    
    # Backup shared data
    ssh -o StrictHostKeyChecking=no ubuntu@$master_ip "
        cd /shared
        sudo tar -czf /tmp/shared-backup.tar.gz \
            --exclude='*.o' \
            --exclude='*.so' \
            --exclude='bin/*' \
            dataset/ results/ logs/ *.sh hostfile 2>/dev/null || true
    " && {
        scp -o StrictHostKeyChecking=no ubuntu@$master_ip:/tmp/shared-backup.tar.gz "$backup_dir/"
        print_success "Shared data backed up"
    } || {
        print_warning "Failed to backup shared data"
    }
    
    # Backup logs
    ssh -o StrictHostKeyChecking=no ubuntu@$master_ip "
        sudo tar -czf /tmp/logs-backup.tar.gz /var/log/mpi-*.log /var/log/user-data.log 2>/dev/null || true
    " && {
        scp -o StrictHostKeyChecking=no ubuntu@$master_ip:/tmp/logs-backup.tar.gz "$backup_dir/"
        print_success "System logs backed up"
    } || {
        print_warning "Failed to backup system logs"
    }
    
    # Copy outputs.json for reference
    cp "$TERRAFORM_DIR/outputs.json" "$backup_dir/cluster-outputs.json" 2>/dev/null || true
    
    # Upload backup to S3 if bucket exists
    if [ "$s3_bucket" != "null" ] && [ -n "$s3_bucket" ]; then
        print_status "Uploading backup to S3..."
        aws s3 cp "$backup_dir" "s3://$s3_bucket/backups/$(basename $backup_dir)/" --recursive || {
            print_warning "Failed to upload backup to S3"
        }
    fi
    
    print_success "Backup completed: $backup_dir"
}

# Function to show cluster information before cleanup
show_cluster_info() {
    print_status "Current cluster information:"
    
    if check_cluster_exists; then
        cd "$TERRAFORM_DIR"
        terraform show -json | jq -r '
            .values.root_module.resources[] |
            select(.type == "aws_instance") |
            .values |
            "Instance: \(.tags.Name // "Unknown") - \(.instance_type) - \(.public_ip // .private_ip)"
        ' 2>/dev/null || {
            print_warning "Could not parse cluster information"
        }
        
        echo ""
        print_status "Estimated resources to be destroyed:"
        terraform plan -destroy 2>/dev/null | grep -E "(Plan:|will be destroyed)" || {
            print_warning "Could not generate destruction plan"
        }
    else
        print_warning "No cluster information available"
    fi
}

# Function to perform cleanup
cleanup_cluster() {
    print_status "Starting cluster cleanup..."
    
    if ! check_cluster_exists; then
        print_warning "No active cluster found to cleanup"
        return 0
    fi
    
    cd "$TERRAFORM_DIR"
    
    # Show what will be destroyed
    if [ "$FORCE" = false ]; then
        print_status "Generating destruction plan..."
        terraform plan -destroy
        
        echo ""
        print_warning "This will permanently destroy all cluster resources!"
        print_warning "Make sure you have backed up any important data."
        echo ""
        read -p "Are you sure you want to continue? (yes/no): " confirm
        
        if [ "$confirm" != "yes" ]; then
            print_status "Cleanup cancelled by user"
            return 0
        fi
    fi
    
    # Perform the actual destruction
    print_status "Destroying infrastructure..."
    
    if [ "$FORCE" = true ]; then
        terraform destroy -auto-approve
    else
        terraform destroy
    fi
    
    # Clean up local files
    print_status "Cleaning up local files..."
    rm -f terraform.tfvars
    rm -f tfplan
    rm -f outputs.json
    
    print_success "Cluster cleanup completed successfully"
}

# Function to cleanup specific resources if Terraform fails
emergency_cleanup() {
    print_warning "Performing emergency cleanup of AWS resources..."
    
    local cluster_name="$1"
    if [ -z "$cluster_name" ]; then
        cluster_name="mpi-search-engine"
    fi
    
    print_status "Searching for resources with cluster name: $cluster_name"
    
    # Get AWS region
    local aws_region=$(aws configure get region 2>/dev/null || echo "us-east-1")
    
    # Terminate EC2 instances
    print_status "Terminating EC2 instances..."
    local instance_ids=$(aws ec2 describe-instances \
        --region "$aws_region" \
        --filters "Name=tag:Project,Values=MPI-Search-Engine" "Name=instance-state-name,Values=running,stopped" \
        --query 'Reservations[].Instances[].InstanceId' \
        --output text 2>/dev/null || echo "")
    
    if [ -n "$instance_ids" ]; then
        aws ec2 terminate-instances --region "$aws_region" --instance-ids $instance_ids || {
            print_error "Failed to terminate some instances"
        }
        print_success "Initiated termination of instances: $instance_ids"
    else
        print_status "No instances found to terminate"
    fi
    
    # Delete load balancers
    print_status "Deleting load balancers..."
    local lb_arns=$(aws elbv2 describe-load-balancers \
        --region "$aws_region" \
        --query "LoadBalancers[?contains(LoadBalancerName, '$cluster_name')].LoadBalancerArn" \
        --output text 2>/dev/null || echo "")
    
    for arn in $lb_arns; do
        aws elbv2 delete-load-balancer --region "$aws_region" --load-balancer-arn "$arn" || {
            print_warning "Failed to delete load balancer: $arn"
        }
    done
    
    # Clean up security groups (after instances are terminated)
    print_status "Waiting for instances to terminate before cleaning security groups..."
    sleep 60
    
    local sg_ids=$(aws ec2 describe-security-groups \
        --region "$aws_region" \
        --filters "Name=tag:Project,Values=MPI-Search-Engine" \
        --query 'SecurityGroups[].GroupId' \
        --output text 2>/dev/null || echo "")
    
    for sg_id in $sg_ids; do
        aws ec2 delete-security-group --region "$aws_region" --group-id "$sg_id" 2>/dev/null || {
            print_warning "Failed to delete security group: $sg_id (may still be in use)"
        }
    done
    
    print_warning "Emergency cleanup completed. Some resources may require manual removal."
    print_warning "Please check the AWS console for any remaining resources."
}

# Function to display help
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Cleanup MPI Search Engine cluster on AWS"
    echo ""
    echo "Options:"
    echo "  --force               Skip confirmation prompts"
    echo "  --no-backup           Skip data backup"
    echo "  --emergency CLUSTER   Emergency cleanup by cluster name"
    echo "  --show-info           Show cluster info and exit"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                           # Interactive cleanup with backup"
    echo "  $0 --force --no-backup       # Fast cleanup without backup"
    echo "  $0 --emergency my-cluster    # Emergency cleanup by name"
    echo "  $0 --show-info               # Just show cluster information"
}

# Main function
main() {
    local emergency_cluster=""
    local show_info_only=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --force)
                FORCE=true
                shift
                ;;
            --no-backup)
                BACKUP_DATA=false
                shift
                ;;
            --emergency)
                emergency_cluster="$2"
                shift 2
                ;;
            --show-info)
                show_info_only=true
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
    
    print_status "MPI Search Engine Cluster Cleanup Tool"
    
    # Handle emergency cleanup
    if [ -n "$emergency_cluster" ]; then
        emergency_cleanup "$emergency_cluster"
        exit 0
    fi
    
    # Show info only
    if [ "$show_info_only" = true ]; then
        show_cluster_info
        exit 0
    fi
    
    # Normal cleanup process
    show_cluster_info
    
    if [ "$BACKUP_DATA" = true ]; then
        backup_cluster_data
    fi
    
    cleanup_cluster
    
    print_success "Cleanup process completed!"
    print_status "All AWS resources have been destroyed."
    
    if [ "$BACKUP_DATA" = true ]; then
        print_status "Data backups are available in /tmp/mpi-cluster-backup-*"
    fi
}

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
