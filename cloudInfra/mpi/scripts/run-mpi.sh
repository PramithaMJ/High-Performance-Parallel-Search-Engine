#!/bin/bash

# Script to run MPI search on the cluster
# This script provides various ways to execute distributed searches

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
PROCESSES=""
QUERY=""
DATASET_PATH="/shared/dataset"
RESULTS_PATH="/shared/results"
HOSTFILE="/shared/hostfile"
BINARY_PATH="/shared/mpi-search-engine/bin/search_engine"
OUTPUT_FORMAT="text"
VERBOSE=false
BENCHMARK=false

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

# Function to get cluster information
get_cluster_info() {
    if [ ! -f "$TERRAFORM_DIR/outputs.json" ]; then
        print_error "Cluster outputs not found. Please deploy the cluster first."
        exit 1
    fi
    
    MASTER_IP=$(jq -r '.master_public_ip.value' "$TERRAFORM_DIR/outputs.json")
    CLUSTER_NAME=$(jq -r '.cluster_info.value.cluster_name' "$TERRAFORM_DIR/outputs.json")
    TOTAL_SLOTS=$(jq -r '.cluster_info.value.total_slots' "$TERRAFORM_DIR/outputs.json")
    
    if [ "$MASTER_IP" = "null" ] || [ -z "$MASTER_IP" ]; then
        print_error "Could not get master IP from cluster outputs."
        exit 1
    fi
}

# Function to check cluster status
check_cluster_status() {
    print_status "Checking cluster status..."
    
    # Check if master is accessible
    if ! ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no ubuntu@$MASTER_IP "echo 'Connection successful'" 2>/dev/null; then
        print_error "Cannot connect to master node at $MASTER_IP"
        exit 1
    fi
    
    # Check if MPI binary exists
    if ! ssh -o StrictHostKeyChecking=no ubuntu@$MASTER_IP "test -f $BINARY_PATH" 2>/dev/null; then
        print_error "MPI search engine binary not found on cluster"
        print_warning "Please ensure the application is built: ssh ubuntu@$MASTER_IP 'cd /shared/mpi-search-engine && make all'"
        exit 1
    fi
    
    # Check hostfile
    if ! ssh -o StrictHostKeyChecking=no ubuntu@$MASTER_IP "test -f $HOSTFILE" 2>/dev/null; then
        print_error "MPI hostfile not found on cluster"
        exit 1
    fi
    
    # Display cluster info
    print_success "Cluster is accessible"
    ssh -o StrictHostKeyChecking=no ubuntu@$MASTER_IP "echo 'Active nodes:' && cat $HOSTFILE" 2>/dev/null || {
        print_warning "Could not read hostfile"
    }
}

# Function to upload dataset
upload_dataset() {
    local local_path="$1"
    
    if [ ! -d "$local_path" ] && [ ! -f "$local_path" ]; then
        print_error "Dataset path does not exist: $local_path"
        exit 1
    fi
    
    print_status "Uploading dataset to cluster..."
    
    # Create dataset directory on cluster
    ssh -o StrictHostKeyChecking=no ubuntu@$MASTER_IP "sudo -u mpiuser mkdir -p $DATASET_PATH"
    
    # Upload dataset
    if [ -d "$local_path" ]; then
        scp -o StrictHostKeyChecking=no -r "$local_path"/* ubuntu@$MASTER_IP:/tmp/dataset/
    else
        scp -o StrictHostKeyChecking=no "$local_path" ubuntu@$MASTER_IP:/tmp/dataset/
    fi
    
    # Move to shared storage with correct permissions
    ssh -o StrictHostKeyChecking=no ubuntu@$MASTER_IP "sudo cp -r /tmp/dataset/* $DATASET_PATH/ && sudo chown -R mpiuser:mpiuser $DATASET_PATH/"
    
    print_success "Dataset uploaded successfully"
}

# Function to run MPI search
run_mpi_search() {
    local query="$1"
    local processes="$2"
    
    print_status "Running MPI search for query: '$query'"
    print_status "Using $processes processes"
    
    # Prepare MPI command
    local mpi_cmd="mpirun"
    
    if [ -n "$processes" ]; then
        mpi_cmd="$mpi_cmd -np $processes"
    fi
    
    mpi_cmd="$mpi_cmd --hostfile $HOSTFILE"
    mpi_cmd="$mpi_cmd --map-by node"
    mpi_cmd="$mpi_cmd --bind-to core"
    
    if [ "$VERBOSE" = true ]; then
        mpi_cmd="$mpi_cmd --display-map"
    fi
    
    mpi_cmd="$mpi_cmd $BINARY_PATH"
    mpi_cmd="$mpi_cmd -q '$query'"
    mpi_cmd="$mpi_cmd -d $DATASET_PATH"
    
    if [ "$OUTPUT_FORMAT" = "json" ]; then
        mpi_cmd="$mpi_cmd --json"
    fi
    
    # Create results directory
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local result_file="$RESULTS_PATH/search_${timestamp}.log"
    
    ssh -o StrictHostKeyChecking=no ubuntu@$MASTER_IP "sudo -u mpiuser mkdir -p $RESULTS_PATH"
    
    # Run the search
    print_status "Executing: $mpi_cmd"
    
    if [ "$BENCHMARK" = true ]; then
        # Run with timing
        ssh -o StrictHostKeyChecking=no ubuntu@$MASTER_IP "
            cd /shared &&
            sudo -u mpiuser bash -c 'time $mpi_cmd' 2>&1 | tee $result_file
        "
    else
        # Normal run
        ssh -o StrictHostKeyChecking=no ubuntu@$MASTER_IP "
            cd /shared &&
            sudo -u mpiuser bash -c '$mpi_cmd' 2>&1 | tee $result_file
        "
    fi
    
    local exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -eq 0 ]; then
        print_success "Search completed successfully"
        print_status "Results saved to: $result_file"
    else
        print_error "Search failed with exit code: $exit_code"
        return $exit_code
    fi
}

# Function to run benchmark
run_benchmark() {
    print_status "Running MPI search benchmark..."
    
    local queries=("search" "algorithm" "parallel" "distributed" "performance")
    local process_counts=("2" "4" "8" "$TOTAL_SLOTS")
    
    local benchmark_dir="$RESULTS_PATH/benchmark_$(date +%Y%m%d_%H%M%S)"
    ssh -o StrictHostKeyChecking=no ubuntu@$MASTER_IP "sudo -u mpiuser mkdir -p $benchmark_dir"
    
    echo "Query,Processes,ExecutionTime,Throughput" > /tmp/benchmark_results.csv
    
    for query in "${queries[@]}"; do
        for procs in "${process_counts[@]}"; do
            print_status "Benchmarking: '$query' with $procs processes"
            
            local start_time=$(date +%s.%N)
            
            if run_mpi_search "$query" "$procs" >/dev/null 2>&1; then
                local end_time=$(date +%s.%N)
                local execution_time=$(echo "$end_time - $start_time" | bc)
                local throughput=$(echo "scale=2; 1 / $execution_time" | bc)
                
                echo "$query,$procs,$execution_time,$throughput" >> /tmp/benchmark_results.csv
                print_success "Completed in ${execution_time}s (throughput: $throughput queries/s)"
            else
                print_warning "Benchmark failed for '$query' with $procs processes"
                echo "$query,$procs,FAILED,0" >> /tmp/benchmark_results.csv
            fi
            
            sleep 2  # Brief pause between runs
        done
    done
    
    # Upload results to cluster
    scp -o StrictHostKeyChecking=no /tmp/benchmark_results.csv ubuntu@$MASTER_IP:$benchmark_dir/
    ssh -o StrictHostKeyChecking=no ubuntu@$MASTER_IP "sudo chown mpiuser:mpiuser $benchmark_dir/benchmark_results.csv"
    
    print_success "Benchmark completed. Results saved to: $benchmark_dir/benchmark_results.csv"
    
    # Display summary
    print_status "Benchmark Summary:"
    ssh -o StrictHostKeyChecking=no ubuntu@$MASTER_IP "sudo -u mpiuser column -t -s ',' $benchmark_dir/benchmark_results.csv"
}

# Function to monitor cluster performance
monitor_cluster() {
    print_status "Starting cluster monitoring (press Ctrl+C to stop)..."
    
    while true; do
        clear
        echo "================================================================"
        echo "           MPI Search Engine Cluster Monitor"
        echo "================================================================"
        echo "Cluster: $CLUSTER_NAME"
        echo "Time: $(date)"
        echo ""
        
        # Get cluster status
        ssh -o StrictHostKeyChecking=no ubuntu@$MASTER_IP "
            echo 'Active Nodes:'
            cat $HOSTFILE
            echo ''
            echo 'Resource Usage:'
            echo '  Master Node:'
            echo -n '    CPU: '; top -bn1 | grep 'Cpu(s)' | awk '{print \$2}'
            echo -n '    Memory: '; free | grep Mem | awk '{printf \"%.1f%%\", \$3/\$2 * 100.0}'
            echo ''
            echo '  Worker Nodes:'
            if ls /shared/metrics/worker-*.json >/dev/null 2>&1; then
                for file in /shared/metrics/worker-*.json; do
                    if [ -f \"\$file\" ]; then
                        echo -n '    '; basename \"\$file\" .json | sed 's/worker-//': 
                        jq -r '.ip + \" CPU:\" + (.cpu_usage|tostring) + \"% MEM:\" + (.memory_usage|tostring) + \"%\"' \"\$file\" 2>/dev/null || echo 'N/A'
                    fi
                done
            else
                echo '    No worker metrics available'
            fi
            echo ''
            echo 'Active MPI Processes:'
            ps aux | grep -E '(mpirun|search_engine)' | grep -v grep || echo '    None'
        " 2>/dev/null || {
            print_error "Could not connect to cluster"
            break
        }
        
        sleep 5
    done
}

# Function to show cluster logs
show_logs() {
    local log_type="$1"
    
    case $log_type in
        "master")
            ssh -o StrictHostKeyChecking=no ubuntu@$MASTER_IP "sudo tail -f /var/log/mpi-search-engine.log" 2>/dev/null || {
                print_warning "Master log not available, showing user-data log:"
                ssh -o StrictHostKeyChecking=no ubuntu@$MASTER_IP "sudo tail -f /var/log/user-data.log"
            }
            ;;
        "workers")
            ssh -o StrictHostKeyChecking=no ubuntu@$MASTER_IP "
                echo 'Worker Logs:'
                for file in /shared/logs/worker-*.log; do
                    if [ -f \"\$file\" ]; then
                        echo \"=== \$(basename \"\$file\") ===\"
                        tail -10 \"\$file\"
                        echo
                    fi
                done
            " 2>/dev/null || print_warning "Worker logs not available"
            ;;
        "search")
            ssh -o StrictHostKeyChecking=no ubuntu@$MASTER_IP "
                if ls $RESULTS_PATH/search_*.log >/dev/null 2>&1; then
                    latest=\$(ls -t $RESULTS_PATH/search_*.log | head -1)
                    echo \"Latest search log: \$latest\"
                    cat \"\$latest\"
                else
                    echo 'No search logs found'
                fi
            " 2>/dev/null
            ;;
        *)
            print_error "Unknown log type: $log_type"
            print_status "Available log types: master, workers, search"
            ;;
    esac
}

# Function to display help
show_help() {
    echo "Usage: $0 [OPTIONS] COMMAND"
    echo ""
    echo "Run MPI search on AWS EC2 cluster"
    echo ""
    echo "Commands:"
    echo "  search QUERY           Run distributed search for QUERY"
    echo "  benchmark              Run performance benchmark"
    echo "  monitor                Monitor cluster performance"
    echo "  status                 Show cluster status"
    echo "  logs TYPE              Show logs (master|workers|search)"
    echo "  upload PATH            Upload dataset from PATH"
    echo ""
    echo "Options:"
    echo "  -n, --processes NUM    Number of MPI processes to use"
    echo "  -f, --format FORMAT    Output format (text|json)"
    echo "  -v, --verbose          Verbose output"
    echo "  -b, --benchmark        Include timing information"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 search 'machine learning'              # Basic search"
    echo "  $0 -n 8 search 'parallel computing'       # Use 8 processes"
    echo "  $0 -f json search 'distributed systems'   # JSON output"
    echo "  $0 benchmark                              # Run benchmark"
    echo "  $0 monitor                                # Monitor cluster"
    echo "  $0 upload ./my-dataset/                   # Upload dataset"
}

# Main function
main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -n|--processes)
                PROCESSES="$2"
                shift 2
                ;;
            -f|--format)
                OUTPUT_FORMAT="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -b|--benchmark)
                BENCHMARK=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            search)
                COMMAND="search"
                QUERY="$2"
                shift 2
                ;;
            benchmark)
                COMMAND="benchmark"
                shift
                ;;
            monitor)
                COMMAND="monitor"
                shift
                ;;
            status)
                COMMAND="status"
                shift
                ;;
            logs)
                COMMAND="logs"
                LOG_TYPE="$2"
                shift 2
                ;;
            upload)
                COMMAND="upload"
                UPLOAD_PATH="$2"
                shift 2
                ;;
            *)
                print_error "Unknown option or command: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Get cluster information
    get_cluster_info
    
    # Execute command
    case $COMMAND in
        "search")
            if [ -z "$QUERY" ]; then
                print_error "Search query is required"
                show_help
                exit 1
            fi
            check_cluster_status
            run_mpi_search "$QUERY" "$PROCESSES"
            ;;
        "benchmark")
            check_cluster_status
            run_benchmark
            ;;
        "monitor")
            check_cluster_status
            monitor_cluster
            ;;
        "status")
            check_cluster_status
            print_success "Cluster is operational"
            ;;
        "logs")
            if [ -z "$LOG_TYPE" ]; then
                print_error "Log type is required"
                show_help
                exit 1
            fi
            show_logs "$LOG_TYPE"
            ;;
        "upload")
            if [ -z "$UPLOAD_PATH" ]; then
                print_error "Upload path is required"
                show_help
                exit 1
            fi
            upload_dataset "$UPLOAD_PATH"
            ;;
        *)
            print_error "Command is required"
            show_help
            exit 1
            ;;
    esac
}

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
