#!/bin/bash

# AWS HPC Search Engine Runner
# Optimized for t2.medium cluster with MPI+OpenMP parallel processing

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration for t2.medium cluster
CLUSTER_CONFIG() {
    export AWS_NODES=3
    export CORES_PER_NODE=2
    export MPI_PROCESSES=$AWS_NODES
    export OMP_THREADS=$CORES_PER_NODE
    export TOTAL_CORES=$((AWS_NODES * CORES_PER_NODE))
    
    # AWS-optimized environment variables
    export OMP_NUM_THREADS=$OMP_THREADS
    export OMP_PROC_BIND=true
    export OMP_PLACES=cores
    export OMP_STACKSIZE=512K
    export OMP_WAIT_POLICY=passive
    
    # Memory management for 4GB RAM per node
    export MALLOC_TRIM_THRESHOLD=100000
    export MALLOC_MMAP_THRESHOLD=65536
    
    # MPI optimizations for AWS networking
    export OMPI_MCA_btl_tcp_if_include=eth0
    export OMPI_MCA_oob_tcp_if_include=eth0
    export OMPI_MCA_btl_vader_single_copy_mechanism=none
    
    # Disable CPU frequency scaling messages
    export OMPI_MCA_hwloc_base_binding_policy=core
}

# Print cluster banner
print_banner() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║                                                          ║"
    echo "║        🚀 AWS HPC Search Engine Cluster 🚀               ║"
    echo "║                                                          ║"
    echo "║  Hybrid MPI+OpenMP Parallel Processing                  ║"
    echo "║                                                          ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Display cluster configuration
show_config() {
    echo -e "${BLUE}📊 AWS Cluster Configuration:${NC}"
    echo "   • Instance Type: t2.medium"
    echo "   • Nodes: $AWS_NODES"
    echo "   • Cores per Node: $CORES_PER_NODE"
    echo "   • Total Cores: $TOTAL_CORES"
    echo "   • MPI Processes: $MPI_PROCESSES"
    echo "   • OpenMP Threads per Process: $OMP_THREADS"
    echo "   • Memory per Node: 4GB"
    echo "   • Total Memory: 12GB"
    echo ""
}

# Check cluster health
check_cluster() {
    echo -e "${BLUE}🔍 Checking cluster health...${NC}"
    
    # Check if hostfile exists
    if [ ! -f "/shared/hostfile" ]; then
        echo -e "${RED}❌ Hostfile not found!${NC}"
        exit 1
    fi
    
    # Check if search engine binary exists
    if [ ! -f "/shared/bin/search_engine" ]; then
        echo -e "${RED}❌ Search engine binary not found!${NC}"
        exit 1
    fi
    
    # Check MPI installation
    if ! command -v mpirun &> /dev/null; then
        echo -e "${RED}❌ MPI not found!${NC}"
        exit 1
    fi
    
    # Test node connectivity
    echo "   • Testing node connectivity..."
    while read -r line; do
        if [ -n "$line" ]; then
            node=$(echo $line | awk '{print $1}')
            if ping -c 1 -W 2 "$node" > /dev/null 2>&1; then
                echo "     ✅ $node: Online"
            else
                echo -e "     ${RED}❌ $node: Offline${NC}"
                exit 1
            fi
        fi
    done < /shared/hostfile
    
    echo -e "${GREEN}✅ Cluster health check passed!${NC}"
    echo ""
}

# Usage information
print_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -c URL     Crawl website starting from URL"
    echo "  -m USER    Crawl Medium profile (@username)"
    echo "  -d NUM     Maximum crawl depth (default: 2)"
    echo "  -p NUM     Maximum pages to crawl (default: 50)"
    echo "  -q QUERY   Run search query"
    echo "  -t NUM     Override OpenMP threads (default: 2)"
    echo "  -v         Verbose output"
    echo "  -h         Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 -c 'https://medium.com/@lpramithamj' -d 2 -p 30"
    echo "  $0 -m '@lpramithamj'"
    echo "  $0 -q 'artificial intelligence machine learning'"
    echo "  $0 -c 'https://example.com' -v"
}

# Parse command line arguments
parse_args() {
    CRAWL_URL=""
    MEDIUM_USER=""
    SEARCH_QUERY=""
    MAX_DEPTH=2
    MAX_PAGES=50
    VERBOSE=false
    CUSTOM_THREADS=""
    
    while getopts "c:m:d:p:q:t:vh" opt; do
        case $opt in
            c) CRAWL_URL="$OPTARG" ;;
            m) MEDIUM_USER="$OPTARG" ;;
            d) MAX_DEPTH="$OPTARG" ;;
            p) MAX_PAGES="$OPTARG" ;;
            q) SEARCH_QUERY="$OPTARG" ;;
            t) CUSTOM_THREADS="$OPTARG" ;;
            v) VERBOSE=true ;;
            h) print_usage; exit 0 ;;
            *) print_usage; exit 1 ;;
        esac
    done
    
    # Override threads if specified
    if [ -n "$CUSTOM_THREADS" ]; then
        export OMP_NUM_THREADS="$CUSTOM_THREADS"
        echo -e "${YELLOW}⚠️ Overriding OpenMP threads to $CUSTOM_THREADS${NC}"
    fi
}

# Run the search engine with MPI
run_search_engine() {
    local args=()
    
    # Build arguments
    if [ -n "$CRAWL_URL" ]; then
        args+=("-c" "$CRAWL_URL")
    fi
    
    if [ -n "$MEDIUM_USER" ]; then
        args+=("-m" "$MEDIUM_USER")
    fi
    
    if [ -n "$SEARCH_QUERY" ]; then
        args+=("-q" "$SEARCH_QUERY")
    fi
    
    args+=("-d" "$MAX_DEPTH")
    args+=("-p" "$MAX_PAGES")
    args+=("-np" "$MPI_PROCESSES")
    args+=("-t" "$OMP_THREADS")
    
    if [ "$VERBOSE" = true ]; then
        args+=("-v")
    fi
    
    echo -e "${BLUE}🏁 Starting parallel execution...${NC}"
    echo "   Command: mpirun -np $MPI_PROCESSES /shared/bin/search_engine ${args[*]}"
    echo ""
    
    # Change to shared directory
    cd /shared
    
    # Create output directory
    mkdir -p output logs
    
    # Set up logging
    local log_file="logs/aws_run_$(date +%Y%m%d_%H%M%S).log"
    
    # Run with MPI
    if [ "$VERBOSE" = true ]; then
        mpirun -np $MPI_PROCESSES \
               --hostfile /shared/hostfile \
               --map-by node \
               --bind-to core \
               --report-bindings \
               -x OMP_NUM_THREADS \
               -x OMP_PROC_BIND \
               -x OMP_PLACES \
               -x OMP_STACKSIZE \
               -x MALLOC_TRIM_THRESHOLD \
               -x OMPI_MCA_btl_tcp_if_include \
               -x OMPI_MCA_oob_tcp_if_include \
               /shared/bin/search_engine "${args[@]}" 2>&1 | tee "$log_file"
    else
        mpirun -np $MPI_PROCESSES \
               --hostfile /shared/hostfile \
               --map-by node \
               --bind-to core \
               -x OMP_NUM_THREADS \
               -x OMP_PROC_BIND \
               -x OMP_PLACES \
               -x OMP_STACKSIZE \
               /shared/bin/search_engine "${args[@]}"
    fi
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✅ Execution completed successfully!${NC}"
        
        # Show results if available
        if [ -f "aws_hybrid_metrics.csv" ]; then
            echo ""
            echo -e "${BLUE}📈 Performance Summary:${NC}"
            tail -5 "aws_hybrid_metrics.csv"
        fi
        
        if [ -d "output" ] && [ "$(ls -A output)" ]; then
            echo ""
            echo -e "${BLUE}📁 Output files:${NC}"
            ls -la output/
        fi
        
    else
        echo ""
        echo -e "${RED}❌ Execution failed with exit code $exit_code${NC}"
        if [ "$VERBOSE" = true ]; then
            echo "Check log file: $log_file"
        fi
        exit $exit_code
    fi
}

# Performance monitoring
show_performance() {
    echo -e "${BLUE}📊 Real-time Performance Monitor${NC}"
    echo "Press Ctrl+C to stop..."
    echo ""
    
    while true; do
        clear
        echo "=== AWS HPC Cluster Performance - $(date) ==="
        echo ""
        
        # CPU and Memory usage
        echo "Node Status:"
        while read -r line; do
            if [ -n "$line" ]; then
                node=$(echo $line | awk '{print $1}')
                if [ "$node" = "$(hostname)" ]; then
                    echo "  $node (local):"
                    echo "    CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
                    echo "    Mem: $(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100.0}')"
                    echo "    Load: $(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')"
                else
                    if ping -c 1 -W 1 "$node" > /dev/null 2>&1; then
                        echo "  $node: Online"
                    else
                        echo "  $node: Offline"
                    fi
                fi
            fi
        done < /shared/hostfile
        
        echo ""
        echo "System Resources:"
        echo "  Total Processes: $(ps aux | wc -l)"
        echo "  Network Connections: $(ss -tn | wc -l)"
        echo "  Disk Usage: $(df -h /shared | tail -1 | awk '{print $5}')"
        
        sleep 5
    done
}

# Main execution
main() {
    # Initialize cluster configuration
    CLUSTER_CONFIG
    
    # Parse arguments
    parse_args "$@"
    
    # Show banner
    print_banner
    show_config
    
    # Special case for performance monitoring
    if [ "$1" = "monitor" ]; then
        show_performance
        exit 0
    fi
    
    # Check if no arguments provided
    if [ $# -eq 0 ]; then
        print_usage
        exit 1
    fi
    
    # Health check
    check_cluster
    
    # Run the search engine
    run_search_engine
}

# Execute main function with all arguments
main "$@"
