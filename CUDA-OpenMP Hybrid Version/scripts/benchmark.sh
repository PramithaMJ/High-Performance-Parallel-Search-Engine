#!/bin/bash

# CUDA+OpenMP Hybrid Search Engine Performance Benchmark Script
# Tests various configurations and generates performance reports

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
EXECUTABLE="$PROJECT_ROOT/bin/cuda_openmp_search_engine"
RESULTS_DIR="$PROJECT_ROOT/benchmark_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Colors for output
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
BLUE='\\033[0;34m'
NC='\\033[0m' # No Color

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

print_header() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    if [ ! -f "$EXECUTABLE" ]; then
        print_error "Executable not found: $EXECUTABLE"
        print_status "Please run 'make' to build the project first"
        exit 1
    fi
    
    # Check for CUDA
    if command -v nvidia-smi > /dev/null 2>&1; then
        print_success "NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1
    else
        print_warning "No NVIDIA GPU detected, will run CPU-only benchmarks"
    fi
    
    # Check for required tools
    local missing_tools=()
    for tool in bc awk; do
        if ! command -v "$tool" > /dev/null 2>&1; then
            missing_tools+=("$tool")
        fi
    done
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        print_error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Setup benchmark environment
setup_benchmark() {
    print_status "Setting up benchmark environment..."
    
    mkdir -p "$RESULTS_DIR"
    
    # Create result files
    SUMMARY_FILE="$RESULTS_DIR/benchmark_summary_$TIMESTAMP.csv"
    DETAILED_FILE="$RESULTS_DIR/benchmark_detailed_$TIMESTAMP.csv"
    LOG_FILE="$RESULTS_DIR/benchmark_log_$TIMESTAMP.log"
    
    # Write CSV headers
    echo "Configuration,Mode,Threads,GPU,Avg_Time_ms,Min_Time_ms,Max_Time_ms,Throughput_qps,GPU_Speedup" > "$SUMMARY_FILE"
    echo "Configuration,Query,Time_ms,Score,Processing_Location" > "$DETAILED_FILE"
    
    print_success "Benchmark environment ready"
    print_status "Results will be saved to: $RESULTS_DIR"
}

# Test queries for benchmarking
TEST_QUERIES=(
    "artificial intelligence"
    "machine learning algorithms"
    "deep neural networks"
    "natural language processing"
    "computer vision"
    "data mining techniques"
    "distributed systems"
    "parallel computing"
    "gpu acceleration"
    "high performance computing"
    "search algorithms"
    "information retrieval"
    "text processing"
    "document indexing"
    "ranking algorithms"
)

# Benchmark configurations
BENCHMARK_CONFIGS=(
    "CPU_1T:cpu:1:0"
    "CPU_2T:cpu:2:0"
    "CPU_4T:cpu:4:0"
    "CPU_8T:cpu:8:0"
    "CPU_16T:cpu:16:0"
    "GPU_ONLY:gpu:4:1"
    "HYBRID_CPU_HEAVY:hybrid:8:1:0.7"
    "HYBRID_BALANCED:hybrid:8:1:0.5"
    "HYBRID_GPU_HEAVY:hybrid:8:1:0.3"
    "AUTO_OPTIMIZED:auto:0:1:0.0"
)

# Run single benchmark configuration
run_benchmark_config() {
    local config_name="$1"
    local mode="$2"
    local threads="$3"
    local use_gpu="$4"
    local cpu_ratio="${5:-0.5}"
    
    print_status "Testing configuration: $config_name"
    
    local total_time=0
    local min_time=999999
    local max_time=0
    local successful_queries=0
    local baseline_time=0
    
    # Build command arguments
    local cmd_args=""
    cmd_args+=" --mode $mode"
    
    if [ "$threads" != "0" ]; then
        cmd_args+=" --threads $threads"
    fi
    
    if [ "$use_gpu" == "1" ]; then
        cmd_args+=" --gpu"
        if [ "$mode" == "hybrid" ]; then
            cmd_args+=" --ratio $cpu_ratio"
        fi
    else
        cmd_args+=" --no-gpu"
    fi
    
    print_status "Command arguments: $cmd_args"
    
    # Run each test query
    for query in "${TEST_QUERIES[@]}"; do
        print_status "  Testing query: '$query'"
        
        # Run the search with timing
        local start_time=$(date +%s.%N)
        local output
        if output=$($EXECUTABLE $cmd_args --query "$query" 2>&1); then
            local end_time=$(date +%s.%N)
            local query_time=$(echo "$end_time - $start_time" | bc)
            local query_time_ms=$(echo "$query_time * 1000" | bc)
            
            # Extract processing information from output
            local processing_location="Unknown"
            if echo "$output" | grep -q "Processed on: CPU"; then
                processing_location="CPU"
            elif echo "$output" | grep -q "Processed on: GPU"; then
                processing_location="GPU"
            elif echo "$output" | grep -q "Processed on: Hybrid"; then
                processing_location="Hybrid"
            fi
            
            # Update statistics
            total_time=$(echo "$total_time + $query_time" | bc)
            successful_queries=$((successful_queries + 1))
            
            # Update min/max times
            if (( $(echo "$query_time < $min_time" | bc -l) )); then
                min_time=$query_time
            fi
            if (( $(echo "$query_time > $max_time" | bc -l) )); then
                max_time=$query_time
            fi
            
            # Store baseline time for speedup calculation
            if [ "$config_name" == "CPU_1T" ]; then
                baseline_time=$(echo "$baseline_time + $query_time" | bc)
            fi
            
            # Write detailed results
            echo "$config_name,\"$query\",$query_time_ms,0.0,$processing_location" >> "$DETAILED_FILE"
            
            print_status "    Time: ${query_time_ms}ms, Location: $processing_location"
        else
            print_warning "    Query failed: $query"
            echo "$config_name,\"$query\",FAILED,0.0,ERROR" >> "$DETAILED_FILE"
        fi
    done
    
    # Calculate statistics
    if [ "$successful_queries" -gt 0 ]; then
        local avg_time=$(echo "scale=6; $total_time / $successful_queries" | bc)
        local avg_time_ms=$(echo "$avg_time * 1000" | bc)
        local min_time_ms=$(echo "$min_time * 1000" | bc)
        local max_time_ms=$(echo "$max_time * 1000" | bc)
        local throughput=$(echo "scale=2; $successful_queries / $total_time" | bc)
        
        # Calculate speedup (if baseline is available)
        local speedup="N/A"
        if [ -f "$RESULTS_DIR/baseline_time.tmp" ] && [ "$config_name" != "CPU_1T" ]; then
            local baseline=$(cat "$RESULTS_DIR/baseline_time.tmp")
            speedup=$(echo "scale=2; $baseline / $avg_time" | bc)
        elif [ "$config_name" == "CPU_1T" ]; then
            echo "$avg_time" > "$RESULTS_DIR/baseline_time.tmp"
            speedup="1.00"
        fi
        
        # Write summary results
        echo "$config_name,$mode,$threads,$use_gpu,$avg_time_ms,$min_time_ms,$max_time_ms,$throughput,$speedup" >> "$SUMMARY_FILE"
        
        print_success "  Configuration complete:"
        print_success "    Average time: ${avg_time_ms}ms"
        print_success "    Throughput: ${throughput} queries/sec"
        if [ "$speedup" != "N/A" ]; then
            print_success "    Speedup: ${speedup}x"
        fi
    else
        print_error "  All queries failed for configuration: $config_name"
        echo "$config_name,$mode,$threads,$use_gpu,FAILED,FAILED,FAILED,0.0,N/A" >> "$SUMMARY_FILE"
    fi
    
    echo "" # Add spacing between configurations
}

# Run all benchmark configurations
run_all_benchmarks() {
    print_header "Running All Benchmark Configurations"
    
    for config in "${BENCHMARK_CONFIGS[@]}"; do
        IFS=':' read -ra CONFIG_PARTS <<< "$config"
        local config_name="${CONFIG_PARTS[0]}"
        local mode="${CONFIG_PARTS[1]}"
        local threads="${CONFIG_PARTS[2]}"
        local use_gpu="${CONFIG_PARTS[3]}"
        local cpu_ratio="${CONFIG_PARTS[4]:-0.5}"
        
        # Skip GPU configurations if no GPU is available
        if [ "$use_gpu" == "1" ] && ! command -v nvidia-smi > /dev/null 2>&1; then
            print_warning "Skipping GPU configuration '$config_name' - no NVIDIA GPU detected"
            echo "$config_name,$mode,$threads,$use_gpu,SKIPPED,NO_GPU,SKIPPED,0.0,N/A" >> "$SUMMARY_FILE"
            continue
        fi
        
        run_benchmark_config "$config_name" "$mode" "$threads" "$use_gpu" "$cpu_ratio"
        
        # Small delay between configurations
        sleep 2
    done
    
    # Cleanup temporary files
    rm -f "$RESULTS_DIR/baseline_time.tmp"
}

# Generate performance report
generate_report() {
    print_header "Generating Performance Report"
    
    local report_file="$RESULTS_DIR/performance_report_$TIMESTAMP.txt"
    
    {
        echo "CUDA+OpenMP Hybrid Search Engine Performance Report"
        echo "Generated on: $(date)"
        echo "=================================================="
        echo ""
        
        # System information
        echo "System Information:"
        echo "  OS: $(uname -s) $(uname -r)"
        echo "  CPU: $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | sed 's/^ *//' 2>/dev/null || echo 'Unknown')"
        echo "  CPU Cores: $(nproc)"
        echo "  Memory: $(free -h | grep '^Mem:' | awk '{print $2}' 2>/dev/null || echo 'Unknown')"
        
        if command -v nvidia-smi > /dev/null 2>&1; then
            echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
            echo "  GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) MB"
        else
            echo "  GPU: Not available"
        fi
        
        echo ""
        echo "Benchmark Results Summary:"
        echo "========================="
        
        # Read and format summary results
        tail -n +2 "$SUMMARY_FILE" | while IFS=',' read -r config mode threads gpu avg_time min_time max_time throughput speedup; do
            echo "Configuration: $config"
            echo "  Mode: $mode, Threads: $threads, GPU: $gpu"
            echo "  Average Time: $avg_time ms"
            echo "  Throughput: $throughput queries/sec"
            echo "  Speedup: $speedup"
            echo ""
        done
        
        echo "Top Performing Configurations:"
        echo "=============================="
        
        # Find top configurations by throughput
        tail -n +2 "$SUMMARY_FILE" | sort -t',' -k8 -nr | head -5 | while IFS=',' read -r config mode threads gpu avg_time min_time max_time throughput speedup; do
            echo "  $config: $throughput queries/sec (${speedup}x speedup)"
        done
        
        echo ""
        echo "Performance Analysis:"
        echo "===================="
        
        # Calculate some basic statistics
        local cpu_only_avg=$(tail -n +2 "$SUMMARY_FILE" | grep ",cpu," | awk -F',' '{sum+=$5; count++} END {if(count>0) print sum/count; else print "N/A"}')
        local gpu_only_avg=$(tail -n +2 "$SUMMARY_FILE" | grep ",gpu," | awk -F',' '{if($5!="FAILED" && $5!="SKIPPED") {sum+=$5; count++}} END {if(count>0) print sum/count; else print "N/A"}')
        local hybrid_avg=$(tail -n +2 "$SUMMARY_FILE" | grep ",hybrid," | awk -F',' '{if($5!="FAILED" && $5!="SKIPPED") {sum+=$5; count++}} END {if(count>0) print sum/count; else print "N/A"}')
        
        echo "  Average CPU-only time: $cpu_only_avg ms"
        echo "  Average GPU-only time: $gpu_only_avg ms"
        echo "  Average Hybrid time: $hybrid_avg ms"
        
        if [ "$gpu_only_avg" != "N/A" ] && [ "$cpu_only_avg" != "N/A" ]; then
            local gpu_speedup=$(echo "scale=2; $cpu_only_avg / $gpu_only_avg" | bc 2>/dev/null || echo "N/A")
            echo "  GPU vs CPU speedup: ${gpu_speedup}x"
        fi
        
        echo ""
        echo "Files Generated:"
        echo "================"
        echo "  Summary: $SUMMARY_FILE"
        echo "  Detailed: $DETAILED_FILE"
        echo "  Log: $LOG_FILE"
        echo "  Report: $report_file"
        
    } > "$report_file"
    
    print_success "Performance report generated: $report_file"
    
    # Display key results
    echo ""
    print_header "Key Results Summary"
    
    echo "Top 3 performing configurations:"
    tail -n +2 "$SUMMARY_FILE" | sort -t',' -k8 -nr | head -3 | while IFS=',' read -r config mode threads gpu avg_time min_time max_time throughput speedup; do
        echo "  $config: $throughput queries/sec (${avg_time}ms avg)"
    done
}

# Create visualization script
create_visualization_script() {
    local viz_script="$RESULTS_DIR/visualize_results.py"
    
    cat > "$viz_script" << 'EOF'
#!/usr/bin/env python3
"""
Visualization script for CUDA+OpenMP benchmark results
Requires: matplotlib, pandas, seaborn
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

def create_visualizations(summary_file, detailed_file, output_dir):
    """Create performance visualization plots"""
    
    # Load data
    try:
        summary_df = pd.read_csv(summary_file)
        detailed_df = pd.read_csv(detailed_file)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Filter out failed/skipped results
    summary_df = summary_df[~summary_df['Avg_Time_ms'].isin(['FAILED', 'SKIPPED'])]
    summary_df['Avg_Time_ms'] = pd.to_numeric(summary_df['Avg_Time_ms'], errors='coerce')
    summary_df['Throughput_qps'] = pd.to_numeric(summary_df['Throughput_qps'], errors='coerce')
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    fig_size = (12, 8)
    
    # 1. Throughput comparison
    plt.figure(figsize=fig_size)
    sns.barplot(data=summary_df, x='Configuration', y='Throughput_qps', hue='Mode')
    plt.title('Query Throughput by Configuration')
    plt.xlabel('Configuration')
    plt.ylabel('Queries per Second')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'throughput_comparison.png'), dpi=300)
    plt.close()
    
    # 2. Average time comparison
    plt.figure(figsize=fig_size)
    sns.barplot(data=summary_df, x='Configuration', y='Avg_Time_ms', hue='Mode')
    plt.title('Average Query Time by Configuration')
    plt.xlabel('Configuration')
    plt.ylabel('Average Time (ms)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_comparison.png'), dpi=300)
    plt.close()
    
    # 3. Speedup analysis
    speedup_df = summary_df[summary_df['GPU_Speedup'] != 'N/A'].copy()
    if not speedup_df.empty:
        speedup_df['GPU_Speedup'] = pd.to_numeric(speedup_df['GPU_Speedup'], errors='coerce')
        
        plt.figure(figsize=fig_size)
        sns.barplot(data=speedup_df, x='Configuration', y='GPU_Speedup')
        plt.title('GPU Acceleration Speedup')
        plt.xlabel('Configuration')
        plt.ylabel('Speedup Factor')
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='Baseline')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'speedup_analysis.png'), dpi=300)
        plt.close()
    
    # 4. Thread scaling analysis
    cpu_configs = summary_df[summary_df['Mode'] == 'cpu'].copy()
    if len(cpu_configs) > 1:
        plt.figure(figsize=fig_size)
        sns.lineplot(data=cpu_configs, x='Threads', y='Throughput_qps', marker='o')
        plt.title('CPU Thread Scaling Performance')
        plt.xlabel('Number of Threads')
        plt.ylabel('Queries per Second')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'thread_scaling.png'), dpi=300)
        plt.close()
    
    print(f"Visualizations saved to {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 visualize_results.py <summary_csv> <detailed_csv> <output_dir>")
        sys.exit(1)
    
    summary_file = sys.argv[1]
    detailed_file = sys.argv[2]
    output_dir = sys.argv[3]
    
    os.makedirs(output_dir, exist_ok=True)
    create_visualizations(summary_file, detailed_file, output_dir)
EOF
    
    chmod +x "$viz_script"
    print_success "Visualization script created: $viz_script"
}

# Main execution
main() {
    print_header "CUDA+OpenMP Hybrid Search Engine Benchmark"
    
    # Handle command line arguments
    local run_viz=false
    local config_only=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --visualize)
                run_viz=true
                shift
                ;;
            --config-only)
                config_only=true
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --visualize    Generate visualization plots (requires Python/matplotlib)"
                echo "  --config-only  Test only specific configuration"
                echo "  --help         Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Run benchmark steps
    check_prerequisites
    setup_benchmark
    
    if [ "$config_only" = true ]; then
        print_status "Running single configuration test..."
        run_benchmark_config "TEST_CONFIG" "hybrid" "8" "1" "0.5"
    else
        run_all_benchmarks
    fi
    
    generate_report
    create_visualization_script
    
    # Run visualization if requested
    if [ "$run_viz" = true ]; then
        print_status "Generating visualizations..."
        if command -v python3 > /dev/null 2>&1; then
            local viz_dir="$RESULTS_DIR/visualizations_$TIMESTAMP"
            mkdir -p "$viz_dir"
            python3 "$RESULTS_DIR/visualize_results.py" "$SUMMARY_FILE" "$DETAILED_FILE" "$viz_dir" || \\
                print_warning "Visualization generation failed (missing dependencies?)"
        else
            print_warning "Python3 not found, skipping visualization generation"
        fi
    fi
    
    print_success "Benchmark completed successfully!"
    print_status "Results directory: $RESULTS_DIR"
}

# Trap for cleanup
trap 'print_error "Benchmark interrupted"; exit 1' INT TERM

# Execute main function
main "$@"
