#!/bin/bash
# Super Hybrid Search Engine Comprehensive Benchmark Suite
# Tests CUDA + OpenMP + MPI performance across different configurations

echo "████████████████████████████████████████████████████████████████████████████████"
echo "           SUPER HYBRID SEARCH ENGINE COMPREHENSIVE BENCHMARK SUITE             "
echo "                    CUDA + OpenMP + MPI Performance Testing                     "
echo "████████████████████████████████████████████████████████████████████████████████"
echo ""

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BINARY="$PROJECT_DIR/bin/super_hybrid_engine"
RESULTS_DIR="$PROJECT_DIR/benchmark_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_FILE="$RESULTS_DIR/super_hybrid_benchmark_$TIMESTAMP.csv"
SUMMARY_FILE="$RESULTS_DIR/benchmark_summary_$TIMESTAMP.txt"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Test data
TEST_URL="https://medium.com/@lpramithamj"
TEST_QUERY="artificial intelligence machine learning"
CRAWL_DEPTH=2
CRAWL_PAGES=5

# System detection
echo " DETECTING SYSTEM CAPABILITIES..."
echo "════════════════════════════════════════════════════════════════════════════════"

# Detect CPU cores
CPU_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "4")
echo "CPU Cores: $CPU_CORES"

# Detect CUDA devices
CUDA_DEVICES=0
if command -v nvidia-smi &> /dev/null; then
    CUDA_DEVICES=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ $CUDA_DEVICES -gt 0 ]; then
        echo "CUDA Devices: $CUDA_DEVICES"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | nl -v0
    else
        echo "CUDA Devices: 0 (nvidia-smi found but no devices detected)"
    fi
else
    echo "CUDA Devices: 0 (nvidia-smi not found)"
fi

# Detect MPI
MPI_AVAILABLE=0
if command -v mpirun &> /dev/null; then
    MPI_AVAILABLE=1
    echo "MPI: Available ($(mpirun --version 2>/dev/null | head -n1))"
else
    echo "MPI: Not available"
fi

# Detect OpenMP
OPENMP_AVAILABLE=0
if command -v gcc &> /dev/null; then
    if gcc -fopenmp -dM -E - </dev/null 2>/dev/null | grep -q "_OPENMP"; then
        OPENMP_AVAILABLE=1
        echo "OpenMP: Available"
    else
        echo "OpenMP: Not available"
    fi
else
    echo "OpenMP: Cannot detect (gcc not found)"
fi

echo ""

# Build the project
echo " BUILDING SUPER HYBRID ENGINE..."
echo "════════════════════════════════════════════════════════════════════════════════"

cd "$PROJECT_DIR"

# Clean and build
make -f Makefile.super clean
if ! make -f Makefile.super super USE_CUDA=1 USE_OPENMP=1 USE_MPI=1; then
    echo " Build failed! Please check compilation errors."
    exit 1
fi

if [ ! -f "$BINARY" ]; then
    echo " Binary not found at $BINARY"
    exit 1
fi

echo " Super Hybrid Engine built successfully!"
echo ""

# Initialize results file
echo "Timestamp,Configuration,MPI_Processes,OpenMP_Threads,CUDA_Devices,Total_Parallel_Units,Phase,Operation,Duration_Seconds,Documents_Processed,Throughput_Docs_Per_Sec,Memory_Usage_MB,Notes" > "$RESULTS_FILE"

# Function to run benchmark and collect metrics
run_benchmark() {
    local config_name="$1"
    local mpi_procs="$2"
    local omp_threads="$3"
    local cuda_devices="$4"
    local extra_args="$5"
    local test_phase="$6"
    
    echo " Testing: $config_name"
    echo "   MPI Processes: $mpi_procs"
    echo "   OpenMP Threads: $omp_threads"
    echo "   CUDA Devices: $cuda_devices"
    echo "   Extra Args: $extra_args"
    
    local total_parallel_units=$((mpi_procs * omp_threads))
    if [ $cuda_devices -gt 0 ]; then
        total_parallel_units=$((total_parallel_units * 1000))  # Approximate CUDA cores
    fi
    
    # Run the benchmark
    local start_time=$(date +%s.%N)
    local output_file=$(mktemp)
    
    # Set environment
    export OMP_NUM_THREADS=$omp_threads
    
    # Run with timeout
    local cmd=""
    if [ $MPI_AVAILABLE -eq 1 ] && [ $mpi_procs -gt 1 ]; then
        cmd="timeout 300s mpirun --oversubscribe -np $mpi_procs $BINARY -t $omp_threads -g $cuda_devices $extra_args"
    else
        cmd="timeout 300s $BINARY -t $omp_threads -g $cuda_devices $extra_args"
    fi
    
    echo "   Command: $cmd"
    
    if eval $cmd > "$output_file" 2>&1; then
        local end_time=$(date +%s.%N)
        local duration=$(echo "$end_time - $start_time" | bc)
        
        # Parse output for metrics
        local docs_processed=$(grep -o "Successfully.*[0-9]\+ documents\|[0-9]\+ documents.*indexed" "$output_file" | grep -o "[0-9]\+" | tail -1 || echo "0")
        local memory_usage=$(grep -o "[0-9]\+\.[0-9]\+ MB" "$output_file" | head -1 | grep -o "[0-9]\+\.[0-9]\+" || echo "0")
        
        if [ -z "$docs_processed" ]; then docs_processed=0; fi
        if [ -z "$memory_usage" ]; then memory_usage=0; fi
        
        local throughput=0
        if [ "$docs_processed" -gt 0 ] && [ "$(echo "$duration > 0" | bc)" -eq 1 ]; then
            throughput=$(echo "scale=2; $docs_processed / $duration" | bc)
        fi
        
        # Record results
        echo "$TIMESTAMP,$config_name,$mpi_procs,$omp_threads,$cuda_devices,$total_parallel_units,$test_phase,Complete,$duration,$docs_processed,$throughput,$memory_usage,Success" >> "$RESULTS_FILE"
        
        echo "    Success: ${duration}s, $docs_processed docs, ${throughput} docs/s"
        
        # Extract specific timing information from output
        if grep -q "CUDA.*seconds" "$output_file"; then
            local cuda_time=$(grep "CUDA.*seconds" "$output_file" | grep -o "[0-9]\+\.[0-9]\+" | head -1)
            echo "$TIMESTAMP,$config_name,$mpi_procs,$omp_threads,$cuda_devices,$total_parallel_units,$test_phase,CUDA_Processing,$cuda_time,$docs_processed,$(echo "scale=2; $docs_processed / $cuda_time" | bc),$memory_usage,CUDA_Timing" >> "$RESULTS_FILE"
        fi
        
        if grep -q "OpenMP.*seconds" "$output_file"; then
            local openmp_time=$(grep "OpenMP.*seconds" "$output_file" | grep -o "[0-9]\+\.[0-9]\+" | head -1)
            echo "$TIMESTAMP,$config_name,$mpi_procs,$omp_threads,$cuda_devices,$total_parallel_units,$test_phase,OpenMP_Processing,$openmp_time,$docs_processed,$(echo "scale=2; $docs_processed / $openmp_time" | bc),$memory_usage,OpenMP_Timing" >> "$RESULTS_FILE"
        fi
        
        if grep -q "MPI.*seconds" "$output_file"; then
            local mpi_time=$(grep "MPI.*seconds" "$output_file" | grep -o "[0-9]\+\.[0-9]\+" | head -1)
            echo "$TIMESTAMP,$config_name,$mpi_procs,$omp_threads,$cuda_devices,$total_parallel_units,$test_phase,MPI_Communication,$mpi_time,0,0,$memory_usage,MPI_Timing" >> "$RESULTS_FILE"
        fi
        
    else
        echo "   Failed or timed out"
        echo "$TIMESTAMP,$config_name,$mpi_procs,$omp_threads,$cuda_devices,$total_parallel_units,$test_phase,Complete,300,0,0,0,Timeout_or_Error" >> "$RESULTS_FILE"
    fi
    
    rm -f "$output_file"
    echo ""
}

# Test configurations
echo " STARTING COMPREHENSIVE BENCHMARK TESTS..."
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Phase 1: Baseline tests (single technology)
echo "🔬 PHASE 1: BASELINE PERFORMANCE TESTS"
echo "────────────────────────────────────────────────────────────────────────────────"

# Serial baseline
run_benchmark "Serial_Baseline" 1 1 0 "--no-cuda --no-openmp --no-mpi -c $TEST_URL -d $CRAWL_DEPTH -p $CRAWL_PAGES" "Baseline"

# OpenMP only
if [ $OPENMP_AVAILABLE -eq 1 ]; then
    for threads in 2 4 8; do
        if [ $threads -le $CPU_CORES ]; then
            run_benchmark "OpenMP_Only_${threads}T" 1 $threads 0 "--no-cuda --no-mpi -c $TEST_URL -d $CRAWL_DEPTH -p $CRAWL_PAGES" "Baseline"
        fi
    done
fi

# MPI only
if [ $MPI_AVAILABLE -eq 1 ]; then
    for procs in 2 4; do
        if [ $procs -le $CPU_CORES ]; then
            run_benchmark "MPI_Only_${procs}P" $procs 1 0 "--no-cuda --no-openmp -c $TEST_URL -d $CRAWL_DEPTH -p $CRAWL_PAGES" "Baseline"
        fi
    done
fi

# CUDA only
if [ $CUDA_DEVICES -gt 0 ]; then
    run_benchmark "CUDA_Only" 1 1 1 "--no-openmp --no-mpi -c $TEST_URL -d $CRAWL_DEPTH -p $CRAWL_PAGES" "Baseline"
fi

echo ""

# Phase 2: Dual technology combinations
echo "🔬 PHASE 2: DUAL TECHNOLOGY COMBINATIONS"
echo "────────────────────────────────────────────────────────────────────────────────"

# OpenMP + MPI
if [ $OPENMP_AVAILABLE -eq 1 ] && [ $MPI_AVAILABLE -eq 1 ]; then
    for procs in 2 4; do
        for threads in 2 4; do
            if [ $((procs * threads)) -le $((CPU_CORES * 2)) ]; then
                run_benchmark "OpenMP_MPI_${procs}P_${threads}T" $procs $threads 0 "--no-cuda -c $TEST_URL -d $CRAWL_DEPTH -p $CRAWL_PAGES" "Dual_Tech"
            fi
        done
    done
fi

# CUDA + OpenMP
if [ $CUDA_DEVICES -gt 0 ] && [ $OPENMP_AVAILABLE -eq 1 ]; then
    for threads in 2 4 8; do
        if [ $threads -le $CPU_CORES ]; then
            run_benchmark "CUDA_OpenMP_${threads}T" 1 $threads 1 "--no-mpi -c $TEST_URL -d $CRAWL_DEPTH -p $CRAWL_PAGES" "Dual_Tech"
        fi
    done
fi

# CUDA + MPI
if [ $CUDA_DEVICES -gt 0 ] && [ $MPI_AVAILABLE -eq 1 ]; then
    for procs in 2 4; do
        if [ $procs -le $CUDA_DEVICES ]; then
            run_benchmark "CUDA_MPI_${procs}P" $procs 1 1 "--no-openmp -c $TEST_URL -d $CRAWL_DEPTH -p $CRAWL_PAGES" "Dual_Tech"
        fi
    done
fi

echo ""

# Phase 3: Triple technology (Super Hybrid)
echo "🔬 PHASE 3: SUPER HYBRID CONFIGURATIONS (CUDA + OpenMP + MPI)"
echo "────────────────────────────────────────────────────────────────────────────────"

if [ $CUDA_DEVICES -gt 0 ] && [ $OPENMP_AVAILABLE -eq 1 ] && [ $MPI_AVAILABLE -eq 1 ]; then
    # Conservative configurations
    run_benchmark "Super_Hybrid_Conservative" 2 2 1 "-c $TEST_URL -d $CRAWL_DEPTH -p $CRAWL_PAGES" "Super_Hybrid"
    run_benchmark "Super_Hybrid_Balanced" 2 4 1 "-c $TEST_URL -d $CRAWL_DEPTH -p $CRAWL_PAGES" "Super_Hybrid"
    
    # Aggressive configurations (if enough resources)
    if [ $CPU_CORES -ge 8 ]; then
        run_benchmark "Super_Hybrid_Aggressive" 4 4 1 "-c $TEST_URL -d $CRAWL_DEPTH -p $CRAWL_PAGES" "Super_Hybrid"
    fi
    
    # Maximum configuration
    local max_procs=$([ $CUDA_DEVICES -lt 4 ] && echo $CUDA_DEVICES || echo 4)
    local max_threads=$([ $CPU_CORES -gt 8 ] && echo 8 || echo $CPU_CORES)
    run_benchmark "Super_Hybrid_Maximum" $max_procs $max_threads $CUDA_DEVICES "-c $TEST_URL -d $CRAWL_DEPTH -p $CRAWL_PAGES" "Super_Hybrid"
else
    echo "️  Super Hybrid testing skipped (missing CUDA, OpenMP, or MPI support)"
fi

echo ""

# Phase 4: Query performance tests
echo "🔬 PHASE 4: QUERY PERFORMANCE TESTS"
echo "────────────────────────────────────────────────────────────────────────────────"

# Build index first (using optimal configuration)
echo " Building index for query tests..."
if [ $CUDA_DEVICES -gt 0 ] && [ $OPENMP_AVAILABLE -eq 1 ] && [ $MPI_AVAILABLE -eq 1 ]; then
    # Use super hybrid for index building
    run_benchmark "Index_Building_Super_Hybrid" 2 4 1 "-c $TEST_URL -d $CRAWL_DEPTH -p 10" "Index_Building"
else
    # Use best available configuration
    if [ $MPI_AVAILABLE -eq 1 ] && [ $OPENMP_AVAILABLE -eq 1 ]; then
        run_benchmark "Index_Building_Hybrid" 2 4 0 "--no-cuda -c $TEST_URL -d $CRAWL_DEPTH -p 10" "Index_Building"
    else
        run_benchmark "Index_Building_Serial" 1 1 0 "--no-cuda --no-openmp --no-mpi -c $TEST_URL -d $CRAWL_DEPTH -p 10" "Index_Building"
    fi
fi

# Test various query configurations
declare -a test_queries=(
    "artificial intelligence"
    "machine learning"
    "deep learning neural networks"
    "natural language processing"
    "computer vision"
)

for query in "${test_queries[@]}"; do
    if [ $CUDA_DEVICES -gt 0 ] && [ $OPENMP_AVAILABLE -eq 1 ] && [ $MPI_AVAILABLE -eq 1 ]; then
        run_benchmark "Query_Super_Hybrid" 2 4 1 "-q \"$query\"" "Query_Processing"
    fi
    
    if [ $OPENMP_AVAILABLE -eq 1 ]; then
        run_benchmark "Query_OpenMP" 1 4 0 "--no-cuda --no-mpi -q \"$query\"" "Query_Processing"
    fi
    
    run_benchmark "Query_Serial" 1 1 0 "--no-cuda --no-openmp --no-mpi -q \"$query\"" "Query_Processing"
done

echo ""

# Phase 5: Scalability tests
echo "🔬 PHASE 5: SCALABILITY TESTS"
echo "────────────────────────────────────────────────────────────────────────────────"

if [ $CUDA_DEVICES -gt 0 ] && [ $OPENMP_AVAILABLE -eq 1 ] && [ $MPI_AVAILABLE -eq 1 ]; then
    echo "Testing scalability with increasing workload..."
    
    for pages in 5 10 20; do
        run_benchmark "Scalability_${pages}pages" 2 4 1 "-c $TEST_URL -d 2 -p $pages" "Scalability"
    done
    
    # Test different GPU/CPU ratios
    for ratio in 0.3 0.5 0.7 0.9; do
        run_benchmark "GPU_Ratio_${ratio}" 2 4 1 "--gpu-ratio $ratio -c $TEST_URL -d $CRAWL_DEPTH -p $CRAWL_PAGES" "GPU_Ratio_Tuning"
    done
fi

echo ""

# Generate comprehensive report
echo "📋 GENERATING COMPREHENSIVE PERFORMANCE REPORT..."
echo "════════════════════════════════════════════════════════════════════════════════"

# Create summary report
cat > "$SUMMARY_FILE" << EOF
████████████████████████████████████████████████████████████████████████████████
                    SUPER HYBRID SEARCH ENGINE BENCHMARK REPORT
                              Generated: $(date)
████████████████████████████████████████████████████████████████████████████████

SYSTEM CONFIGURATION:
═══════════════════════════════════════════════════════════════════════════════
CPU Cores: $CPU_CORES
CUDA Devices: $CUDA_DEVICES
MPI Available: $([ $MPI_AVAILABLE -eq 1 ] && echo "Yes" || echo "No")
OpenMP Available: $([ $OPENMP_AVAILABLE -eq 1 ] && echo "Yes" || echo "No")

BENCHMARK SUMMARY:
═══════════════════════════════════════════════════════════════════════════════
EOF

# Analyze results and generate insights
if [ -f "$RESULTS_FILE" ]; then
    echo "Total Test Runs: $(tail -n +2 "$RESULTS_FILE" | wc -l)" >> "$SUMMARY_FILE"
    echo "Successful Runs: $(grep -c "Success" "$RESULTS_FILE")" >> "$SUMMARY_FILE"
    echo "Failed/Timeout Runs: $(grep -c "Timeout_or_Error" "$RESULTS_FILE")" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
    
    # Find best performing configuration
    echo "TOP PERFORMING CONFIGURATIONS:" >> "$SUMMARY_FILE"
    echo "────────────────────────────────────────────────────────────────────────────" >> "$SUMMARY_FILE"
    
    # Best throughput
    best_throughput=$(tail -n +2 "$RESULTS_FILE" | grep "Success" | sort -t',' -k11 -nr | head -1)
    if [ -n "$best_throughput" ]; then
        config=$(echo "$best_throughput" | cut -d',' -f2)
        throughput=$(echo "$best_throughput" | cut -d',' -f11)
        echo "Best Throughput: $config ($throughput docs/sec)" >> "$SUMMARY_FILE"
    fi
    
    # Fastest execution
    fastest=$(tail -n +2 "$RESULTS_FILE" | grep "Success" | grep "Complete" | sort -t',' -k9 -n | head -1)
    if [ -n "$fastest" ]; then
        config=$(echo "$fastest" | cut -d',' -f2)
        duration=$(echo "$fastest" | cut -d',' -f9)
        echo "Fastest Execution: $config (${duration}s)" >> "$SUMMARY_FILE"
    fi
    
    echo "" >> "$SUMMARY_FILE"
    
    # Technology comparison
    echo "TECHNOLOGY PERFORMANCE COMPARISON:" >> "$SUMMARY_FILE"
    echo "────────────────────────────────────────────────────────────────────────────" >> "$SUMMARY_FILE"
    
    # Calculate average performance for each technology combination
    for tech in "Serial" "OpenMP" "MPI" "CUDA" "Super_Hybrid"; do
        avg_throughput=$(grep "$tech" "$RESULTS_FILE" | grep "Success" | awk -F',' '{sum+=$11; count++} END {if(count>0) print sum/count; else print 0}')
        if [ -n "$avg_throughput" ] && [ "$(echo "$avg_throughput > 0" | bc)" -eq 1 ]; then
            echo "$tech Average Throughput: $avg_throughput docs/sec" >> "$SUMMARY_FILE"
        fi
    done
    
    echo "" >> "$SUMMARY_FILE"
    
    # Detailed results table
    echo "DETAILED RESULTS:" >> "$SUMMARY_FILE"
    echo "────────────────────────────────────────────────────────────────────────────" >> "$SUMMARY_FILE"
    echo "Configuration,MPI,OMP,CUDA,Duration,Docs,Throughput,Status" >> "$SUMMARY_FILE"
    
    tail -n +2 "$RESULTS_FILE" | grep "Complete" | while IFS=',' read -r timestamp config mpi omp cuda total phase op duration docs throughput mem notes; do
        status=$(echo "$notes" | grep -q "Success" && echo "" || echo "")
        printf "%-25s %3s %3s %4s %8.3f %4s %9.2f %s\n" "$config" "$mpi" "$omp" "$cuda" "$duration" "$docs" "$throughput" "$status" >> "$SUMMARY_FILE"
    done
fi

echo "" >> "$SUMMARY_FILE"
echo "Raw data available in: $RESULTS_FILE" >> "$SUMMARY_FILE"
echo "════════════════════════════════════════════════════════════════════════════════" >> "$SUMMARY_FILE"

echo " Benchmark completed successfully!"
echo ""
echo " RESULTS SUMMARY:"
echo "   Detailed CSV: $RESULTS_FILE"
echo "   Summary Report: $SUMMARY_FILE"
echo ""
echo " Quick Statistics:"
if [ -f "$RESULTS_FILE" ]; then
    total_tests=$(tail -n +2 "$RESULTS_FILE" | wc -l)
    successful_tests=$(grep -c "Success" "$RESULTS_FILE" 2>/dev/null || echo "0")
    echo "   Total Tests: $total_tests"
    echo "   Successful: $successful_tests"
    echo "   Success Rate: $(echo "scale=1; $successful_tests * 100 / $total_tests" | bc)%"
fi

echo ""
echo " To view the complete report:"
echo "   cat $SUMMARY_FILE"
echo ""
echo " To analyze CSV data:"
echo "   python3 scripts/analyze_benchmark.py $RESULTS_FILE"
echo ""

echo "🏁 Super Hybrid Benchmark Suite Completed!"
echo "████████████████████████████████████████████████████████████████████████████████"
