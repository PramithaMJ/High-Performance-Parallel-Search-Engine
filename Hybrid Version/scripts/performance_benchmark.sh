#!/bin/bash
# filepath: /Users/pramithajayasooriya/Library/CloudStorage/OneDrive-Personal/Academic/Academic/Semester 7/HPC/hpc/HPC Project/High-Performance-Parallel-Search-Engine/Hybrid Version/scripts/performance_benchmark.sh

echo "========== HYBRID MPI+OPENMP SEARCH ENGINE PERFORMANCE BENCHMARK =========="
echo "Testing performance across different MPI process and OpenMP thread combinations"
echo "Results will be saved to ../data/hybrid_metrics.csv"
echo ""

# Make sure search engine is built with hybrid optimization
cd "$(dirname "$0")/.."
make clean
make
    echo "Running baseline measurement with no optimizations..."
    ./bin/search_engine -c https://medium.com/@lpramithamj -d 1 -p 3
    # Set up output file
OUTPUT_FILE="../data/hybrid_metrics.csv"
echo "MPI_Processes,OMP_Threads,Total_Threads,Query,Processing_Time_ms" > $OUTPUT_FILE

# Test queries
QUERIES=(
  "artificial intelligence"
  "machine learning algorithm"
  "deep neural networks"
  "natural language processing"
  "computer vision techniques"
)

# Pre-download some content for testing to avoid network variations
echo "Pre-downloading content for consistent benchmarking..."
if [ ! -d "../dataset" ] || [ $(ls -1 ../dataset | wc -l) -lt 5 ]; then
  mkdir -p ../dataset
  ../bin/search_engine -c https://medium.com/@lpramithamj -d 1 -p 5
fi

# Function to run a test with specific process and thread count
run_test() {
  local processes=$1
  local threads=$2
  local query="$3"
  
  echo "Testing with $processes MPI processes and $threads OpenMP threads per process..."
  
  # Calculate total threads
  local total_threads=$((processes * threads))
  
  # Set the OMP_NUM_THREADS environment variable
  export OMP_NUM_THREADS=$threads
  
  # Run the search engine with the query
  result=$(mpirun -np $processes ../bin/search_engine -q "$query")
  
  # Extract the processing time from the output
  processing_time=$(echo "$result" | grep "Query processed in" | awk '{print $4}')
  
  # Append to output file
  echo "$processes,$threads,$total_threads,\"$query\",$processing_time" >> $OUTPUT_FILE
  
  echo "Processing time: $processing_time ms"
  echo "--------------------------------"
}

echo -e "\n===== COMPARING DIFFERENT MPI + OPENMP CONFIGURATIONS =====\n"

# Detect system capabilities
TOTAL_CORES=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)
echo "Detected $TOTAL_CORES total cores/threads on this system"

# Define testing configurations based on system capabilities
MPI_CONFIGS=(1 2)
if [ $TOTAL_CORES -ge 4 ]; then
  MPI_CONFIGS+=(4)
fi
if [ $TOTAL_CORES -ge 8 ]; then
  MPI_CONFIGS+=(8)
fi

OMP_CONFIGS=(1 2)
if [ $TOTAL_CORES -ge 4 ]; then
  OMP_CONFIGS+=(4)
fi
if [ $TOTAL_CORES -ge 8 ]; then
  OMP_CONFIGS+=(8)
fi

# Test different combinations of processes and threads
for query in "${QUERIES[@]}"; do
  echo "Running benchmarks for query: $query"
  
  for mpi_proc in "${MPI_CONFIGS[@]}"; do
    for omp_threads in "${OMP_CONFIGS[@]}"; do
      # Skip configurations that exceed system capabilities
      total_threads=$((mpi_proc * omp_threads))
      if [ $total_threads -le $((TOTAL_CORES * 2)) ]; then
        run_test $mpi_proc $omp_threads "$query"
      fi
    done
  done
  
  # Also test maximum processes with 1 thread each
  max_procs=$((TOTAL_CORES / 2))
  if [ $max_procs -gt 1 ] && [ $max_procs -ne 1 ] && [ $max_procs -ne 2 ] && [ $max_procs -ne 4 ] && [ $max_procs -ne 8 ]; then
    run_test $max_procs 1 "$query"
  fi
  
  # And test 1 process with maximum threads
  max_threads=$TOTAL_CORES
  if [ $max_threads -gt 8 ]; then
    run_test 1 $max_threads "$query"
  fi
done

echo -e "\n============ END OF BENCHMARK RESULTS ============\n"

# Print summary
echo -e "\n===== PERFORMANCE ANALYSIS =====\n"

echo "Summary of average processing times by configuration:"
awk -F, 'NR>1 {sum[$1","$2]+=$5; count[$1","$2]++} END {for (key in sum) print key, sum[key]/count[key] " ms (avg)"}' $OUTPUT_FILE | sort -t, -k1n -k2n | 
  awk '{printf "MPI: %2s | OpenMP: %2s | Avg Time: %7.2f ms\n", $1, $2, $3}'

echo -e "\nBest configurations by performance:"
awk -F, 'NR>1 {if(min[$1","$2]==""){min[$1","$2]=$5} else if($5<min[$1","$2]){min[$1","$2]=$5}} END {for (key in min) print key, min[key] " ms (best)"}' $OUTPUT_FILE | 
  sort -n -k3 | awk '{printf "MPI: %2s | OpenMP: %2s | Best Time: %7.2f ms\n", $1, $2, $3}' | head -5

echo -e "\nEfficiency analysis (lower is better):"
awk -F, 'NR>1 {
  total_threads = $1 * $2;
  sum[total_threads","$1","$2] += $5; 
  count[total_threads","$1","$2]++
} 
END {
  for (key in sum) {
    split(key, parts, ",");
    total = parts[1];
    mpi = parts[2];
    omp = parts[3];
    avg = sum[key]/count[key];
    efficiency = avg * total;
    printf "%s,%s,%s,%f,%f\n", total, mpi, omp, avg, efficiency;
  }
}' $OUTPUT_FILE | sort -t, -k5n | 
  awk -F, '{printf "Total cores: %2s (MPI: %2s × OpenMP: %2s) | Avg: %7.2f ms | Efficiency score: %7.2f\n", $1, $2, $3, $4, $5}' | head -5

echo -e "\nFull results saved to: $OUTPUT_FILE"
echo -e "Run 'python -m matplotlib' or another tool to visualize these results for deeper analysis."
