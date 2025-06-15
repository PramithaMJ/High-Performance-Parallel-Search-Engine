#!/bin/bash
# filepath: /Users/pramithajayasooriya/Desktop/Academic/Semester 7/HPC/hpc/search engine/scripts/performance_benchmark.sh

echo "========== SEARCH ENGINE PERFORMANCE BENCHMARK =========="

# Make sure we have a baseline to compare against
if [ ! -f "data/serial_metrics.csv" ]; then
    echo "No baseline metrics found. Running baseline measurement first..."
    # Create baseline with -O0 optimization
    make clean
    make CC=gcc CFLAGS="-Wall -O0" 
    echo "Running baseline measurement with no optimizations..."
    ./bin/search_engine -c https://medium.com/@lpramithamj -d 1 -p 3
    # Automatically save as baseline
    echo "y" | ./bin/search_engine
    echo "Baseline metrics saved."
fi

# Different optimization levels to test
declare -a OPTIMIZATIONS=("-O1" "-O2" "-O3" "-O3 -march=native -mtune=native")
declare -a OPT_NAMES=("Level 1" "Level 2" "Level 3" "Level 3 + Arch Specific")

echo -e "\n===== COMPARING DIFFERENT OPTIMIZATION LEVELS =====\n"

# Pre-download some content for testing to avoid network variations
echo "Pre-downloading content for consistent benchmarking..."
if [ ! -d "dataset" ] || [ $(ls -1 dataset | wc -l) -lt 5 ]; then
  mkdir -p dataset
  ./bin/search_engine -c https://medium.com/@lpramithamj -d 1 -p 3
fi

for i in "${!OPTIMIZATIONS[@]}"; do
    echo "Testing optimization ${OPT_NAMES[$i]} (${OPTIMIZATIONS[$i]})..."
    
    # Build with current optimization level
    make clean
    make CC=gcc CFLAGS="-Wall ${OPTIMIZATIONS[$i]}"
    
    echo "Running search engine with ${OPT_NAMES[$i]} optimization..."
    
    # Run with existing dataset (avoid network issues and focus on processing speed)
    echo -e "high performance computing\nn" | ./bin/search_engine
    
    echo -e "\n------------------------------------------------\n"
done

echo -e "\n============ END OF BENCHMARK RESULTS ============\n"

# Print summary
echo "Summary:"
echo "Baseline metrics: data/serial_metrics.csv"
echo "For detailed timing information, see the output above."
echo "To manually compare with more data points, try:"
echo "./bin/search_engine -c https://medium.com/@lpramithamj -d 2 -p 10"
