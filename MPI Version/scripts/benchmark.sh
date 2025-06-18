#!/bin/bash
# filepath: /Users/pramithajayasooriya/Desktop/Academic/Semester 7/HPC/hpc/search engine/benchmark.sh

echo "========== SEARCH ENGINE SERIAL MODE BENCHMARK =========="
echo "Running benchmark with 5 iterations..."

# Number of iterations
ITERATIONS=5

# Headers for results
echo -e "\nTiming Results (in seconds):"
printf "%-10s %-10s %-10s %-10s %-10s\n" "Run" "Real" "User" "Sys" "Total"

# Run benchmark multiple times
for i in $(seq 1 $ITERATIONS); do
    echo -n "Run $i: "
    # Use /usr/bin/time to get detailed metrics
    { /usr/bin/time -p ./evaluate > /dev/null; } 2>&1 | awk '{print $2}' | tr '\n' ' ' | awk -v run=$i '{printf "%-10s %-10.3f %-10.3f %-10.3f %-10.3f\n", run, $1, $2, $3, ($2+$3)}'
done

# Summarize the results from the CSV file
echo -e "\nAverage Metrics from serial_metrics.csv:"
PARSING_TIME=$(awk -F, '/ParsingTime_ms/ {print $2}' serial_metrics.csv)
TOKENIZING_TIME=$(awk -F, '/TokenizingTime_ms/ {print $2}' serial_metrics.csv)
STEMMING_TIME=$(awk -F, '/StemmingTime_ms/ {print $2}' serial_metrics.csv)
INDEXING_TIME=$(awk -F, '/IndexingTime_ms/ {print $2}' serial_metrics.csv)
QUERY_TIME=$(awk -F, '/QueryProcessingTime_ms/ {print $2}' serial_metrics.csv)
MEMORY_USAGE=$(awk -F, '/MemoryIncrease_KB/ {print $2}' serial_metrics.csv)
DOCS=$(awk -F, '/Documents/ {print $2}' serial_metrics.csv)
UNIQUE_TERMS=$(awk -F, '/UniqueTerms/ {print $2}' serial_metrics.csv)
AVG_QUERY_LATENCY=$(awk -F, '/AvgQueryLatency_ms/ {print $2}' serial_metrics.csv)

echo "Parsing Time: $PARSING_TIME ms"
echo "Tokenizing Time: $TOKENIZING_TIME ms"
echo "Stemming Time: $STEMMING_TIME ms" 
echo "Indexing Time: $INDEXING_TIME ms"
echo "Query Processing Time: $QUERY_TIME ms"
echo "Memory Usage: $MEMORY_USAGE KB"
echo "Documents Indexed: $DOCS"
echo "Unique Terms: $UNIQUE_TERMS"
echo "Average Query Latency: $AVG_QUERY_LATENCY ms"

echo -e "\n========== END OF BENCHMARK =========="
