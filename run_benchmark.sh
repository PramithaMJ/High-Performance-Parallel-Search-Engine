#!/bin/bash
# filepath: /Users/pramithajayasooriya/Desktop/Academic/Semester 7/HPC/hpc/search engine/run_benchmark.sh

echo "========== SEARCH ENGINE PERFORMANCE EVALUATION =========="
echo "1. Running evaluation on local dataset..."

./evaluate
echo -e "\nSummary of local dataset metrics:"
grep -E "IndexingTime_ms|ParsingTime_ms|TokenizingTime_ms|StemmingTime_ms|QueryProcessingTime_ms" serial_metrics.csv

echo -e "\n2. Running evaluation with web content crawling..."
./evaluate "https://en.wikipedia.org/wiki/Search_engine"

echo -e "\nSummary of web content metrics:"
grep -E "CrawlingTime_ms|IndexingTime_ms|ParsingTime_ms|TokenizingTime_ms|StemmingTime_ms" serial_metrics.csv

echo -e "\n========== PERFORMANCE COMPARISON =========="
echo "To evaluate serial performance, use the following metrics:"
echo "1. Indexing speed: Documents per second"
echo "2. Parsing efficiency: Characters processed per millisecond"
echo "3. Query latency: Average response time for queries"
echo "4. Memory efficiency: Memory usage per document"
echo "5. Crawling latency: Time to download and process web content"

# Calculate derived metrics if we have the necessary values
DOCS=$(grep -E "Documents" serial_metrics.csv | cut -d, -f2)
INDEX_TIME=$(grep -E "IndexingTime_ms" serial_metrics.csv | cut -d, -f2)
PARSING_TIME=$(grep -E "ParsingTime_ms" serial_metrics.csv | cut -d, -f2)
MEMORY=$(grep -E "MemoryIncrease_KB" serial_metrics.csv | cut -d, -f2)
QUERY_LATENCY=$(grep -E "AvgQueryLatency_ms" serial_metrics.csv | cut -d, -f2)

if [ ! -z "$DOCS" ] && [ ! -z "$INDEX_TIME" ]; then
    if (( $(echo "$INDEX_TIME > 0" | bc -l) )); then
        DOCS_PER_SEC=$(echo "scale=2; $DOCS * 1000 / $INDEX_TIME" | bc)
        echo -e "\nDerived Metrics:"
        echo "Documents indexed per second: $DOCS_PER_SEC"
    fi
fi

if [ ! -z "$DOCS" ] && [ ! -z "$MEMORY" ]; then
    MEM_PER_DOC=$(echo "scale=2; $MEMORY / $DOCS" | bc)
    echo "Memory usage per document: $MEM_PER_DOC KB"
fi

if [ ! -z "$QUERY_LATENCY" ]; then
    echo "Average query latency: $QUERY_LATENCY ms"
fi
