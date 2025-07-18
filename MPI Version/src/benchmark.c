#include "../include/benchmark.h"
#include "../include/metrics.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Global speedup metrics structure
SpeedupMetrics speedup_metrics;

void init_baseline_metrics(const char* filepath) {
    FILE* file = fopen(filepath, "r");
    if (!file) {
        printf("Warning: Could not open baseline metrics file %s\n", filepath);
        // Set default baseline values if file can't be opened
        speedup_metrics.baseline_crawling_time = 1000.0;
        speedup_metrics.baseline_parsing_time = 200.0;
        speedup_metrics.baseline_tokenizing_time = 300.0;
        speedup_metrics.baseline_indexing_time = 500.0;
        speedup_metrics.baseline_query_time = 50.0;
        return;
    }

    char line[256];
    char metric[64];
    double value;

    // Skip header line
    fgets(line, sizeof(line), file);

    // Read metrics from file
    while (fgets(line, sizeof(line), file)) {
        if (sscanf(line, "%[^,],%lf", metric, &value) == 2) {
            if (strcmp(metric, "CrawlingTime_ms") == 0) {
                speedup_metrics.baseline_crawling_time = value;
            } else if (strcmp(metric, "ParsingTime_ms") == 0) {
                speedup_metrics.baseline_parsing_time = value;
            } else if (strcmp(metric, "TokenizingTime_ms") == 0) {
                speedup_metrics.baseline_tokenizing_time = value;
            } else if (strcmp(metric, "IndexingTime_ms") == 0) {
                speedup_metrics.baseline_indexing_time = value;
            } else if (strcmp(metric, "QueryProcessingTime_ms") == 0) {
                speedup_metrics.baseline_query_time = value;
            }
        }
    }

    fclose(file);
    printf("Loaded baseline metrics from %s\n", filepath);
}

void calculate_speedup(SpeedupMetrics* speedup) {
    extern SearchEngineMetrics metrics;
    
    speedup->current_crawling_time = speedup->current_crawling_time > 0 ? 
                                    speedup->current_crawling_time : 
                                    metrics.crawling_time;
    speedup->current_parsing_time = metrics.parsing_time;
    speedup->current_tokenizing_time = metrics.tokenizing_time;
    speedup->current_indexing_time = metrics.indexing_time;
    speedup->current_query_time = metrics.query_processing_time;

    double crawl_speedup = speedup->current_crawling_time > 0 ? 
                          speedup->baseline_crawling_time / speedup->current_crawling_time : 0;
    double parse_speedup = speedup->current_parsing_time > 0 ? 
                          speedup->baseline_parsing_time / speedup->current_parsing_time : 0;
    double token_speedup = speedup->current_tokenizing_time > 0 ? 
                          speedup->baseline_tokenizing_time / speedup->current_tokenizing_time : 0;
    double index_speedup = speedup->current_indexing_time > 0 ? 
                          speedup->baseline_indexing_time / speedup->current_indexing_time : 0;
    double query_speedup = speedup->current_query_time > 0 ? 
                          speedup->baseline_query_time / speedup->current_query_time : 0;

    // Print speedup results
    printf("\n=========== PERFORMANCE SPEEDUP METRICS ===========\n");
    if (speedup->current_crawling_time > 0) {
        printf("Crawling:     %.2f ms  (Baseline: %.2f ms)  Speedup: %.2fx\n", 
              speedup->current_crawling_time, speedup->baseline_crawling_time, crawl_speedup);
    }
    printf("Parsing:      %.2f ms  (Baseline: %.2f ms)  Speedup: %.2fx\n", 
          speedup->current_parsing_time, speedup->baseline_parsing_time, parse_speedup);
    printf("Tokenizing:   %.2f ms  (Baseline: %.2f ms)  Speedup: %.2fx\n", 
          speedup->current_tokenizing_time, speedup->baseline_tokenizing_time, token_speedup);
    printf("Indexing:     %.2f ms  (Baseline: %.2f ms)  Speedup: %.2fx\n", 
          speedup->current_indexing_time, speedup->baseline_indexing_time, index_speedup);
    printf("Query:        %.2f ms  (Baseline: %.2f ms)  Speedup: %.2fx\n", 
          speedup->current_query_time, speedup->baseline_query_time, query_speedup);
    printf("===================================================\n");
}

void save_as_baseline(const char* filepath) {
    FILE* file = fopen(filepath, "w");
    if (!file) {
        printf("Error: Could not create baseline metrics file %s\n", filepath);
        return;
    }

    fprintf(file, "Metric,Value\n");
    
    fprintf(file, "CrawlingTime_ms,%.2f\n", metrics.crawling_time);
    fprintf(file, "ParsingTime_ms,%.2f\n", metrics.parsing_time);
    fprintf(file, "TokenizingTime_ms,%.2f\n", metrics.tokenizing_time);
    fprintf(file, "IndexingTime_ms,%.2f\n", metrics.indexing_time);
    fprintf(file, "QueryProcessingTime_ms,%.2f\n", metrics.query_processing_time);
    fprintf(file, "TotalExecutionTime_ms,%.2f\n", metrics.total_time);
    fprintf(file, "MemoryUsage_KB,%ld\n", 
           metrics.memory_usage_after - metrics.memory_usage_before);
    fprintf(file, "Documents,%d\n", metrics.total_docs);
    fprintf(file, "UniqueTerms,%d\n", metrics.unique_terms);
    fprintf(file, "AvgQueryLatency_ms,%.2f\n", metrics.avg_query_latency);

    fclose(file);
    printf("Saved current metrics as baseline to %s\n", filepath);
}
