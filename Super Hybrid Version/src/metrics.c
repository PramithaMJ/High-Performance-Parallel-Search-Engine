#include "../include/metrics.h"
#include <stdlib.h>
#include <string.h>

// Global metrics structure
SearchEngineMetrics metrics;

// Variable to store timer start time
static struct timespec timer_start;

void init_metrics() {
    memset(&metrics, 0, sizeof(SearchEngineMetrics));
    metrics.memory_usage_before = get_current_memory_usage();
}

void start_timer() {
    clock_gettime(CLOCK_MONOTONIC, &timer_start);
}

double stop_timer() {
    struct timespec timer_end;
    clock_gettime(CLOCK_MONOTONIC, &timer_end);
    
    double elapsed = (timer_end.tv_sec - timer_start.tv_sec) * 1000.0;
    elapsed += (timer_end.tv_nsec - timer_start.tv_nsec) / 1000000.0;
    
    return elapsed;
}

long get_current_memory_usage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss;
}

long get_peak_memory_usage() {
    long current = get_current_memory_usage();
    if (current > metrics.peak_memory_usage) {
        metrics.peak_memory_usage = current;
    }
    return metrics.peak_memory_usage;
}

void update_index_stats(int docs, int tokens, int terms) {
    metrics.total_docs = docs;
    metrics.total_tokens = tokens;
    metrics.unique_terms = terms;
}

void record_query_latency(double latency) {
    metrics.num_queries++;
    // Update running average of query latency
    metrics.avg_query_latency = 
        (metrics.avg_query_latency * (metrics.num_queries - 1) + latency) / metrics.num_queries;
}

void print_metrics() {
    printf("\n========== SEARCH ENGINE METRICS ==========\n");
    
    // Time metrics
    printf("\n--- Timing Metrics (ms) ---\n");
    if (metrics.crawling_time > 0) {
        printf("Crawling Time:          %.2f ms\n", metrics.crawling_time);
    }
    printf("Parsing Time:           %.2f ms\n", metrics.parsing_time);
    printf("Tokenizing Time:        %.2f ms\n", metrics.tokenizing_time);
    printf("Stemming Time:          %.2f ms\n", metrics.stemming_time);
    printf("Total Indexing Time:    %.2f ms\n", metrics.indexing_time);
    printf("Query Processing Time:  %.2f ms\n", metrics.query_processing_time);
    printf("Total Execution Time:   %.2f ms\n", metrics.total_time);
    
    // Memory metrics
    printf("\n--- Memory Metrics (KB) ---\n");
    printf("Initial Memory Usage:   %ld KB\n", metrics.memory_usage_before);
    printf("Final Memory Usage:     %ld KB\n", metrics.memory_usage_after);
    printf("Peak Memory Usage:      %ld KB\n", metrics.peak_memory_usage);
    printf("Memory Increase:        %ld KB\n", 
           metrics.memory_usage_after - metrics.memory_usage_before);
    
    // Index statistics
    printf("\n--- Index Statistics ---\n");
    printf("Documents Indexed:      %d\n", metrics.total_docs);
    printf("Total Tokens:           %d\n", metrics.total_tokens);
    printf("Unique Terms:           %d\n", metrics.unique_terms);
    
    // Query statistics
    printf("\n--- Query Statistics ---\n");
    printf("Queries Processed:      %d\n", metrics.num_queries);
    printf("Average Query Latency:  %.2f ms\n", metrics.avg_query_latency);
    
    printf("\n===========================================\n");
}

void save_metrics_to_csv(const char* filename) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        printf("Error: Could not create metrics file %s\n", filename);
        return;
    }
    
    // Write CSV header
    fprintf(fp, "Metric,Value\n");
    
    // Write time metrics
    if (metrics.crawling_time > 0) {
        fprintf(fp, "CrawlingTime_ms,%.2f\n", metrics.crawling_time);
    }
    fprintf(fp, "ParsingTime_ms,%.2f\n", metrics.parsing_time);
    fprintf(fp, "TokenizingTime_ms,%.2f\n", metrics.tokenizing_time);
    fprintf(fp, "StemmingTime_ms,%.2f\n", metrics.stemming_time);
    fprintf(fp, "IndexingTime_ms,%.2f\n", metrics.indexing_time);
    fprintf(fp, "QueryProcessingTime_ms,%.2f\n", metrics.query_processing_time);
    fprintf(fp, "TotalExecutionTime_ms,%.2f\n", metrics.total_time);
    
    // Write memory metrics
    fprintf(fp, "InitialMemoryUsage_KB,%ld\n", metrics.memory_usage_before);
    fprintf(fp, "FinalMemoryUsage_KB,%ld\n", metrics.memory_usage_after);
    fprintf(fp, "PeakMemoryUsage_KB,%ld\n", metrics.peak_memory_usage);
    fprintf(fp, "MemoryIncrease_KB,%ld\n", 
           metrics.memory_usage_after - metrics.memory_usage_before);
    
    // Write index statistics
    fprintf(fp, "Documents,%d\n", metrics.total_docs);
    fprintf(fp, "TotalTokens,%d\n", metrics.total_tokens);
    fprintf(fp, "UniqueTerms,%d\n", metrics.unique_terms);
    
    // Write query statistics
    fprintf(fp, "QueriesProcessed,%d\n", metrics.num_queries);
    fprintf(fp, "AvgQueryLatency_ms,%.2f\n", metrics.avg_query_latency);
    
    fclose(fp);
    printf("Metrics saved to %s\n", filename);
}

double get_current_time_ms() {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return now.tv_sec * 1000.0 + now.tv_nsec / 1000000.0;
}
