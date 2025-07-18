#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/parser.h"
#include "../include/index.h"
#include "../include/ranking.h"
#include "../include/metrics.h"
#include "../include/utils.h"
#include "../include/crawler.h"

// Sample queries for evaluation
const char *sample_queries[] = {
    "circuit breaker",
    "distributed tracing",
    "linux wake up",
    "parallel computing",
    "microservices"
};

#define NUM_QUERIES (sizeof(sample_queries) / sizeof(sample_queries[0]))

int main(int argc, char* argv[])
{
    printf("========== SEARCH ENGINE EVALUATION (SERIAL MODE) ==========\n");
    

    init_metrics();
    
    start_timer();
    
    printf("Running evaluation of search engine in serial mode...\n\n");
    
    const char* test_url = NULL;
    if (argc > 1) {
        test_url = argv[1];
        printf("Crawling URL: %s\n", test_url);
        
        // Start timing for crawling
        start_timer();
        
        char* filepath = download_url(test_url);
        if (filepath) {
            printf("Successfully downloaded content to %s\n", filepath);
            metrics.crawling_time = stop_timer();
            printf("Crawling completed in %.2f ms\n\n", metrics.crawling_time);
        } else {
            printf("Failed to download content from URL\n");
            return 1;
        }
    }
    
    clear_index();
    
    printf("Building index from dataset directory...\n");
    double start_index_time = get_current_time_ms();
    
    int total_docs = build_index("dataset");
    
    // Record total indexing time (end - start)
    double end_index_time = get_current_time_ms();
    metrics.indexing_time = end_index_time - start_index_time;
    
    printf("Indexed %d documents in %.2f ms\n", total_docs, metrics.indexing_time);
    
    metrics.total_docs = total_docs;
    metrics.unique_terms = index_size;
    
    printf("\nEvaluating query performance with %lu sample queries...\n", NUM_QUERIES);
    
    for (size_t i = 0; i < NUM_QUERIES; i++) {
        printf("\nQuery %zu: \"%s\"\n", i+1, sample_queries[i]);
        
        start_timer();
        
        rank_bm25(sample_queries[i], total_docs, 5);
        
        double query_time = stop_timer();
        metrics.query_processing_time += query_time;
        record_query_latency(query_time);
        
        printf("Query processed in %.2f ms\n", query_time);
    }
    
    metrics.memory_usage_after = get_current_memory_usage();
    
    metrics.total_time = stop_timer();
    
    print_metrics();
    
    save_metrics_to_csv("serial_metrics.csv");
    
    return 0;
}