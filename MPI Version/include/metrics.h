#ifndef METRICS_H
#define METRICS_H

#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>

// Structure to hold performance metrics
typedef struct {
    // Time measurements in milliseconds
    double total_time;             // Total execution time
    double crawling_time;          // Time spent crawling/downloading content
    double indexing_time;          // Time spent building the index
    double parsing_time;           // Time spent parsing documents
    double tokenizing_time;        // Time spent tokenizing text
    double stemming_time;          // Time spent on stemming
    double query_processing_time;  // Time spent processing queries
    
    // Memory usage in kilobytes
    long memory_usage_before;      // Memory usage before operation
    long memory_usage_after;       // Memory usage after operation
    long peak_memory_usage;        // Peak memory usage
    
    // Document and index statistics
    int total_docs;                // Total number of documents processed
    int total_tokens;              // Total number of tokens processed
    int unique_terms;              // Number of unique terms in the index
    
    // Query statistics
    int num_queries;               // Number of queries processed
    double avg_query_latency;      // Average query latency
} SearchEngineMetrics;

extern SearchEngineMetrics metrics;

// Initialize metrics structure
void init_metrics();

// Record start time for a specific operation
void start_timer();

// Record end time and add to a specific metric
double stop_timer();

// Get current memory usage in KB
long get_current_memory_usage();

// Get peak memory usage in KB
long get_peak_memory_usage();

// Get current time in milliseconds
double get_current_time_ms();

// Print all metrics in a formatted way
void print_metrics();

// Save metrics to a CSV file
void save_metrics_to_csv(const char* filename);

// Update document and index statistics
void update_index_stats(int docs, int tokens, int terms);

// Record query latency
void record_query_latency(double latency);

#endif // METRICS_H
