#ifndef BENCHMARK_H
#define BENCHMARK_H

// Structure to hold baseline and current performance metrics for speedup calculation
typedef struct {
    // Baseline measurements (in milliseconds)
    double baseline_crawling_time;
    double baseline_parsing_time;
    double baseline_tokenizing_time;
    double baseline_indexing_time;
    double baseline_query_time;
    
    // Current measurements (in milliseconds)
    double current_crawling_time;
    double current_parsing_time;
    double current_tokenizing_time;
    double current_indexing_time;
    double current_query_time;
} SpeedupMetrics;

// Initialize baseline metrics from a CSV file
void init_baseline_metrics(const char* filepath);

// Calculate and display speedup metrics
void calculate_speedup(SpeedupMetrics* metrics);

// Save current metrics as baseline
void save_as_baseline(const char* filepath);

#endif // BENCHMARK_H
