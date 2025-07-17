#ifndef HYBRID_ENGINE_H
#define HYBRID_ENGINE_H

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#ifdef USE_CUDA
#include "cuda_kernels.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Hybrid engine configuration
#define MAX_DOCUMENTS 100000
#define MAX_TERMS 50000
#define MAX_QUERY_LENGTH 1024
#define MAX_FILENAME_LENGTH 256
#define MAX_TOKEN_LENGTH 128
#define DEFAULT_OMP_THREADS 8

// Processing mode enumeration
typedef enum {
    PROCESSING_CPU_ONLY = 0,
    PROCESSING_GPU_ONLY = 1,
    PROCESSING_HYBRID = 2,
    PROCESSING_AUTO = 3
} processing_mode_t;

// Load balancing strategy
typedef enum {
    LOAD_BALANCE_STATIC = 0,
    LOAD_BALANCE_DYNAMIC = 1,
    LOAD_BALANCE_GUIDED = 2,
    LOAD_BALANCE_AUTO = 3
} load_balance_t;

// Memory allocation strategy
typedef enum {
    MEMORY_BASIC = 0,
    MEMORY_PINNED = 1,
    MEMORY_UNIFIED = 2,
    MEMORY_MANAGED = 3
} memory_strategy_t;

// Hybrid engine configuration structure
typedef struct {
    // Processing configuration
    processing_mode_t mode;
    int use_gpu;
    int omp_threads;
    int cuda_block_size;
    int cuda_grid_size;
    
    // Performance tuning
    load_balance_t load_balance;
    memory_strategy_t memory_strategy;
    float cpu_gpu_ratio;  // 0.0 = all GPU, 1.0 = all CPU
    int batch_size;
    int prefetch_enabled;
    
    // Resource limits
    size_t max_gpu_memory;
    size_t max_cpu_memory;
    int max_concurrent_streams;
    int enable_peer_access;
    
    // Optimization flags
    int enable_memory_coalescing;
    int enable_async_processing;
    int enable_kernel_fusion;
    int enable_auto_tuning;
} hybrid_config_t;

// Document structure for hybrid processing
typedef struct {
    int doc_id;
    char filename[MAX_FILENAME_LENGTH];
    char* content;
    int content_length;
    int token_count;
    char** tokens;
    float* term_frequencies;
    int processed_on_gpu;  // Flag indicating where it was processed
} hybrid_document_t;

// Query structure for hybrid processing
typedef struct {
    char query_text[MAX_QUERY_LENGTH];
    char** terms;
    int term_count;
    float* term_weights;
    int use_gpu_scoring;
} hybrid_query_t;

// Search result with performance metrics
typedef struct {
    int doc_id;
    float score;
    float cpu_time;
    float gpu_time;
    int processing_location;  // 0=CPU, 1=GPU, 2=Hybrid
} hybrid_result_t;

// Performance monitoring structure
typedef struct {
    // Timing metrics
    double total_time;
    double cpu_time;
    double gpu_time;
    double memory_transfer_time;
    double io_time;
    
    // Resource utilization
    float cpu_utilization;
    float gpu_utilization;
    float memory_bandwidth_utilization;
    size_t peak_cpu_memory;
    size_t peak_gpu_memory;
    
    // Processing statistics
    int documents_processed_cpu;
    int documents_processed_gpu;
    int queries_processed_cpu;
    int queries_processed_gpu;
    
    // Efficiency metrics
    float parallel_efficiency;
    float load_balance_factor;
    float gpu_acceleration_factor;
    float energy_efficiency;
    
    // Error and warning counts
    int cuda_errors;
    int omp_warnings;
    int memory_errors;
} hybrid_performance_t;

// Work distribution structure
typedef struct {
    int total_work_units;
    int cpu_work_units;
    int gpu_work_units;
    int* cpu_assignment;
    int* gpu_assignment;
    float estimated_cpu_time;
    float estimated_gpu_time;
} work_distribution_t;

// Global hybrid engine state
extern hybrid_config_t g_hybrid_config;
extern hybrid_performance_t g_hybrid_performance;
extern int g_hybrid_initialized;

// Core hybrid engine functions
int hybrid_engine_init(hybrid_config_t* config);
void hybrid_engine_cleanup(void);
int hybrid_engine_auto_configure(void);
void hybrid_engine_print_config(void);

// Document processing functions
int hybrid_process_documents(hybrid_document_t* documents, int num_docs);
int hybrid_build_index(const char* dataset_path);
int hybrid_add_document(const char* filename, const char* content);
void hybrid_clear_index(void);

// Search functions
int hybrid_search(hybrid_query_t* query, hybrid_result_t* results, int max_results);
int hybrid_batch_search(hybrid_query_t* queries, int num_queries, 
                        hybrid_result_t** results, int* result_counts, int max_results_per_query);
float hybrid_calculate_bm25_score(int doc_id, const char* term);

// Load balancing and work distribution
work_distribution_t* hybrid_analyze_workload(void* data, int data_size, int complexity);
int hybrid_distribute_work(work_distribution_t* distribution);
void hybrid_rebalance_workload(float cpu_efficiency, float gpu_efficiency);

// Memory management
void* hybrid_allocate_memory(size_t size, int prefer_gpu);
void hybrid_free_memory(void* ptr);
int hybrid_transfer_data(void* dst, void* src, size_t size, int to_gpu);
void hybrid_prefetch_data(void* ptr, size_t size, int target_device);

// Performance monitoring and optimization
void hybrid_start_timer(const char* operation);
void hybrid_stop_timer(const char* operation);
hybrid_performance_t* hybrid_get_performance_metrics(void);
void hybrid_reset_performance_counters(void);
void hybrid_print_performance_report(void);

// Auto-tuning functions
int hybrid_auto_tune_parameters(void);
void hybrid_benchmark_configuration(hybrid_config_t* config, float* performance_score);
hybrid_config_t* hybrid_find_optimal_configuration(void);

// CPU-specific OpenMP functions
void hybrid_set_omp_threads(int num_threads);
int hybrid_get_omp_threads(void);
void hybrid_set_omp_schedule(const char* schedule_type, int chunk_size);
void hybrid_cpu_process_documents(hybrid_document_t* documents, int start_idx, int end_idx);
void hybrid_cpu_search_parallel(hybrid_query_t* query, hybrid_result_t* results, 
                                int start_doc, int end_doc);

// GPU-specific CUDA functions (when available)
#ifdef USE_CUDA
int hybrid_gpu_is_available(void);
int hybrid_gpu_init(void);
void hybrid_gpu_cleanup(void);
void hybrid_gpu_process_documents(hybrid_document_t* documents, int num_docs);
void hybrid_gpu_search_parallel(hybrid_query_t* query, hybrid_result_t* results, int num_docs);
float hybrid_gpu_get_utilization(void);
size_t hybrid_gpu_get_memory_usage(void);
#endif

// Debugging and profiling functions
void hybrid_enable_profiling(const char* output_file);
void hybrid_disable_profiling(void);
void hybrid_print_memory_usage(void);
void hybrid_validate_results(hybrid_result_t* cpu_results, hybrid_result_t* gpu_results, 
                             int num_results, float tolerance);

// Configuration file I/O
int hybrid_load_config(const char* config_file);
int hybrid_save_config(const char* config_file);
void hybrid_set_default_config(hybrid_config_t* config);

// Utility functions
double hybrid_get_wall_time(void);
size_t hybrid_get_available_memory(int device);  // -1 for CPU, >=0 for GPU
int hybrid_is_memory_sufficient(size_t required_memory, int device);
void hybrid_print_system_info(void);

// Error handling
typedef enum {
    HYBRID_SUCCESS = 0,
    HYBRID_ERROR_INIT = -1,
    HYBRID_ERROR_MEMORY = -2,
    HYBRID_ERROR_CUDA = -3,
    HYBRID_ERROR_OMP = -4,
    HYBRID_ERROR_IO = -5,
    HYBRID_ERROR_CONFIG = -6
} hybrid_error_t;

const char* hybrid_get_error_string(hybrid_error_t error);
void hybrid_set_error_callback(void (*callback)(hybrid_error_t, const char*));

// Thread-safe operations
void hybrid_lock_index(void);
void hybrid_unlock_index(void);
void hybrid_lock_results(void);
void hybrid_unlock_results(void);

// Optimization hints
typedef struct {
    int data_size;
    int complexity_level;  // 1-10 scale
    int memory_intensive;
    int compute_intensive;
    int io_intensive;
    float recommended_cpu_ratio;
    float recommended_gpu_ratio;
    const char* optimization_notes;
} hybrid_optimization_hint_t;

hybrid_optimization_hint_t* hybrid_analyze_workload_characteristics(void* data, int size);
void hybrid_apply_optimization_hints(hybrid_optimization_hint_t* hints);

#ifdef __cplusplus
}
#endif

#endif // HYBRID_ENGINE_H
