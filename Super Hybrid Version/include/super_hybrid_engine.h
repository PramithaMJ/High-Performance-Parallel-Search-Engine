#ifndef SUPER_HYBRID_ENGINE_H
#define SUPER_HYBRID_ENGINE_H

#include <stddef.h>

// Error codes
#define HYBRID_SUCCESS 0
#define HYBRID_ERROR -1
#define HYBRID_ERROR_CUDA -2
#define HYBRID_ERROR_OPENMP -3
#define HYBRID_ERROR_MPI -4

// Configuration structure for super hybrid engine
typedef struct {
    int use_cuda;
    int use_openmp;
    int use_mpi;
    int cuda_devices;
    int openmp_threads;
    int mpi_processes;
    int gpu_batch_size;
    int memory_pool_size;
    float gpu_cpu_ratio;
    int adaptive_scheduling;
    int pipeline_depth;
} hybrid_config_t;

// Performance metrics structure
typedef struct {
    double gpu_time;
    double cpu_time;
    double mpi_comm_time;
    double total_time;
    double gpu_utilization;
    double cpu_utilization;
    int documents_processed;
    int queries_processed;
    size_t memory_used_gpu;
    size_t memory_used_cpu;
} hybrid_metrics_t;

// Function prototypes
int hybrid_engine_init(const hybrid_config_t* config);
int hybrid_engine_finalize(void);
void hybrid_set_default_config(hybrid_config_t* config);
int hybrid_process_documents(const char** documents, int num_docs);
int hybrid_search_query(const char* query, void* results, int max_results);
void hybrid_get_metrics(hybrid_metrics_t* metrics);

// Utility functions
int hybrid_detect_cuda_devices(void);
int hybrid_get_optimal_threads(void);
void hybrid_print_configuration(const hybrid_config_t* config);

#endif // SUPER_HYBRID_ENGINE_H
