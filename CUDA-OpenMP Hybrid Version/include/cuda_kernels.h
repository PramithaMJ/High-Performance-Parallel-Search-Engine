#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#ifdef __cplusplus
extern "C" {
#endif

// CUDA configuration constants
#define CUDA_BLOCK_SIZE 256
#define CUDA_MAX_THREADS_PER_BLOCK 1024
#define CUDA_MAX_SHARED_MEMORY 48 * 1024  // 48KB shared memory
#define CUDA_WARP_SIZE 32
#define MAX_GPU_MEMORY_GB 16

// Performance optimization constants
#define GPU_MEMORY_ALIGNMENT 256
#define GPU_BATCH_SIZE 10000
#define GPU_STRING_MAX_LENGTH 512
#define GPU_DOCS_PER_KERNEL 1000

// CUDA error checking macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUDA_CHECK_KERNEL() \
    do { \
        cudaError_t error = cudaGetLastError(); \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA kernel error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
        CUDA_CHECK(cudaDeviceSynchronize()); \
    } while(0)

// GPU device information structure
typedef struct {
    int device_id;
    char name[256];
    size_t total_memory;
    size_t free_memory;
    int major_version;
    int minor_version;
    int multiprocessor_count;
    int max_threads_per_block;
    int max_threads_per_multiprocessor;
    int warp_size;
    int concurrent_kernels;
    float memory_clock_rate;
    float gpu_clock_rate;
} gpu_device_info_t;

// GPU memory management structure
typedef struct {
    void* d_ptr;           // Device pointer
    void* h_ptr;           // Host pointer
    size_t size;           // Size in bytes
    int is_pinned;         // Whether host memory is pinned
    int is_unified;        // Whether using unified memory
    cudaStream_t stream;   // CUDA stream for async operations
} gpu_memory_t;

// BM25 calculation structure for GPU
typedef struct {
    float* scores;         // Output scores array
    int* doc_ids;          // Document IDs
    float* tf_values;      // Term frequency values
    float* df_values;      // Document frequency values
    float* doc_lengths;    // Document lengths
    float avg_doc_length;  // Average document length
    int num_docs;          // Number of documents
    int num_terms;         // Number of query terms
    float k1;              // BM25 k1 parameter
    float b;               // BM25 b parameter
} gpu_bm25_data_t;

// String matching structure for GPU
typedef struct {
    char* documents;       // Concatenated documents
    char* patterns;        // Search patterns
    int* doc_offsets;      // Document start offsets
    int* pattern_lengths;  // Pattern lengths
    int* results;          // Match results
    int num_docs;          // Number of documents
    int num_patterns;      // Number of patterns
    int max_doc_length;    // Maximum document length
} gpu_string_data_t;

// Performance metrics structure
typedef struct {
    float kernel_time;           // GPU kernel execution time
    float memory_transfer_time;  // CPU-GPU transfer time
    float total_gpu_time;        // Total GPU processing time
    float gpu_utilization;       // GPU utilization percentage
    size_t gpu_memory_used;      // GPU memory usage
    int cuda_cores_used;         // Number of CUDA cores utilized
    float memory_bandwidth;      // Memory bandwidth utilization
    float compute_efficiency;    // Compute efficiency ratio
} gpu_performance_t;

// Function declarations for GPU device management
int cuda_initialize_device(int device_id);
void cuda_cleanup_device(void);
gpu_device_info_t* cuda_get_device_info(int device_id);
int cuda_get_optimal_block_size(int function_id);
void cuda_print_device_properties(void);

// Memory management functions
gpu_memory_t* cuda_allocate_memory(size_t size, int use_pinned, int use_unified);
void cuda_free_memory(gpu_memory_t* mem);
int cuda_transfer_to_device(gpu_memory_t* mem, void* host_data, size_t size);
int cuda_transfer_from_device(gpu_memory_t* mem, void* host_data, size_t size);
void cuda_prefetch_unified_memory(void* ptr, size_t size, int device);

// CUDA kernel declarations
__global__ void gpu_tokenize_documents_kernel(
    char* documents, int* doc_offsets, int* doc_lengths, 
    char* tokens, int* token_offsets, int* token_counts,
    int num_docs, int max_tokens_per_doc
);

__global__ void gpu_bm25_scoring_kernel(
    float* scores, int* doc_ids, float* tf_values, float* idf_values,
    float* doc_lengths, float avg_doc_length, int num_docs, int num_terms,
    float k1, float b
);

__global__ void gpu_string_matching_kernel(
    char* documents, char* patterns, int* doc_offsets, int* pattern_lengths,
    int* match_results, int num_docs, int num_patterns, int max_doc_length
);

__global__ void gpu_vector_similarity_kernel(
    float* vec1, float* vec2, float* results, int num_vectors, int vector_dim
);

__global__ void gpu_parallel_sort_kernel(
    float* keys, int* values, int num_elements, int ascending
);

__global__ void gpu_text_preprocessing_kernel(
    char* input_text, char* output_text, int* text_lengths, 
    int num_texts, int convert_lowercase, int remove_punctuation
);

// High-level GPU processing functions
int gpu_process_documents(char** documents, int num_docs, char*** tokens, int** token_counts);
int gpu_calculate_bm25_scores(char* query, char** documents, int num_docs, 
                              float* scores, int* doc_ids, int top_k);
int gpu_parallel_string_search(char* pattern, char** documents, int num_docs, 
                               int** results, int* num_matches);
int gpu_batch_process_queries(char** queries, int num_queries, char** documents, 
                              int num_docs, float** all_scores, int** all_doc_ids);

// Performance optimization functions
void gpu_optimize_memory_access(void* data, size_t size);
int gpu_auto_tune_parameters(int data_size, int* optimal_block_size, int* optimal_grid_size);
void gpu_enable_peer_access(int device1, int device2);
gpu_performance_t* gpu_get_performance_metrics(void);
void gpu_reset_performance_counters(void);

// Stream and async processing
cudaStream_t* cuda_create_streams(int num_streams);
void cuda_destroy_streams(cudaStream_t* streams, int num_streams);
int cuda_async_memory_copy(gpu_memory_t* mem, void* data, size_t size, 
                           cudaMemcpyKind kind, cudaStream_t stream);
void cuda_synchronize_streams(cudaStream_t* streams, int num_streams);

// Multi-GPU support functions
int cuda_get_device_count(void);
int cuda_enable_multi_gpu(int* device_ids, int num_devices);
void cuda_distribute_workload(void* data, size_t size, int num_devices, 
                              void** device_data, size_t* device_sizes);

// Error handling and debugging
void cuda_print_error(cudaError_t error, const char* file, int line);
void cuda_check_memory_usage(void);
void cuda_profile_kernel_performance(const char* kernel_name, 
                                     cudaEvent_t start, cudaEvent_t stop);

// Utility functions
float cuda_get_memory_bandwidth_utilization(void);
float cuda_get_compute_utilization(void);
int cuda_is_memory_coalesced(void* ptr, size_t access_size);
void cuda_warmup_gpu(void);

#ifdef __cplusplus
}
#endif

#endif // CUDA_KERNELS_H
