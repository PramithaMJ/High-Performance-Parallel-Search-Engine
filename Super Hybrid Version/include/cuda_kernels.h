#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

// CUDA kernel constants
#define MAX_TERM_LENGTH 256
#define MAX_DOCUMENT_LENGTH 10000
#define CUDA_THREADS_PER_BLOCK 256
#define MAX_CUDA_STREAMS 16

// GPU device information structure
typedef struct {
    int device_count;
    int current_device;
    size_t total_memory;
    size_t free_memory;
    int compute_capability_major;
    int compute_capability_minor;
    int multiprocessor_count;
    int max_threads_per_block;
    char device_name[256];
} gpu_device_info_t;

// GPU performance metrics
typedef struct {
    double kernel_execution_time;
    double memory_transfer_time;
    double total_gpu_time;
    size_t bytes_transferred_h2d;
    size_t bytes_transferred_d2h;
    int kernel_launches;
    float gpu_utilization;
} gpu_performance_t;

// CUDA function prototypes

// Device management
int cuda_initialize_device(int device_id);
int cuda_finalize_device(void);
int cuda_get_device_info(gpu_device_info_t* info);
int cuda_set_memory_pool(size_t pool_size);

// Memory management
void* cuda_malloc_managed(size_t size);
int cuda_free_managed(void* ptr);
int cuda_memcpy_async(void* dst, const void* src, size_t size, 
                      cudaMemcpyKind kind, cudaStream_t stream);

// Kernel launches for search engine operations

// Document tokenization kernel
__global__ void gpu_tokenize_documents_kernel(
    char* documents, int* doc_offsets, int* doc_lengths,
    char* tokens, int* token_offsets, int* token_counts,
    int num_docs, int max_tokens_per_doc);

// BM25 scoring kernel
__global__ void gpu_bm25_scoring_kernel(
    float* doc_vectors, float* query_vector, float* scores,
    int* doc_lengths, float avg_doc_length, float k1, float b,
    int num_docs, int num_terms);

// Text similarity kernel
__global__ void gpu_text_similarity_kernel(
    char* text1, char* text2, float* similarity_scores,
    int* text1_lengths, int* text2_lengths, int num_pairs);

// Parallel reduction kernel for aggregating results
__global__ void gpu_parallel_reduction_kernel(
    float* input, float* output, int n);

// String matching kernel for fast text search
__global__ void gpu_string_matching_kernel(
    char* haystack, char* needle, int* match_positions,
    int haystack_length, int needle_length);

// Host-side wrapper functions
int cuda_tokenize_documents(const char** documents, int num_docs,
                           char*** tokens, int** token_counts);

int cuda_compute_bm25_scores(float* doc_vectors, float* query_vector,
                            float* scores, int num_docs, int num_terms,
                            float k1, float b, float avg_doc_length);

int cuda_compute_text_similarity(const char** text1, const char** text2,
                                float* similarities, int num_pairs);

int cuda_parallel_search(const char* query, const char** documents,
                        int num_docs, int* result_indices, float* scores,
                        int max_results);

// Performance monitoring
int cuda_start_timing(void);
float cuda_stop_timing(void);
int cuda_get_memory_usage(size_t* used, size_t* total);
float cuda_get_utilization(void);

// Stream management
int cuda_create_streams(cudaStream_t** streams, int num_streams);
int cuda_destroy_streams(cudaStream_t* streams, int num_streams);
int cuda_synchronize_streams(cudaStream_t* streams, int num_streams);

#endif // USE_CUDA

#endif // CUDA_KERNELS_H
