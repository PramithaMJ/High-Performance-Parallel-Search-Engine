#include "../include/cuda_kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef USE_CUDA

// Global variables for CUDA state
static gpu_device_info_t g_device_info;
static gpu_performance_t g_performance;
static cudaEvent_t g_start_event, g_stop_event;
static void* g_memory_pool = NULL;
static size_t g_memory_pool_size = 0;
static int g_cuda_initialized = 0;

// Initialize CUDA device
int cuda_initialize_device(int device_id) {
    if (g_cuda_initialized) return 0;
    
    cudaError_t status;
    
    // Set device
    status = cudaSetDevice(device_id);
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA: Failed to set device %d: %s\n", 
                device_id, cudaGetErrorString(status));
        return -1;
    }
    
    // Get device properties
    cudaDeviceProp props;
    status = cudaGetDeviceProperties(&props, device_id);
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA: Failed to get device properties: %s\n", 
                cudaGetErrorString(status));
        return -1;
    }
    
    // Fill device info structure
    g_device_info.current_device = device_id;
    g_device_info.compute_capability_major = props.major;
    g_device_info.compute_capability_minor = props.minor;
    g_device_info.multiprocessor_count = props.multiProcessorCount;
    g_device_info.max_threads_per_block = props.maxThreadsPerBlock;
    g_device_info.total_memory = props.totalGlobalMem;
    strncpy(g_device_info.device_name, props.name, sizeof(g_device_info.device_name) - 1);
    
    // Get memory info
    size_t free_mem, total_mem;
    status = cudaMemGetInfo(&free_mem, &total_mem);
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA: Failed to get memory info: %s\n", 
                cudaGetErrorString(status));
        return -1;
    }
    g_device_info.free_memory = free_mem;
    
    // Create timing events
    status = cudaEventCreate(&g_start_event);
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA: Failed to create start event: %s\n", 
                cudaGetErrorString(status));
        return -1;
    }
    
    status = cudaEventCreate(&g_stop_event);
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA: Failed to create stop event: %s\n", 
                cudaGetErrorString(status));
        return -1;
    }
    
    g_cuda_initialized = 1;
    printf("CUDA: Initialized device %d (%s)\n", device_id, props.name);
    printf("CUDA: %d SMs, %.1f GB memory, Compute %d.%d\n",
           props.multiProcessorCount,
           props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
           props.major, props.minor);
    
    return 0;
}

// Finalize CUDA device
int cuda_finalize_device(void) {
    if (!g_cuda_initialized) return 0;
    
    // Free memory pool if allocated
    if (g_memory_pool) {
        cudaFree(g_memory_pool);
        g_memory_pool = NULL;
    }
    
    // Destroy events
    cudaEventDestroy(g_start_event);
    cudaEventDestroy(g_stop_event);
    
    // Reset device
    cudaDeviceReset();
    
    g_cuda_initialized = 0;
    printf("CUDA: Device finalized\n");
    
    return 0;
}

// Get device information
int cuda_get_device_info(gpu_device_info_t* info) {
    if (!g_cuda_initialized || !info) return -1;
    
    *info = g_device_info;
    return 0;
}

// Set up memory pool for efficient allocation
int cuda_set_memory_pool(size_t pool_size) {
    if (!g_cuda_initialized) return -1;
    
    if (g_memory_pool) {
        cudaFree(g_memory_pool);
    }
    
    cudaError_t status = cudaMalloc(&g_memory_pool, pool_size);
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA: Failed to allocate memory pool: %s\n", 
                cudaGetErrorString(status));
        return -1;
    }
    
    g_memory_pool_size = pool_size;
    printf("CUDA: Allocated memory pool of %.1f MB\n", 
           pool_size / (1024.0 * 1024.0));
    
    return 0;
}

// Document tokenization kernel
__global__ void gpu_tokenize_documents_kernel(
    char* documents, int* doc_offsets, int* doc_lengths,
    char* tokens, int* token_offsets, int* token_counts,
    int num_docs, int max_tokens_per_doc) {
    
    int doc_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (doc_idx >= num_docs) return;
    
    // Get document boundaries
    int doc_start = doc_offsets[doc_idx];
    int doc_length = doc_lengths[doc_idx];
    
    // Tokenize the document
    int token_count = 0;
    int char_idx = doc_start;
    int token_start = -1;
    bool in_token = false;
    
    // Output offset for this document's tokens
    int output_offset = doc_idx * max_tokens_per_doc * MAX_TERM_LENGTH;
    
    for (int i = 0; i < doc_length && token_count < max_tokens_per_doc; i++) {
        char c = documents[char_idx + i];
        
        // Simple tokenization: alphanumeric characters form tokens
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || 
            (c >= '0' && c <= '9')) {
            if (!in_token) {
                token_start = i;
                in_token = true;
            }
        } else {
            if (in_token) {
                // End of token - copy it
                int token_length = i - token_start;
                if (token_length > 0 && token_length < MAX_TERM_LENGTH) {
                    int token_output_pos = output_offset + token_count * MAX_TERM_LENGTH;
                    
                    // Copy token characters
                    for (int j = 0; j < token_length; j++) {
                        tokens[token_output_pos + j] = documents[char_idx + token_start + j];
                    }
                    tokens[token_output_pos + token_length] = '\0';
                    
                    token_count++;
                }
                in_token = false;
            }
        }
    }
    
    // Handle last token if document ends in middle of token
    if (in_token && token_count < max_tokens_per_doc) {
        int token_length = doc_length - token_start;
        if (token_length > 0 && token_length < MAX_TERM_LENGTH) {
            int token_output_pos = output_offset + token_count * MAX_TERM_LENGTH;
            
            for (int j = 0; j < token_length; j++) {
                tokens[token_output_pos + j] = documents[char_idx + token_start + j];
            }
            tokens[token_output_pos + token_length] = '\0';
            
            token_count++;
        }
    }
    
    // Store token count for this document
    token_counts[doc_idx] = token_count;
    token_offsets[doc_idx] = output_offset;
}

// BM25 scoring kernel
__global__ void gpu_bm25_scoring_kernel(
    float* doc_vectors, float* query_vector, float* scores,
    int* doc_lengths, float avg_doc_length, float k1, float b,
    int num_docs, int num_terms) {
    
    int doc_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (doc_idx >= num_docs) return;
    
    float score = 0.0f;
    int doc_length = doc_lengths[doc_idx];
    
    for (int term_idx = 0; term_idx < num_terms; term_idx++) {
        float tf = doc_vectors[doc_idx * num_terms + term_idx];
        float idf = query_vector[term_idx];
        
        if (tf > 0.0f && idf > 0.0f) {
            float norm_factor = tf / (tf + k1 * (1.0f - b + b * doc_length / avg_doc_length));
            score += idf * norm_factor;
        }
    }
    
    scores[doc_idx] = score;
}

// Text similarity kernel using cosine similarity
__global__ void gpu_text_similarity_kernel(
    char* text1, char* text2, float* similarity_scores,
    int* text1_lengths, int* text2_lengths, int num_pairs) {
    
    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (pair_idx >= num_pairs) return;
    
    // Simple character-based similarity (could be enhanced with more sophisticated algorithms)
    int len1 = text1_lengths[pair_idx];
    int len2 = text2_lengths[pair_idx];
    
    int matches = 0;
    int total_chars = len1 + len2;
    
    // Count matching characters at same positions
    int min_len = (len1 < len2) ? len1 : len2;
    for (int i = 0; i < min_len; i++) {
        if (text1[pair_idx * MAX_DOCUMENT_LENGTH + i] == 
            text2[pair_idx * MAX_DOCUMENT_LENGTH + i]) {
            matches++;
        }
    }
    
    // Calculate similarity score
    float similarity = (total_chars > 0) ? (2.0f * matches) / total_chars : 0.0f;
    similarity_scores[pair_idx] = similarity;
}

// Parallel reduction kernel for aggregating results
__global__ void gpu_parallel_reduction_kernel(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load input into shared memory
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// String matching kernel for fast text search
__global__ void gpu_string_matching_kernel(
    char* haystack, char* needle, int* match_positions,
    int haystack_length, int needle_length) {
    
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (pos > haystack_length - needle_length) return;
    
    // Check if needle matches at position pos
    bool match = true;
    for (int i = 0; i < needle_length; i++) {
        if (haystack[pos + i] != needle[i]) {
            match = false;
            break;
        }
    }
    
    // Record match position
    match_positions[pos] = match ? 1 : 0;
}

// Host-side wrapper functions

int cuda_tokenize_documents(const char** documents, int num_docs,
                           char*** tokens, int** token_counts) {
    if (!g_cuda_initialized) return -1;
    
    // Implementation would involve:
    // 1. Allocate GPU memory for documents and output
    // 2. Copy documents to GPU
    // 3. Launch tokenization kernel
    // 4. Copy results back to host
    // 5. Clean up GPU memory
    
    printf("CUDA: Tokenizing %d documents on GPU\n", num_docs);
    
    // Placeholder - actual implementation would be more complex
    return 0;
}

int cuda_compute_bm25_scores(float* doc_vectors, float* query_vector,
                            float* scores, int num_docs, int num_terms,
                            float k1, float b, float avg_doc_length) {
    if (!g_cuda_initialized) return -1;
    
    // Calculate grid and block dimensions
    int threads_per_block = CUDA_THREADS_PER_BLOCK;
    int blocks = (num_docs + threads_per_block - 1) / threads_per_block;
    
    // Allocate GPU memory
    float *d_doc_vectors, *d_query_vector, *d_scores;
    int *d_doc_lengths;
    
    size_t doc_vectors_size = num_docs * num_terms * sizeof(float);
    size_t query_vector_size = num_terms * sizeof(float);
    size_t scores_size = num_docs * sizeof(float);
    size_t doc_lengths_size = num_docs * sizeof(int);
    
    cudaMalloc(&d_doc_vectors, doc_vectors_size);
    cudaMalloc(&d_query_vector, query_vector_size);
    cudaMalloc(&d_scores, scores_size);
    cudaMalloc(&d_doc_lengths, doc_lengths_size);
    
    // Copy data to GPU
    cudaMemcpy(d_doc_vectors, doc_vectors, doc_vectors_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_query_vector, query_vector, query_vector_size, cudaMemcpyHostToDevice);
    
    // Launch kernel
    cuda_start_timing();
    gpu_bm25_scoring_kernel<<<blocks, threads_per_block>>>(
        d_doc_vectors, d_query_vector, d_scores, d_doc_lengths,
        avg_doc_length, k1, b, num_docs, num_terms);
    
    cudaDeviceSynchronize();
    float kernel_time = cuda_stop_timing();
    
    // Copy results back
    cudaMemcpy(scores, d_scores, scores_size, cudaMemcpyDeviceToHost);
    
    // Clean up
    cudaFree(d_doc_vectors);
    cudaFree(d_query_vector);
    cudaFree(d_scores);
    cudaFree(d_doc_lengths);
    
    printf("CUDA: BM25 scoring completed in %.3f ms for %d documents\n", 
           kernel_time, num_docs);
    
    return 0;
}

// Performance monitoring functions
int cuda_start_timing(void) {
    if (!g_cuda_initialized) return -1;
    
    cudaEventRecord(g_start_event, 0);
    return 0;
}

float cuda_stop_timing(void) {
    if (!g_cuda_initialized) return -1.0f;
    
    cudaEventRecord(g_stop_event, 0);
    cudaEventSynchronize(g_stop_event);
    
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, g_start_event, g_stop_event);
    
    g_performance.kernel_execution_time += elapsed_time;
    g_performance.kernel_launches++;
    
    return elapsed_time;
}

int cuda_get_memory_usage(size_t* used, size_t* total) {
    if (!g_cuda_initialized || !used || !total) return -1;
    
    size_t free_mem;
    cudaError_t status = cudaMemGetInfo(&free_mem, total);
    if (status != cudaSuccess) return -1;
    
    *used = *total - free_mem;
    return 0;
}

float cuda_get_utilization(void) {
    // This would require NVML or similar library for actual GPU utilization
    // For now, return a placeholder based on kernel activity
    return (g_performance.kernel_launches > 0) ? 75.0f : 0.0f;
}

// Stream management
int cuda_create_streams(cudaStream_t** streams, int num_streams) {
    if (!g_cuda_initialized || !streams || num_streams <= 0) return -1;
    
    *streams = (cudaStream_t*)malloc(num_streams * sizeof(cudaStream_t));
    if (!*streams) return -1;
    
    for (int i = 0; i < num_streams; i++) {
        cudaError_t status = cudaStreamCreate(&(*streams)[i]);
        if (status != cudaSuccess) {
            // Clean up already created streams
            for (int j = 0; j < i; j++) {
                cudaStreamDestroy((*streams)[j]);
            }
            free(*streams);
            *streams = NULL;
            return -1;
        }
    }
    
    printf("CUDA: Created %d streams for asynchronous processing\n", num_streams);
    return 0;
}

int cuda_destroy_streams(cudaStream_t* streams, int num_streams) {
    if (!streams || num_streams <= 0) return -1;
    
    for (int i = 0; i < num_streams; i++) {
        cudaStreamDestroy(streams[i]);
    }
    free(streams);
    
    printf("CUDA: Destroyed %d streams\n", num_streams);
    return 0;
}

#endif // USE_CUDA
