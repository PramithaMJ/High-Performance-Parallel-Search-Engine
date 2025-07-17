#include "cuda_kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cub/cub.cuh>  // For CUB library optimized primitives

// Global variables for device management
static gpu_device_info_t g_device_info;
static int g_device_initialized = 0;
static cudaStream_t* g_streams = NULL;
static int g_num_streams = 0;
static gpu_performance_t g_performance;

// Device properties cache
static cudaDeviceProp g_device_props;

// Memory pools for efficient allocation
static void* g_gpu_memory_pool = NULL;
static size_t g_memory_pool_size = 0;
static size_t g_memory_pool_offset = 0;

// Performance counters
static cudaEvent_t g_start_event, g_stop_event;

// CUDA kernel implementations

/**
 * GPU kernel for parallel document tokenization
 * Each thread processes one document and extracts tokens
 */
__global__ void gpu_tokenize_documents_kernel(
    char* documents, int* doc_offsets, int* doc_lengths, 
    char* tokens, int* token_offsets, int* token_counts,
    int num_docs, int max_tokens_per_doc) {
    
    int doc_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (doc_idx >= num_docs) return;
    
    // Get document boundaries
    int doc_start = doc_offsets[doc_idx];
    int doc_length = doc_lengths[doc_idx];
    int doc_end = doc_start + doc_length;
    
    // Token extraction state
    int token_count = 0;
    int current_token_start = -1;
    int token_start_offset = doc_idx * max_tokens_per_doc * MAX_TERM_LENGTH;
    
    // Process each character in the document
    for (int i = doc_start; i < doc_end && token_count < max_tokens_per_doc; i++) {
        char c = documents[i];
        
        // Check if character is alphanumeric
        bool is_alpha = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || 
                       (c >= '0' && c <= '9');
        
        if (is_alpha) {
            if (current_token_start == -1) {
                current_token_start = i;  // Start of new token
            }
        } else {
            if (current_token_start != -1) {
                // End of token - copy it
                int token_length = i - current_token_start;
                if (token_length > 0 && token_length < MAX_TERM_LENGTH) {
                    int token_offset = token_start_offset + token_count * MAX_TERM_LENGTH;
                    
                    // Copy token and convert to lowercase
                    for (int j = 0; j < token_length; j++) {
                        char ch = documents[current_token_start + j];
                        if (ch >= 'A' && ch <= 'Z') {
                            ch = ch - 'A' + 'a';  // Convert to lowercase
                        }
                        tokens[token_offset + j] = ch;
                    }
                    tokens[token_offset + token_length] = '\\0';
                    
                    token_offsets[doc_idx * max_tokens_per_doc + token_count] = token_offset;
                    token_count++;
                }
                current_token_start = -1;
            }
        }
    }
    
    // Handle token at end of document
    if (current_token_start != -1 && token_count < max_tokens_per_doc) {
        int token_length = doc_end - current_token_start;
        if (token_length > 0 && token_length < MAX_TERM_LENGTH) {
            int token_offset = token_start_offset + token_count * MAX_TERM_LENGTH;
            
            for (int j = 0; j < token_length; j++) {
                char ch = documents[current_token_start + j];
                if (ch >= 'A' && ch <= 'Z') {
                    ch = ch - 'A' + 'a';
                }
                tokens[token_offset + j] = ch;
            }
            tokens[token_offset + token_length] = '\\0';
            
            token_offsets[doc_idx * max_tokens_per_doc + token_count] = token_offset;
            token_count++;
        }
    }
    
    token_counts[doc_idx] = token_count;
}

/**
 * GPU kernel for parallel BM25 score calculation
 * Each thread calculates BM25 score for one document-term pair
 */
__global__ void gpu_bm25_scoring_kernel(
    float* scores, int* doc_ids, float* tf_values, float* idf_values,
    float* doc_lengths, float avg_doc_length, int num_docs, int num_terms,
    float k1, float b) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = num_docs * num_terms;
    
    if (idx >= total_pairs) return;
    
    int doc_idx = idx / num_terms;
    int term_idx = idx % num_terms;
    
    float tf = tf_values[idx];
    float idf = idf_values[term_idx];
    float doc_length = doc_lengths[doc_idx];
    
    // BM25 formula: IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (|d| / avgdl)))
    float denominator = tf + k1 * (1.0f - b + b * (doc_length / avg_doc_length));
    float bm25_score = idf * (tf * (k1 + 1.0f)) / denominator;
    
    scores[idx] = bm25_score;
    doc_ids[idx] = doc_idx;
}

/**
 * GPU kernel for parallel string matching
 * Each thread searches for patterns in one document
 */
__global__ void gpu_string_matching_kernel(
    char* documents, char* patterns, int* doc_offsets, int* pattern_lengths,
    int* match_results, int num_docs, int num_patterns, int max_doc_length) {
    
    int doc_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (doc_idx >= num_docs) return;
    
    int doc_start = doc_offsets[doc_idx];
    int doc_end = doc_start + max_doc_length;
    
    // Shared memory for pattern caching
    extern __shared__ char shared_patterns[];
    
    // Load patterns into shared memory
    if (threadIdx.x < num_patterns && threadIdx.x < blockDim.x) {
        int pattern_offset = threadIdx.x * GPU_STRING_MAX_LENGTH;
        int pattern_len = pattern_lengths[threadIdx.x];
        for (int i = 0; i < pattern_len && i < GPU_STRING_MAX_LENGTH - 1; i++) {
            shared_patterns[pattern_offset + i] = patterns[threadIdx.x * GPU_STRING_MAX_LENGTH + i];
        }
        shared_patterns[pattern_offset + pattern_len] = '\\0';
    }
    
    __syncthreads();
    
    // Search for each pattern in this document
    for (int pattern_idx = 0; pattern_idx < num_patterns; pattern_idx++) {
        int pattern_len = pattern_lengths[pattern_idx];
        char* pattern = &shared_patterns[pattern_idx * GPU_STRING_MAX_LENGTH];
        int matches = 0;
        
        // Boyer-Moore-like search
        for (int doc_pos = doc_start; doc_pos <= doc_end - pattern_len; doc_pos++) {
            bool match = true;
            for (int i = 0; i < pattern_len; i++) {
                if (documents[doc_pos + i] != pattern[i]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                matches++;
            }
        }
        
        match_results[doc_idx * num_patterns + pattern_idx] = matches;
    }
}

/**
 * GPU kernel for vector similarity calculation
 * Each thread calculates similarity between two vectors
 */
__global__ void gpu_vector_similarity_kernel(
    float* vec1, float* vec2, float* results, int num_vectors, int vector_dim) {
    
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (vec_idx >= num_vectors) return;
    
    float dot_product = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;
    
    int vec1_offset = vec_idx * vector_dim;
    int vec2_offset = vec_idx * vector_dim;
    
    // Calculate dot product and norms using vectorized operations
    for (int i = 0; i < vector_dim; i++) {
        float v1 = vec1[vec1_offset + i];
        float v2 = vec2[vec2_offset + i];
        
        dot_product += v1 * v2;
        norm1 += v1 * v1;
        norm2 += v2 * v2;
    }
    
    // Cosine similarity
    float similarity = dot_product / (sqrtf(norm1) * sqrtf(norm2));
    results[vec_idx] = similarity;
}

/**
 * GPU kernel for parallel sorting using bitonic sort
 * Optimized for small to medium datasets
 */
__global__ void gpu_parallel_sort_kernel(
    float* keys, int* values, int num_elements, int ascending) {
    
    extern __shared__ float shared_keys[];
    int* shared_values = (int*)&shared_keys[blockDim.x];
    
    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    if (global_idx < num_elements) {
        shared_keys[tid] = keys[global_idx];
        shared_values[tid] = values[global_idx];
    } else {
        shared_keys[tid] = ascending ? FLT_MAX : -FLT_MAX;
        shared_values[tid] = -1;
    }
    
    __syncthreads();
    
    // Bitonic sort within block
    for (int k = 2; k <= blockDim.x; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;
            
            if (ixj > tid) {
                bool should_swap = ascending ? 
                    (shared_keys[tid] > shared_keys[ixj]) : 
                    (shared_keys[tid] < shared_keys[ixj]);
                
                if (((tid & k) == 0) == should_swap) {
                    // Swap
                    float temp_key = shared_keys[tid];
                    int temp_value = shared_values[tid];
                    shared_keys[tid] = shared_keys[ixj];
                    shared_values[tid] = shared_values[ixj];
                    shared_keys[ixj] = temp_key;
                    shared_values[ixj] = temp_value;
                }
            }
            
            __syncthreads();
        }
    }
    
    // Write back to global memory
    if (global_idx < num_elements) {
        keys[global_idx] = shared_keys[tid];
        values[global_idx] = shared_values[tid];
    }
}

/**
 * GPU kernel for text preprocessing
 * Converts text to lowercase and removes punctuation
 */
__global__ void gpu_text_preprocessing_kernel(
    char* input_text, char* output_text, int* text_lengths, 
    int num_texts, int convert_lowercase, int remove_punctuation) {
    
    int text_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (text_idx >= num_texts) return;
    
    int text_length = text_lengths[text_idx];
    int input_offset = text_idx * GPU_STRING_MAX_LENGTH;
    int output_offset = text_idx * GPU_STRING_MAX_LENGTH;
    int output_pos = 0;
    
    for (int i = 0; i < text_length && i < GPU_STRING_MAX_LENGTH - 1; i++) {
        char c = input_text[input_offset + i];
        
        // Convert to lowercase if requested
        if (convert_lowercase && c >= 'A' && c <= 'Z') {
            c = c - 'A' + 'a';
        }
        
        // Remove punctuation if requested
        if (remove_punctuation) {
            bool is_alphanumeric = (c >= 'a' && c <= 'z') || 
                                  (c >= 'A' && c <= 'Z') || 
                                  (c >= '0' && c <= '9') || 
                                  (c == ' ');
            if (!is_alphanumeric) {
                continue;
            }
        }
        
        output_text[output_offset + output_pos] = c;
        output_pos++;
    }
    
    output_text[output_offset + output_pos] = '\\0';
    text_lengths[text_idx] = output_pos;
}

// Device management functions

int cuda_initialize_device(int device_id) {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    if (device_id >= device_count) {
        fprintf(stderr, "Invalid device ID: %d (available: %d)\\n", device_id, device_count);
        return -1;
    }
    
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaGetDeviceProperties(&g_device_props, device_id));
    
    // Fill device info structure
    g_device_info.device_id = device_id;
    strncpy(g_device_info.name, g_device_props.name, sizeof(g_device_info.name) - 1);
    g_device_info.name[sizeof(g_device_info.name) - 1] = '\\0';
    
    g_device_info.total_memory = g_device_props.totalGlobalMem;
    g_device_info.major_version = g_device_props.major;
    g_device_info.minor_version = g_device_props.minor;
    g_device_info.multiprocessor_count = g_device_props.multiProcessorCount;
    g_device_info.max_threads_per_block = g_device_props.maxThreadsPerBlock;
    g_device_info.warp_size = g_device_props.warpSize;
    g_device_info.concurrent_kernels = g_device_props.concurrentKernels;
    g_device_info.memory_clock_rate = g_device_props.memoryClockRate / 1000.0f; // Convert to MHz
    g_device_info.gpu_clock_rate = g_device_props.clockRate / 1000.0f; // Convert to MHz
    
    // Get free memory
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    g_device_info.free_memory = free_mem;
    
    // Create performance events
    CUDA_CHECK(cudaEventCreate(&g_start_event));
    CUDA_CHECK(cudaEventCreate(&g_stop_event));
    
    // Initialize performance counters
    memset(&g_performance, 0, sizeof(gpu_performance_t));
    
    g_device_initialized = 1;
    
    printf("CUDA device %d initialized: %s\\n", device_id, g_device_info.name);
    printf("  Compute capability: %d.%d\\n", g_device_info.major_version, g_device_info.minor_version);
    printf("  Total memory: %.2f GB\\n", g_device_info.total_memory / (1024.0 * 1024.0 * 1024.0));
    printf("  Free memory: %.2f GB\\n", g_device_info.free_memory / (1024.0 * 1024.0 * 1024.0));
    printf("  Multiprocessors: %d\\n", g_device_info.multiprocessor_count);
    
    return 0;
}

void cuda_cleanup_device(void) {
    if (!g_device_initialized) return;
    
    // Destroy streams
    if (g_streams) {
        cuda_destroy_streams(g_streams, g_num_streams);
        g_streams = NULL;
        g_num_streams = 0;
    }
    
    // Free memory pool
    if (g_gpu_memory_pool) {
        cudaFree(g_gpu_memory_pool);
        g_gpu_memory_pool = NULL;
        g_memory_pool_size = 0;
        g_memory_pool_offset = 0;
    }
    
    // Destroy events
    cudaEventDestroy(g_start_event);
    cudaEventDestroy(g_stop_event);
    
    // Reset device
    cudaDeviceReset();
    
    g_device_initialized = 0;
    printf("CUDA device cleanup completed\\n");
}

gpu_device_info_t* cuda_get_device_info(int device_id) {
    if (!g_device_initialized || g_device_info.device_id != device_id) {
        if (cuda_initialize_device(device_id) != 0) {
            return NULL;
        }
    }
    return &g_device_info;
}

int cuda_get_optimal_block_size(int function_id) {
    // Use CUDA occupancy calculator for different kernel types
    int min_grid_size, block_size;
    
    switch (function_id) {
        case 0: // Tokenization kernel
            cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, 
                                              gpu_tokenize_documents_kernel, 0, 0);
            break;
        case 1: // BM25 scoring kernel
            cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, 
                                              gpu_bm25_scoring_kernel, 0, 0);
            break;
        case 2: // String matching kernel
            cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, 
                                              gpu_string_matching_kernel, 
                                              GPU_STRING_MAX_LENGTH * 32, 0);
            break;
        default:
            block_size = CUDA_BLOCK_SIZE;
    }
    
    // Ensure block size is multiple of warp size
    block_size = (block_size / CUDA_WARP_SIZE) * CUDA_WARP_SIZE;
    
    // Clamp to reasonable range
    if (block_size < CUDA_WARP_SIZE) block_size = CUDA_WARP_SIZE;
    if (block_size > CUDA_MAX_THREADS_PER_BLOCK) block_size = CUDA_MAX_THREADS_PER_BLOCK;
    
    return block_size;
}

void cuda_print_device_properties(void) {
    if (!g_device_initialized) {
        printf("CUDA device not initialized\\n");
        return;
    }
    
    printf("\\n=== CUDA Device Properties ===\\n");
    printf("Device: %s\\n", g_device_info.name);
    printf("Compute Capability: %d.%d\\n", g_device_info.major_version, g_device_info.minor_version);
    printf("Total Global Memory: %.2f GB\\n", g_device_info.total_memory / (1024.0 * 1024.0 * 1024.0));
    printf("Free Memory: %.2f GB\\n", g_device_info.free_memory / (1024.0 * 1024.0 * 1024.0));
    printf("Multiprocessors: %d\\n", g_device_info.multiprocessor_count);
    printf("Max Threads per Block: %d\\n", g_device_info.max_threads_per_block);
    printf("Warp Size: %d\\n", g_device_info.warp_size);
    printf("Concurrent Kernels: %s\\n", g_device_info.concurrent_kernels ? "Yes" : "No");
    printf("Memory Clock Rate: %.2f MHz\\n", g_device_info.memory_clock_rate);
    printf("GPU Clock Rate: %.2f MHz\\n", g_device_info.gpu_clock_rate);
    printf("=====================================\\n\\n");
}

// Memory management functions

gpu_memory_t* cuda_allocate_memory(size_t size, int use_pinned, int use_unified) {
    gpu_memory_t* mem = (gpu_memory_t*)malloc(sizeof(gpu_memory_t));
    if (!mem) return NULL;
    
    mem->size = size;
    mem->is_pinned = use_pinned;
    mem->is_unified = use_unified;
    mem->stream = 0;  // Default stream
    
    if (use_unified) {
        // Use unified memory
        CUDA_CHECK(cudaMallocManaged(&mem->d_ptr, size));
        mem->h_ptr = mem->d_ptr;  // Same pointer for unified memory
    } else {
        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&mem->d_ptr, size));
        
        // Allocate host memory
        if (use_pinned) {
            CUDA_CHECK(cudaMallocHost(&mem->h_ptr, size));
        } else {
            mem->h_ptr = malloc(size);
            if (!mem->h_ptr) {
                cudaFree(mem->d_ptr);
                free(mem);
                return NULL;
            }
        }
    }
    
    return mem;
}

void cuda_free_memory(gpu_memory_t* mem) {
    if (!mem) return;
    
    if (mem->is_unified) {
        cudaFree(mem->d_ptr);  // Unified memory only needs one free
    } else {
        cudaFree(mem->d_ptr);
        if (mem->is_pinned) {
            cudaFreeHost(mem->h_ptr);
        } else {
            free(mem->h_ptr);
        }
    }
    
    free(mem);
}

int cuda_transfer_to_device(gpu_memory_t* mem, void* host_data, size_t size) {
    if (!mem || size > mem->size) return -1;
    
    if (mem->is_unified) {
        // Just copy to the unified memory
        memcpy(mem->h_ptr, host_data, size);
        return 0;
    }
    
    // Copy to host memory first if needed
    if (host_data != mem->h_ptr) {
        memcpy(mem->h_ptr, host_data, size);
    }
    
    // Transfer to device
    CUDA_CHECK(cudaMemcpy(mem->d_ptr, mem->h_ptr, size, cudaMemcpyHostToDevice));
    
    return 0;
}

int cuda_transfer_from_device(gpu_memory_t* mem, void* host_data, size_t size) {
    if (!mem || size > mem->size) return -1;
    
    if (mem->is_unified) {
        // Just copy from the unified memory
        memcpy(host_data, mem->h_ptr, size);
        return 0;
    }
    
    // Transfer from device
    CUDA_CHECK(cudaMemcpy(mem->h_ptr, mem->d_ptr, size, cudaMemcpyDeviceToHost));
    
    // Copy to output buffer if needed
    if (host_data != mem->h_ptr) {
        memcpy(host_data, mem->h_ptr, size);
    }
    
    return 0;
}

void cuda_prefetch_unified_memory(void* ptr, size_t size, int device) {
    if (device < 0) {
        // Prefetch to CPU
        CUDA_CHECK(cudaMemPrefetchAsync(ptr, size, cudaCpuDeviceId, 0));
    } else {
        // Prefetch to specified GPU
        CUDA_CHECK(cudaMemPrefetchAsync(ptr, size, device, 0));
    }
}

// Stream management

cudaStream_t* cuda_create_streams(int num_streams) {
    cudaStream_t* streams = (cudaStream_t*)malloc(num_streams * sizeof(cudaStream_t));
    if (!streams) return NULL;
    
    for (int i = 0; i < num_streams; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }
    
    g_streams = streams;
    g_num_streams = num_streams;
    
    return streams;
}

void cuda_destroy_streams(cudaStream_t* streams, int num_streams) {
    if (!streams) return;
    
    for (int i = 0; i < num_streams; i++) {
        cudaStreamDestroy(streams[i]);
    }
    
    free(streams);
}

int cuda_async_memory_copy(gpu_memory_t* mem, void* data, size_t size, 
                           cudaMemcpyKind kind, cudaStream_t stream) {
    if (!mem || size > mem->size) return -1;
    
    CUDA_CHECK(cudaMemcpyAsync(mem->d_ptr, data, size, kind, stream));
    
    return 0;
}

void cuda_synchronize_streams(cudaStream_t* streams, int num_streams) {
    for (int i = 0; i < num_streams; i++) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }
}

// Performance monitoring

gpu_performance_t* gpu_get_performance_metrics(void) {
    if (!g_device_initialized) return NULL;
    
    // Update memory usage
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    g_performance.gpu_memory_used = total_mem - free_mem;
    
    return &g_performance;
}

void gpu_reset_performance_counters(void) {
    memset(&g_performance, 0, sizeof(gpu_performance_t));
}

float cuda_get_memory_bandwidth_utilization(void) {
    // This would require additional profiling APIs
    // For now, return a placeholder
    return 0.0f;
}

float cuda_get_compute_utilization(void) {
    // This would require NVML or profiling APIs
    // For now, return a placeholder
    return 0.0f;
}

void cuda_warmup_gpu(void) {
    // Simple kernel to warm up the GPU
    int* dummy;
    cudaMalloc(&dummy, sizeof(int));
    
    dim3 block(32);
    dim3 grid(1);
    
    // Launch a simple kernel
    // (Would need a simple warmup kernel here)
    
    cudaDeviceSynchronize();
    cudaFree(dummy);
}
