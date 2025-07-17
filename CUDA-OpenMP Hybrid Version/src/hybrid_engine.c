#include "hybrid_engine.h"
#include "index.h"
#include "ranking.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <unistd.h>

#ifdef USE_CUDA
#include "cuda_kernels.h"
#endif

// Global variables
hybrid_config_t g_hybrid_config;
hybrid_performance_t g_hybrid_performance;
int g_hybrid_initialized = 0;

// Thread safety locks
static omp_lock_t g_index_lock;
static omp_lock_t g_results_lock;
static omp_lock_t g_performance_lock;

// Performance timing
static struct timeval g_start_time, g_end_time;
static double g_cpu_start_time, g_gpu_start_time;

// Memory pools
static void* g_cpu_memory_pool = NULL;
static size_t g_cpu_pool_size = 0;
static size_t g_cpu_pool_offset = 0;

// Work distribution cache
static work_distribution_t* g_cached_distribution = NULL;
static int g_last_workload_size = 0;

/**
 * Initialize the hybrid engine with given configuration
 */
int hybrid_engine_init(hybrid_config_t* config) {
    if (g_hybrid_initialized) {
        printf("Hybrid engine already initialized\\n");
        return HYBRID_SUCCESS;
    }
    
    // Set default configuration if none provided
    if (!config) {
        hybrid_set_default_config(&g_hybrid_config);
    } else {
        g_hybrid_config = *config;
    }
    
    // Initialize OpenMP
    omp_set_num_threads(g_hybrid_config.omp_threads);
    omp_set_dynamic(0);  // Disable dynamic adjustment
    omp_set_nested(1);   // Enable nested parallelism
    
    // Initialize locks
    omp_init_lock(&g_index_lock);
    omp_init_lock(&g_results_lock);
    omp_init_lock(&g_performance_lock);
    
    printf("Initializing hybrid engine with %d OpenMP threads\\n", g_hybrid_config.omp_threads);
    
#ifdef USE_CUDA
    if (g_hybrid_config.use_gpu) {
        printf("Initializing CUDA GPU acceleration...\\n");
        
        if (cuda_initialize_device(0) != 0) {
            printf("WARNING: Failed to initialize CUDA, falling back to CPU-only mode\\n");
            g_hybrid_config.use_gpu = 0;
            g_hybrid_config.mode = PROCESSING_CPU_ONLY;
        } else {
            printf("CUDA initialization successful\\n");
            cuda_print_device_properties();
            
            // Create CUDA streams for async processing
            if (g_hybrid_config.enable_async_processing) {
                cuda_create_streams(g_hybrid_config.max_concurrent_streams);
            }
            
            // Warm up GPU
            cuda_warmup_gpu();
        }
    }
#else
    if (g_hybrid_config.use_gpu) {
        printf("WARNING: CUDA support not compiled, falling back to CPU-only mode\\n");
        g_hybrid_config.use_gpu = 0;
        g_hybrid_config.mode = PROCESSING_CPU_ONLY;
    }
#endif
    
    // Initialize performance counters
    memset(&g_hybrid_performance, 0, sizeof(hybrid_performance_t));
    
    // Allocate CPU memory pool if requested
    if (g_hybrid_config.max_cpu_memory > 0) {
        g_cpu_pool_size = g_hybrid_config.max_cpu_memory;
        g_cpu_memory_pool = malloc(g_cpu_pool_size);
        if (!g_cpu_memory_pool) {
            printf("WARNING: Failed to allocate CPU memory pool\\n");
            g_cpu_pool_size = 0;
        } else {
            printf("Allocated %.2f MB CPU memory pool\\n", 
                   g_cpu_pool_size / (1024.0 * 1024.0));
        }
    }
    
    // Auto-configure if requested
    if (g_hybrid_config.mode == PROCESSING_AUTO) {
        hybrid_engine_auto_configure();
    }
    
    g_hybrid_initialized = 1;
    
    printf("Hybrid engine initialization complete\\n");
    hybrid_engine_print_config();
    
    return HYBRID_SUCCESS;
}

/**
 * Cleanup hybrid engine resources
 */
void hybrid_engine_cleanup(void) {
    if (!g_hybrid_initialized) return;
    
    printf("Cleaning up hybrid engine...\\n");
    
    // Print final performance report
    hybrid_print_performance_report();
    
#ifdef USE_CUDA
    if (g_hybrid_config.use_gpu) {
        cuda_cleanup_device();
    }
#endif
    
    // Cleanup memory pools
    if (g_cpu_memory_pool) {
        free(g_cpu_memory_pool);
        g_cpu_memory_pool = NULL;
        g_cpu_pool_size = 0;
        g_cpu_pool_offset = 0;
    }
    
    // Cleanup cached work distribution
    if (g_cached_distribution) {
        if (g_cached_distribution->cpu_assignment) free(g_cached_distribution->cpu_assignment);
        if (g_cached_distribution->gpu_assignment) free(g_cached_distribution->gpu_assignment);
        free(g_cached_distribution);
        g_cached_distribution = NULL;
    }
    
    // Destroy locks
    omp_destroy_lock(&g_index_lock);
    omp_destroy_lock(&g_results_lock);
    omp_destroy_lock(&g_performance_lock);
    
    g_hybrid_initialized = 0;
    
    printf("Hybrid engine cleanup complete\\n");
}

/**
 * Auto-configure optimal settings based on hardware
 */
int hybrid_engine_auto_configure(void) {
    printf("Auto-configuring hybrid engine...\\n");
    
    // Detect CPU characteristics
    int num_cores = omp_get_num_procs();
    g_hybrid_config.omp_threads = num_cores;
    
    printf("Detected %d CPU cores\\n", num_cores);
    
#ifdef USE_CUDA
    if (g_hybrid_config.use_gpu) {
        gpu_device_info_t* gpu_info = cuda_get_device_info(0);
        if (gpu_info) {
            printf("GPU: %s with %.1f GB memory\\n", 
                   gpu_info->name, gpu_info->total_memory / (1024.0 * 1024.0 * 1024.0));
            
            // Auto-configure GPU parameters
            g_hybrid_config.cuda_block_size = cuda_get_optimal_block_size(0);
            
            // Set memory limits based on available GPU memory
            g_hybrid_config.max_gpu_memory = gpu_info->free_memory * 0.8;  // Use 80% of available
            
            // Configure processing mode based on GPU capability
            if (gpu_info->major_version >= 7) {  // Volta or newer
                g_hybrid_config.mode = PROCESSING_HYBRID;
                g_hybrid_config.cpu_gpu_ratio = 0.3f;  // Favor GPU
            } else if (gpu_info->major_version >= 6) {  // Pascal
                g_hybrid_config.mode = PROCESSING_HYBRID;
                g_hybrid_config.cpu_gpu_ratio = 0.5f;  // Balanced
            } else {  // Older GPUs
                g_hybrid_config.mode = PROCESSING_CPU_ONLY;
                printf("GPU compute capability too old, using CPU-only mode\\n");
            }
        }
    }
#endif
    
    // Configure load balancing
    g_hybrid_config.load_balance = LOAD_BALANCE_DYNAMIC;
    
    // Configure memory strategy
    if (g_hybrid_config.use_gpu) {
#ifdef USE_CUDA
        // Check if unified memory is supported
        gpu_device_info_t* gpu_info = cuda_get_device_info(0);
        if (gpu_info && gpu_info->major_version >= 6) {
            g_hybrid_config.memory_strategy = MEMORY_UNIFIED;
        } else {
            g_hybrid_config.memory_strategy = MEMORY_PINNED;
        }
#endif
    } else {
        g_hybrid_config.memory_strategy = MEMORY_BASIC;
    }
    
    // Configure batch size based on available memory
    size_t available_memory = hybrid_get_available_memory(-1);  // CPU memory
    g_hybrid_config.batch_size = (int)(available_memory / (1024 * 1024 * 10));  // 10MB per batch item
    if (g_hybrid_config.batch_size < 100) g_hybrid_config.batch_size = 100;
    if (g_hybrid_config.batch_size > 10000) g_hybrid_config.batch_size = 10000;
    
    printf("Auto-configuration complete\\n");
    
    return HYBRID_SUCCESS;
}

/**
 * Print current hybrid engine configuration
 */
void hybrid_engine_print_config(void) {
    printf("\\n=== Hybrid Engine Configuration ===\\n");
    printf("Processing Mode: ");
    switch (g_hybrid_config.mode) {
        case PROCESSING_CPU_ONLY: printf("CPU Only\\n"); break;
        case PROCESSING_GPU_ONLY: printf("GPU Only\\n"); break;
        case PROCESSING_HYBRID: printf("Hybrid CPU+GPU\\n"); break;
        case PROCESSING_AUTO: printf("Auto-select\\n"); break;
    }
    
    printf("GPU Acceleration: %s\\n", g_hybrid_config.use_gpu ? "Enabled" : "Disabled");
    printf("OpenMP Threads: %d\\n", g_hybrid_config.omp_threads);
    
    if (g_hybrid_config.use_gpu) {
        printf("CUDA Block Size: %d\\n", g_hybrid_config.cuda_block_size);
        printf("CPU/GPU Ratio: %.2f\\n", g_hybrid_config.cpu_gpu_ratio);
    }
    
    printf("Load Balancing: ");
    switch (g_hybrid_config.load_balance) {
        case LOAD_BALANCE_STATIC: printf("Static\\n"); break;
        case LOAD_BALANCE_DYNAMIC: printf("Dynamic\\n"); break;
        case LOAD_BALANCE_GUIDED: printf("Guided\\n"); break;
        case LOAD_BALANCE_AUTO: printf("Auto\\n"); break;
    }
    
    printf("Memory Strategy: ");
    switch (g_hybrid_config.memory_strategy) {
        case MEMORY_BASIC: printf("Basic\\n"); break;
        case MEMORY_PINNED: printf("Pinned\\n"); break;
        case MEMORY_UNIFIED: printf("Unified\\n"); break;
        case MEMORY_MANAGED: printf("Managed\\n"); break;
    }
    
    printf("Batch Size: %d\\n", g_hybrid_config.batch_size);
    printf("Async Processing: %s\\n", g_hybrid_config.enable_async_processing ? "Enabled" : "Disabled");
    printf("Auto-tuning: %s\\n", g_hybrid_config.enable_auto_tuning ? "Enabled" : "Disabled");
    printf("====================================\\n\\n");
}

/**
 * Process documents using hybrid CPU+GPU approach
 */
int hybrid_process_documents(hybrid_document_t* documents, int num_docs) {
    if (!g_hybrid_initialized) return HYBRID_ERROR_INIT;
    
    hybrid_start_timer("document_processing");
    
    // Analyze workload characteristics
    hybrid_optimization_hint_t* hints = hybrid_analyze_workload_characteristics(documents, num_docs);
    if (hints) {
        printf("Workload analysis: complexity=%d, memory_intensive=%s, compute_intensive=%s\\n",
               hints->complexity_level,
               hints->memory_intensive ? "yes" : "no",
               hints->compute_intensive ? "yes" : "no");
    }
    
    // Determine processing strategy
    work_distribution_t* distribution = hybrid_analyze_workload(documents, num_docs, 
                                                                hints ? hints->complexity_level : 5);
    
    if (g_hybrid_config.mode == PROCESSING_CPU_ONLY || !g_hybrid_config.use_gpu) {
        // CPU-only processing with OpenMP
        printf("Processing %d documents using CPU-only mode with %d threads\\n", 
               num_docs, g_hybrid_config.omp_threads);
        
        #pragma omp parallel for schedule(dynamic, g_hybrid_config.batch_size / g_hybrid_config.omp_threads)
        for (int i = 0; i < num_docs; i++) {
            hybrid_cpu_process_documents(&documents[i], i, i + 1);
            
            #pragma omp atomic
            g_hybrid_performance.documents_processed_cpu++;
        }
        
    } else if (g_hybrid_config.mode == PROCESSING_GPU_ONLY) {
#ifdef USE_CUDA
        // GPU-only processing
        printf("Processing %d documents using GPU-only mode\\n", num_docs);
        
        hybrid_gpu_process_documents(documents, num_docs);
        g_hybrid_performance.documents_processed_gpu = num_docs;
#else
        printf("GPU mode requested but CUDA not available, falling back to CPU\\n");
        return hybrid_process_documents(documents, num_docs);  // Recursive call with CPU fallback
#endif
        
    } else {
        // Hybrid processing
        printf("Processing %d documents using hybrid mode (CPU: %d, GPU: %d)\\n", 
               num_docs, distribution->cpu_work_units, distribution->gpu_work_units);
        
        // Split work between CPU and GPU
        int cpu_docs = distribution->cpu_work_units;
        int gpu_docs = distribution->gpu_work_units;
        
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                // CPU processing section
                if (cpu_docs > 0) {
                    printf("CPU processing %d documents with %d threads\\n", 
                           cpu_docs, g_hybrid_config.omp_threads / 2);
                    
                    #pragma omp parallel for num_threads(g_hybrid_config.omp_threads / 2) \\
                        schedule(dynamic)
                    for (int i = 0; i < cpu_docs; i++) {
                        hybrid_cpu_process_documents(&documents[i], i, i + 1);
                        
                        #pragma omp atomic
                        g_hybrid_performance.documents_processed_cpu++;
                    }
                }
            }
            
            #pragma omp section
            {
                // GPU processing section
#ifdef USE_CUDA
                if (gpu_docs > 0) {
                    printf("GPU processing %d documents\\n", gpu_docs);
                    
                    hybrid_gpu_process_documents(&documents[cpu_docs], gpu_docs);
                    g_hybrid_performance.documents_processed_gpu = gpu_docs;
                }
#endif
            }
        }
    }
    
    hybrid_stop_timer("document_processing");
    
    if (hints) free(hints);
    if (distribution) {
        if (distribution->cpu_assignment) free(distribution->cpu_assignment);
        if (distribution->gpu_assignment) free(distribution->gpu_assignment);
        free(distribution);
    }
    
    printf("Document processing complete: %d CPU, %d GPU\\n",
           g_hybrid_performance.documents_processed_cpu,
           g_hybrid_performance.documents_processed_gpu);
    
    return HYBRID_SUCCESS;
}

/**
 * Perform hybrid search across indexed documents
 */
int hybrid_search(hybrid_query_t* query, hybrid_result_t* results, int max_results) {
    if (!g_hybrid_initialized) return HYBRID_ERROR_INIT;
    
    hybrid_start_timer("search");
    
    printf("Executing hybrid search for query: '%s'\\n", query->query_text);
    
    // Parse query terms
    printf("Parsing query terms...\\n");
    // Query parsing would be implemented here
    
    if (g_hybrid_config.mode == PROCESSING_CPU_ONLY || !g_hybrid_config.use_gpu) {
        // CPU-only search
        hybrid_cpu_search_parallel(query, results, 0, get_doc_count());
        
    } else if (g_hybrid_config.mode == PROCESSING_GPU_ONLY) {
#ifdef USE_CUDA
        // GPU-only search
        hybrid_gpu_search_parallel(query, results, get_doc_count());
#endif
        
    } else {
        // Hybrid search - split workload
        int total_docs = get_doc_count();
        int cpu_docs = (int)(total_docs * g_hybrid_config.cpu_gpu_ratio);
        int gpu_docs = total_docs - cpu_docs;
        
        hybrid_result_t* cpu_results = malloc(max_results * sizeof(hybrid_result_t));
        hybrid_result_t* gpu_results = malloc(max_results * sizeof(hybrid_result_t));
        
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                // CPU search
                if (cpu_docs > 0) {
                    hybrid_cpu_search_parallel(query, cpu_results, 0, cpu_docs);
                }
            }
            
            #pragma omp section
            {
                // GPU search
#ifdef USE_CUDA
                if (gpu_docs > 0) {
                    hybrid_gpu_search_parallel(query, gpu_results, gpu_docs);
                }
#endif
            }
        }
        
        // Merge results
        int cpu_count = cpu_docs > 0 ? max_results : 0;  // Simplified for now
        int gpu_count = gpu_docs > 0 ? max_results : 0;  // Simplified for now
        
        merge_partial_results(cpu_results, cpu_count, gpu_results, gpu_count, 
                             results, max_results);
        
        free(cpu_results);
        free(gpu_results);
    }
    
    hybrid_stop_timer("search");
    
    printf("Search complete\\n");
    
    return max_results;  // Simplified return value
}

/**
 * Analyze workload to determine optimal distribution
 */
work_distribution_t* hybrid_analyze_workload(void* data, int data_size, int complexity) {
    work_distribution_t* dist = malloc(sizeof(work_distribution_t));
    if (!dist) return NULL;
    
    memset(dist, 0, sizeof(work_distribution_t));
    
    dist->total_work_units = data_size;
    
    if (!g_hybrid_config.use_gpu || g_hybrid_config.mode == PROCESSING_CPU_ONLY) {
        // CPU-only distribution
        dist->cpu_work_units = data_size;
        dist->gpu_work_units = 0;
        dist->estimated_cpu_time = data_size * 0.1f;  // Simplified estimation
        dist->estimated_gpu_time = 0.0f;
        
    } else if (g_hybrid_config.mode == PROCESSING_GPU_ONLY) {
        // GPU-only distribution
        dist->cpu_work_units = 0;
        dist->gpu_work_units = data_size;
        dist->estimated_cpu_time = 0.0f;
        dist->estimated_gpu_time = data_size * 0.05f;  // GPU assumed faster
        
    } else {
        // Hybrid distribution based on complexity and hardware
        float gpu_ratio = 1.0f - g_hybrid_config.cpu_gpu_ratio;
        
        // Adjust based on complexity - GPU better for high complexity
        if (complexity > 7) {
            gpu_ratio += 0.2f;
        } else if (complexity < 3) {
            gpu_ratio -= 0.2f;
        }
        
        // Clamp ratio
        if (gpu_ratio < 0.1f) gpu_ratio = 0.1f;
        if (gpu_ratio > 0.9f) gpu_ratio = 0.9f;
        
        dist->gpu_work_units = (int)(data_size * gpu_ratio);
        dist->cpu_work_units = data_size - dist->gpu_work_units;
        
        // Estimate times (simplified)
        dist->estimated_cpu_time = dist->cpu_work_units * 0.1f;
        dist->estimated_gpu_time = dist->gpu_work_units * 0.05f;
        
        printf("Workload distribution: CPU=%d (%.1f%%), GPU=%d (%.1f%%)\\n",
               dist->cpu_work_units, (float)dist->cpu_work_units / data_size * 100.0f,
               dist->gpu_work_units, (float)dist->gpu_work_units / data_size * 100.0f);
    }
    
    return dist;
}

/**
 * CPU-specific document processing using OpenMP
 */
void hybrid_cpu_process_documents(hybrid_document_t* documents, int start_idx, int end_idx) {
    int thread_id = omp_get_thread_num();
    
    for (int i = start_idx; i < end_idx; i++) {
        // Simulate document processing
        // In a real implementation, this would parse and tokenize the document
        documents[i].processed_on_gpu = 0;  // Mark as CPU processed
        documents[i].processing_time = 0.1f;  // Simplified timing
        
        // Simulate some work
        usleep(1000);  // 1ms delay to simulate processing
    }
}

/**
 * CPU-specific parallel search using OpenMP
 */
void hybrid_cpu_search_parallel(hybrid_query_t* query, hybrid_result_t* results, 
                                int start_doc, int end_doc) {
    int num_docs = end_doc - start_doc;
    
    #pragma omp parallel for schedule(dynamic, 32)
    for (int i = 0; i < num_docs; i++) {
        int doc_id = start_doc + i;
        
        // Simulate BM25 score calculation
        results[i].doc_id = doc_id;
        results[i].score = (float)rand() / RAND_MAX;  // Random score for demo
        results[i].cpu_time = 0.01f;
        results[i].gpu_time = 0.0f;
        results[i].processing_location = 0;  // CPU
    }
}

#ifdef USE_CUDA
/**
 * GPU-specific document processing using CUDA
 */
void hybrid_gpu_process_documents(hybrid_document_t* documents, int num_docs) {
    printf("GPU processing %d documents...\\n", num_docs);
    
    // Allocate GPU memory
    gpu_memory_t* gpu_mem = cuda_allocate_memory(num_docs * sizeof(hybrid_document_t), 
                                                 0, g_hybrid_config.memory_strategy == MEMORY_UNIFIED);
    if (!gpu_mem) {
        printf("Failed to allocate GPU memory, falling back to CPU\\n");
        hybrid_cpu_process_documents(documents, 0, num_docs);
        return;
    }
    
    // Transfer data to GPU
    cuda_transfer_to_device(gpu_mem, documents, num_docs * sizeof(hybrid_document_t));
    
    // Launch GPU kernels (simplified)
    int block_size = g_hybrid_config.cuda_block_size;
    int grid_size = (num_docs + block_size - 1) / block_size;
    
    // This would launch actual CUDA kernels for document processing
    // For now, just simulate processing time
    usleep(num_docs * 50);  // GPU assumed faster per document
    
    // Transfer results back
    cuda_transfer_from_device(gpu_mem, documents, num_docs * sizeof(hybrid_document_t));
    
    // Mark documents as GPU processed
    for (int i = 0; i < num_docs; i++) {
        documents[i].processed_on_gpu = 1;
        documents[i].processing_time = 0.05f;  // GPU assumed faster
    }
    
    cuda_free_memory(gpu_mem);
    
    printf("GPU document processing complete\\n");
}

/**
 * GPU-specific parallel search using CUDA
 */
void hybrid_gpu_search_parallel(hybrid_query_t* query, hybrid_result_t* results, int num_docs) {
    printf("GPU search across %d documents...\\n", num_docs);
    
    // Allocate GPU memory for results
    gpu_memory_t* results_mem = cuda_allocate_memory(num_docs * sizeof(hybrid_result_t), 
                                                     0, g_hybrid_config.memory_strategy == MEMORY_UNIFIED);
    if (!results_mem) {
        printf("Failed to allocate GPU memory for search, falling back to CPU\\n");
        hybrid_cpu_search_parallel(query, results, 0, num_docs);
        return;
    }
    
    // Launch GPU search kernels
    int block_size = g_hybrid_config.cuda_block_size;
    int grid_size = (num_docs + block_size - 1) / block_size;
    
    // Simulate GPU BM25 scoring
    for (int i = 0; i < num_docs; i++) {
        results[i].doc_id = i;
        results[i].score = (float)rand() / RAND_MAX;  // Random score for demo
        results[i].cpu_time = 0.0f;
        results[i].gpu_time = 0.005f;  // GPU assumed faster
        results[i].processing_location = 1;  // GPU
    }
    
    cuda_free_memory(results_mem);
    
    printf("GPU search complete\\n");
}
#endif

/**
 * Performance timing functions
 */
void hybrid_start_timer(const char* operation) {
    gettimeofday(&g_start_time, NULL);
    g_cpu_start_time = omp_get_wtime();
    
#ifdef USE_CUDA
    if (g_hybrid_config.use_gpu) {
        cudaEventRecord(g_start_event, 0);
    }
#endif
}

void hybrid_stop_timer(const char* operation) {
    gettimeofday(&g_end_time, NULL);
    double cpu_time = omp_get_wtime() - g_cpu_start_time;
    
    double wall_time = (g_end_time.tv_sec - g_start_time.tv_sec) + 
                       (g_end_time.tv_usec - g_start_time.tv_usec) / 1000000.0;
    
    omp_set_lock(&g_performance_lock);
    g_hybrid_performance.total_time += wall_time;
    g_hybrid_performance.cpu_time += cpu_time;
    omp_unset_lock(&g_performance_lock);
    
#ifdef USE_CUDA
    if (g_hybrid_config.use_gpu) {
        cudaEventRecord(g_stop_event, 0);
        cudaEventSynchronize(g_stop_event);
        
        float gpu_time;
        cudaEventElapsedTime(&gpu_time, g_start_event, g_stop_event);
        
        omp_set_lock(&g_performance_lock);
        g_hybrid_performance.gpu_time += gpu_time / 1000.0;  // Convert to seconds
        omp_unset_lock(&g_performance_lock);
    }
#endif
    
    printf("Operation '%s' completed in %.3f seconds (CPU: %.3f)\\n", 
           operation, wall_time, cpu_time);
}

/**
 * Get current performance metrics
 */
hybrid_performance_t* hybrid_get_performance_metrics(void) {
    return &g_hybrid_performance;
}

/**
 * Print comprehensive performance report
 */
void hybrid_print_performance_report(void) {
    printf("\\n=== Hybrid Engine Performance Report ===\\n");
    printf("Total Execution Time: %.3f seconds\\n", g_hybrid_performance.total_time);
    printf("CPU Processing Time: %.3f seconds\\n", g_hybrid_performance.cpu_time);
    printf("GPU Processing Time: %.3f seconds\\n", g_hybrid_performance.gpu_time);
    
    if (g_hybrid_performance.gpu_time > 0) {
        float speedup = g_hybrid_performance.cpu_time / g_hybrid_performance.gpu_time;
        printf("GPU Acceleration Factor: %.2fx\\n", speedup);
    }
    
    printf("Documents Processed - CPU: %d, GPU: %d\\n",
           g_hybrid_performance.documents_processed_cpu,
           g_hybrid_performance.documents_processed_gpu);
    
    printf("Queries Processed - CPU: %d, GPU: %d\\n",
           g_hybrid_performance.queries_processed_cpu,
           g_hybrid_performance.queries_processed_gpu);
    
    if (g_hybrid_performance.total_time > 0) {
        float cpu_utilization = g_hybrid_performance.cpu_time / g_hybrid_performance.total_time * 100.0f;
        printf("CPU Utilization: %.1f%%\\n", cpu_utilization);
    }
    
#ifdef USE_CUDA
    if (g_hybrid_config.use_gpu) {
        gpu_performance_t* gpu_perf = gpu_get_performance_metrics();
        if (gpu_perf) {
            printf("GPU Memory Used: %.2f MB\\n", gpu_perf->gpu_memory_used / (1024.0 * 1024.0));
        }
    }
#endif
    
    printf("========================================\\n\\n");
}

/**
 * Set default configuration
 */
void hybrid_set_default_config(hybrid_config_t* config) {
    memset(config, 0, sizeof(hybrid_config_t));
    
    config->mode = PROCESSING_AUTO;
    config->use_gpu = 1;  // Try to use GPU by default
    config->omp_threads = omp_get_num_procs();
    config->cuda_block_size = CUDA_BLOCK_SIZE;
    config->cuda_grid_size = 0;  // Auto-calculate
    
    config->load_balance = LOAD_BALANCE_DYNAMIC;
    config->memory_strategy = MEMORY_UNIFIED;
    config->cpu_gpu_ratio = 0.5f;  // Balanced by default
    config->batch_size = 1000;
    config->prefetch_enabled = 1;
    
    config->max_gpu_memory = 0;  // Auto-detect
    config->max_cpu_memory = 1024 * 1024 * 1024;  // 1GB default
    config->max_concurrent_streams = 4;
    config->enable_peer_access = 0;
    
    config->enable_memory_coalescing = 1;
    config->enable_async_processing = 1;
    config->enable_kernel_fusion = 1;
    config->enable_auto_tuning = 1;
}

/**
 * Analyze workload characteristics for optimization
 */
hybrid_optimization_hint_t* hybrid_analyze_workload_characteristics(void* data, int size) {
    hybrid_optimization_hint_t* hints = malloc(sizeof(hybrid_optimization_hint_t));
    if (!hints) return NULL;
    
    memset(hints, 0, sizeof(hybrid_optimization_hint_t));
    
    hints->data_size = size;
    hints->complexity_level = 5;  // Medium complexity by default
    hints->memory_intensive = size > 10000 ? 1 : 0;
    hints->compute_intensive = 1;  // Search operations are generally compute-intensive
    hints->io_intensive = 0;
    
    // Determine optimal ratios based on characteristics
    if (hints->compute_intensive && size > 1000) {
        hints->recommended_gpu_ratio = 0.7f;
        hints->recommended_cpu_ratio = 0.3f;
        hints->optimization_notes = "Large compute-intensive workload favors GPU";
    } else if (hints->memory_intensive) {
        hints->recommended_gpu_ratio = 0.5f;
        hints->recommended_cpu_ratio = 0.5f;
        hints->optimization_notes = "Memory-intensive workload benefits from balanced approach";
    } else {
        hints->recommended_gpu_ratio = 0.3f;
        hints->recommended_cpu_ratio = 0.7f;
        hints->optimization_notes = "Small workload favors CPU to avoid GPU overhead";
    }
    
    return hints;
}

/**
 * Utility functions
 */
double hybrid_get_wall_time(void) {
    struct timeval time;
    gettimeofday(&time, NULL);
    return (double)time.tv_sec + (double)time.tv_usec * 0.000001;
}

size_t hybrid_get_available_memory(int device) {
    if (device < 0) {
        // CPU memory - simplified estimation
        return 1024 * 1024 * 1024;  // 1GB placeholder
    } else {
#ifdef USE_CUDA
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        return free_mem;
#else
        return 0;
#endif
    }
}

void hybrid_print_system_info(void) {
    printf("\\n=== System Information ===\\n");
    printf("CPU Cores: %d\\n", omp_get_num_procs());
    printf("OpenMP Threads: %d\\n", omp_get_max_threads());
    
#ifdef USE_CUDA
    int device_count;
    if (cudaGetDeviceCount(&device_count) == cudaSuccess) {
        printf("CUDA Devices: %d\\n", device_count);
        if (device_count > 0) {
            cuda_print_device_properties();
        }
    } else {
        printf("CUDA: Not available\\n");
    }
#else
    printf("CUDA: Not compiled\\n");
#endif
    
    printf("==========================\\n\\n");
}

// Thread safety functions
void hybrid_lock_index(void) { omp_set_lock(&g_index_lock); }
void hybrid_unlock_index(void) { omp_unset_lock(&g_index_lock); }
void hybrid_lock_results(void) { omp_set_lock(&g_results_lock); }
void hybrid_unlock_results(void) { omp_unset_lock(&g_results_lock); }

/**
 * Error handling
 */
const char* hybrid_get_error_string(hybrid_error_t error) {
    switch (error) {
        case HYBRID_SUCCESS: return "Success";
        case HYBRID_ERROR_INIT: return "Initialization error";
        case HYBRID_ERROR_MEMORY: return "Memory allocation error";
        case HYBRID_ERROR_CUDA: return "CUDA error";
        case HYBRID_ERROR_OMP: return "OpenMP error";
        case HYBRID_ERROR_IO: return "I/O error";
        case HYBRID_ERROR_CONFIG: return "Configuration error";
        default: return "Unknown error";
    }
}
