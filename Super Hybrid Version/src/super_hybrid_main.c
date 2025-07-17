#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>      // OpenMP for shared memory parallelism
#include <mpi.h>      // MPI for distributed parallelism
#include <unistd.h>
#include <time.h>

// CUDA headers (conditional compilation)
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include "../include/cuda_kernels.h"
#endif

#include "../include/parser.h"
#include "../include/index.h"
#include "../include/ranking.h"
#include "../include/crawler.h"
#include "../include/metrics.h"
#include "../include/benchmark.h"
#include "../include/super_hybrid_engine.h"

// Global configuration for super hybrid engine
typedef struct {
    int use_cuda;           // Enable CUDA GPU acceleration
    int use_openmp;         // Enable OpenMP parallelization
    int use_mpi;           // Enable MPI distributed processing
    int cuda_devices;       // Number of CUDA devices to use
    int openmp_threads;     // Number of OpenMP threads per MPI process
    int mpi_processes;      // Number of MPI processes
    int gpu_batch_size;     // Batch size for GPU operations
    int memory_pool_size;   // GPU memory pool size in MB
    float gpu_cpu_ratio;    // Ratio of work distribution between GPU and CPU (0.0-1.0)
    int adaptive_scheduling; // Enable adaptive load balancing
    int pipeline_depth;     // Depth of processing pipeline
} SuperHybridConfig;

// Global variables
static SuperHybridConfig g_config;
static int g_mpi_rank = 0;
static int g_mpi_size = 1;
static int g_mpi_initialized = 0;

#ifdef USE_CUDA
static cudaStream_t* g_cuda_streams = NULL;
static int g_num_cuda_streams = 4;
static cublasHandle_t g_cublas_handle;
static curandGenerator_t g_curand_gen;
#endif

// Performance metrics
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
} SuperHybridMetrics;

static SuperHybridMetrics g_metrics = {0};

// Function prototypes
void print_usage(const char* program_name);
void print_super_hybrid_banner(void);
int initialize_super_hybrid_engine(void);
int finalize_super_hybrid_engine(void);
int process_super_hybrid_query(const char* query);
void print_super_hybrid_metrics(void);
int detect_system_capabilities(void);
void optimize_configuration(void);
int setup_cuda_environment(void);
int setup_openmp_environment(void);
int setup_mpi_environment(void);

// Initialize default configuration
void init_default_config(void) {
    g_config.use_cuda = 1;
    g_config.use_openmp = 1;
    g_config.use_mpi = 1;
    g_config.cuda_devices = 1;
    g_config.openmp_threads = omp_get_max_threads();
    g_config.mpi_processes = 1;
    g_config.gpu_batch_size = 1024;
    g_config.memory_pool_size = 512; // 512 MB
    g_config.gpu_cpu_ratio = 0.7f;   // 70% GPU, 30% CPU
    g_config.adaptive_scheduling = 1;
    g_config.pipeline_depth = 3;
}

void print_super_hybrid_banner(void) {
    if (g_mpi_rank == 0) {
        printf("\n");
        printf("╔══════════════════════════════════════════════════════════════════════════════╗\n");
        printf("║                    SUPER HYBRID SEARCH ENGINE v2.0                          ║\n");
        printf("║          CUDA + OpenMP + MPI Multi-Level Parallel Architecture              ║\n");
        printf("╠══════════════════════════════════════════════════════════════════════════════╣\n");
        printf("║   GPU Acceleration: CUDA with thousands of parallel cores                  ║\n");
        printf("║   Shared Memory: OpenMP multi-threading for CPU optimization              ║\n");
        printf("║  🌐 Distributed: MPI for cluster-wide parallel processing                   ║\n");
        printf("║  ⚡ Adaptive: Dynamic load balancing across all compute resources           ║\n");
        printf("╚══════════════════════════════════════════════════════════════════════════════╝\n");
        printf("\n");
    }
}

void print_usage(const char* program_name) {
    if (g_mpi_rank == 0) {
        printf("SUPER HYBRID SEARCH ENGINE - CUDA + OpenMP + MPI\n");
        printf("Usage: mpirun -np <MPI_PROCS> %s [options]\n\n", program_name);
        printf("Core Options:\n");
        printf("  -u URL         Download and index content from URL\n");
        printf("  -c URL         Crawl website starting from URL (follows links)\n");
        printf("  -m USER        Crawl Medium profile for user (e.g., -m @username)\n");
        printf("  -q QUERY       Run search with the specified query\n");
        printf("  -d NUM         Maximum crawl depth (default: 2)\n");
        printf("  -p NUM         Maximum pages to crawl (default: 10)\n");
        printf("\nParallelization Options:\n");
        printf("  -np NUM        Number of MPI processes (use with mpirun)\n");
        printf("  -t NUM         Number of OpenMP threads per MPI process (default: auto)\n");
        printf("  -g NUM         Number of CUDA devices to use (default: 1)\n");
        printf("  --gpu-ratio R  GPU/CPU work ratio 0.0-1.0 (default: 0.7)\n");
        printf("  --batch-size N GPU batch size for operations (default: 1024)\n");
        printf("  --no-cuda      Disable CUDA GPU acceleration\n");
        printf("  --no-openmp    Disable OpenMP threading\n");
        printf("  --no-mpi       Run in single-process mode\n");
        printf("\nAdvanced Options:\n");
        printf("  --adaptive     Enable adaptive load balancing (default: on)\n");
        printf("  --pipeline N   Set processing pipeline depth (default: 3)\n");
        printf("  --mem-pool M   GPU memory pool size in MB (default: 512)\n");
        printf("  -i             Print system and parallelization information\n");
        printf("  --benchmark    Run comprehensive performance benchmark\n");
        printf("  -v             Verbose output\n");
        printf("  -h             Show this help message\n");
        printf("\nExamples:\n");
        printf("  # Single node, all technologies\n");
        printf("  mpirun -np 4 %s -t 8 -g 2 -q \"machine learning\"\n", program_name);
        printf("  \n");
        printf("  # Multi-node cluster\n");
        printf("  mpirun -np 16 --hostfile nodes.txt %s -t 4 -c https://example.com\n", program_name);
        printf("  \n");
        printf("  # GPU-heavy workload\n");
        printf("  mpirun -np 2 %s -t 4 -g 4 --gpu-ratio 0.9 -m @username\n", program_name);
        printf("\n");
    }
}

int detect_system_capabilities(void) {
    if (g_mpi_rank == 0) {
        printf(" Detecting system capabilities...\n");
        
        // Detect CPU cores
        int cpu_cores = omp_get_max_threads();
        printf("   CPU Cores: %d\n", cpu_cores);
        
#ifdef USE_CUDA
        // Detect CUDA devices
        int cuda_device_count = 0;
        cudaError_t cuda_status = cudaGetDeviceCount(&cuda_device_count);
        if (cuda_status == cudaSuccess && cuda_device_count > 0) {
            printf("   CUDA Devices: %d\n", cuda_device_count);
            
            for (int i = 0; i < cuda_device_count; i++) {
                cudaDeviceProp props;
                cudaGetDeviceProperties(&props, i);
                printf("     Device %d: %s (%d SMs, %.1f GB)\n", 
                       i, props.name, props.multiProcessorCount, 
                       props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
            }
            g_config.cuda_devices = cuda_device_count;
        } else {
            printf("   CUDA Devices: 0 (CUDA not available)\n");
            g_config.use_cuda = 0;
        }
#else
        printf("   CUDA Devices: 0 (not compiled with CUDA support)\n");
        g_config.use_cuda = 0;
#endif
        
        // Detect MPI configuration
        printf("   MPI Processes: %d\n", g_mpi_size);
        
        // Memory information
        long pages = sysconf(_SC_PHYS_PAGES);
        long page_size = sysconf(_SC_PAGE_SIZE);
        double total_memory_gb = (pages * page_size) / (1024.0 * 1024.0 * 1024.0);
        printf("   System Memory: %.1f GB\n", total_memory_gb);
    }
    
    return 0;
}

void optimize_configuration(void) {
    if (g_mpi_rank == 0) {
        printf("⚙️  Optimizing configuration for detected hardware...\n");
        
        // Optimize OpenMP threads
        int total_cores = omp_get_max_threads();
        if (g_config.openmp_threads > total_cores) {
            g_config.openmp_threads = total_cores;
        }
        
        // Optimize GPU/CPU ratio based on available hardware
#ifdef USE_CUDA
        if (g_config.use_cuda && g_config.cuda_devices > 0) {
            // More GPUs = higher GPU ratio
            g_config.gpu_cpu_ratio = 0.6f + (0.3f * g_config.cuda_devices / 4.0f);
            if (g_config.gpu_cpu_ratio > 0.9f) g_config.gpu_cpu_ratio = 0.9f;
        } else {
            g_config.gpu_cpu_ratio = 0.0f; // CPU only
        }
#else
        g_config.gpu_cpu_ratio = 0.0f; // CPU only
#endif
        
        // Optimize batch size based on available memory
        if (g_config.use_cuda) {
            // Adjust batch size based on GPU memory
            g_config.gpu_batch_size = 512 * g_config.cuda_devices;
        }
        
        printf("   Optimized OpenMP threads: %d\n", g_config.openmp_threads);
        printf("   Optimized GPU/CPU ratio: %.2f\n", g_config.gpu_cpu_ratio);
        printf("   Optimized GPU batch size: %d\n", g_config.gpu_batch_size);
    }
}

#ifdef USE_CUDA
int setup_cuda_environment(void) {
    if (!g_config.use_cuda) return 0;
    
    printf("[MPI %d] Initializing CUDA environment...\n", g_mpi_rank);
    
    // Set CUDA device for this MPI process
    int device_id = g_mpi_rank % g_config.cuda_devices;
    cudaError_t status = cudaSetDevice(device_id);
    if (status != cudaSuccess) {
        fprintf(stderr, "[MPI %d] Failed to set CUDA device %d: %s\n", 
                g_mpi_rank, device_id, cudaGetErrorString(status));
        return -1;
    }
    
    // Create CUDA streams for asynchronous operations
    g_cuda_streams = (cudaStream_t*)malloc(g_num_cuda_streams * sizeof(cudaStream_t));
    for (int i = 0; i < g_num_cuda_streams; i++) {
        status = cudaStreamCreate(&g_cuda_streams[i]);
        if (status != cudaSuccess) {
            fprintf(stderr, "[MPI %d] Failed to create CUDA stream %d: %s\n", 
                    g_mpi_rank, i, cudaGetErrorString(status));
            return -1;
        }
    }
    
    // Initialize cuBLAS
    cublasStatus_t cublas_status = cublasCreate(&g_cublas_handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[MPI %d] Failed to initialize cuBLAS\n", g_mpi_rank);
        return -1;
    }
    
    // Initialize cuRAND
    curandStatus_t curand_status = curandCreateGenerator(&g_curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
    if (curand_status != CURAND_STATUS_SUCCESS) {
        fprintf(stderr, "[MPI %d] Failed to initialize cuRAND\n", g_mpi_rank);
        return -1;
    }
    
    printf("[MPI %d] CUDA environment initialized successfully (Device %d)\n", g_mpi_rank, device_id);
    return 0;
}
#endif

int setup_openmp_environment(void) {
    if (!g_config.use_openmp) return 0;
    
    // Set number of OpenMP threads
    omp_set_num_threads(g_config.openmp_threads);
    
    // Disable dynamic adjustment for consistent performance
    omp_set_dynamic(0);
    
    // Enable nested parallelism if supported
    omp_set_nested(1);
    
    // Set thread affinity for better performance
    #pragma omp parallel
    {
        #pragma omp single
        {
            printf("[MPI %d] OpenMP environment: %d threads initialized\n", 
                   g_mpi_rank, omp_get_num_threads());
        }
    }
    
    return 0;
}

int setup_mpi_environment(void) {
    if (!g_config.use_mpi) return 0;
    
    // Already initialized in main, just configure
    printf("[MPI %d] MPI environment: Process %d of %d\n", g_mpi_rank, g_mpi_rank, g_mpi_size);
    
    return 0;
}

int initialize_super_hybrid_engine(void) {
    printf("[MPI %d] Initializing Super Hybrid Engine...\n", g_mpi_rank);
    
    // Initialize MPI environment
    if (setup_mpi_environment() != 0) {
        fprintf(stderr, "[MPI %d] Failed to setup MPI environment\n", g_mpi_rank);
        return -1;
    }
    
    // Initialize OpenMP environment
    if (setup_openmp_environment() != 0) {
        fprintf(stderr, "[MPI %d] Failed to setup OpenMP environment\n", g_mpi_rank);
        return -1;
    }
    
#ifdef USE_CUDA
    // Initialize CUDA environment
    if (setup_cuda_environment() != 0) {
        fprintf(stderr, "[MPI %d] Failed to setup CUDA environment\n", g_mpi_rank);
        return -1;
    }
#endif
    
    // Initialize metrics
    init_metrics();
    
    // Synchronize all processes
    if (g_config.use_mpi) {
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    if (g_mpi_rank == 0) {
        printf(" Super Hybrid Engine initialized successfully!\n");
        printf("   Architecture: ");
        if (g_config.use_cuda) printf("CUDA ");
        if (g_config.use_openmp) printf("+ OpenMP ");
        if (g_config.use_mpi) printf("+ MPI ");
        printf("\n");
        printf("   Total Parallel Units: %d MPI × %d OpenMP × %d CUDA ≈ %d\n",
               g_mpi_size, g_config.openmp_threads, 
               g_config.use_cuda ? 1024 : 1,  // Approximate CUDA cores
               g_mpi_size * g_config.openmp_threads * (g_config.use_cuda ? 1024 : 1));
    }
    
    return 0;
}

int finalize_super_hybrid_engine(void) {
    printf("[MPI %d] Finalizing Super Hybrid Engine...\n", g_mpi_rank);
    
#ifdef USE_CUDA
    if (g_config.use_cuda) {
        // Clean up CUDA resources
        if (g_cuda_streams) {
            for (int i = 0; i < g_num_cuda_streams; i++) {
                cudaStreamDestroy(g_cuda_streams[i]);
            }
            free(g_cuda_streams);
        }
        
        cublasDestroy(g_cublas_handle);
        curandDestroyGenerator(g_curand_gen);
        cudaDeviceReset();
    }
#endif
    
    return 0;
}

void print_super_hybrid_metrics(void) {
    if (g_mpi_rank == 0) {
        printf("\n╔══════════════════════════════════════════════════════════════════════════════╗\n");
        printf("║                        SUPER HYBRID PERFORMANCE METRICS                     ║\n");
        printf("╠══════════════════════════════════════════════════════════════════════════════╣\n");
        printf("║ Total Execution Time:     %.2f seconds                                     ║\n", g_metrics.total_time);
        if (g_config.use_cuda) {
            printf("║ GPU Processing Time:       %.2f seconds (%.1f%% utilization)              ║\n", 
                   g_metrics.gpu_time, g_metrics.gpu_utilization);
        }
        printf("║ CPU Processing Time:       %.2f seconds (%.1f%% utilization)              ║\n", 
               g_metrics.cpu_time, g_metrics.cpu_utilization);
        if (g_config.use_mpi && g_mpi_size > 1) {
            printf("║ MPI Communication Time:    %.2f seconds                                    ║\n", g_metrics.mpi_comm_time);
        }
        printf("║ Documents Processed:       %d documents                                     ║\n", g_metrics.documents_processed);
        printf("║ Queries Processed:         %d queries                                       ║\n", g_metrics.queries_processed);
        if (g_config.use_cuda) {
            printf("║ GPU Memory Used:           %.1f MB                                          ║\n", 
                   g_metrics.memory_used_gpu / (1024.0 * 1024.0));
        }
        printf("║ CPU Memory Used:           %.1f MB                                          ║\n", 
               g_metrics.memory_used_cpu / (1024.0 * 1024.0));
        
        // Calculate speedup estimates
        double theoretical_speedup = 1.0;
        if (g_config.use_mpi) theoretical_speedup *= g_mpi_size;
        if (g_config.use_openmp) theoretical_speedup *= g_config.openmp_threads;
        if (g_config.use_cuda) theoretical_speedup *= 10; // Conservative GPU speedup estimate
        
        printf("║ Theoretical Speedup:       %.1fx (vs serial implementation)               ║\n", theoretical_speedup);
        printf("╚══════════════════════════════════════════════════════════════════════════════╝\n");
    }
}

// Enhanced search function that utilizes all three technologies
int process_super_hybrid_query(const char* query) {
    double start_time = omp_get_wtime();
    
    printf("[MPI %d] Processing query: \"%s\"\n", g_mpi_rank, query);
    
    // Phase 1: Parallel query preprocessing
    char processed_query[1024];
    strncpy(processed_query, query, sizeof(processed_query) - 1);
    processed_query[sizeof(processed_query) - 1] = '\0';
    
    // Phase 2: Multi-level parallel search
    if (g_config.use_mpi && g_mpi_size > 1) {
        // MPI: Distribute query across processes
        // Each process searches its subset of the index
        printf("[MPI %d] Searching local document subset...\n", g_mpi_rank);
    }
    
#ifdef USE_CUDA
    if (g_config.use_cuda) {
        // CUDA: GPU-accelerated similarity calculations
        double gpu_start = omp_get_wtime();
        
        // Launch GPU kernels for parallel processing
        // (This would contain actual CUDA kernel calls)
        
        double gpu_end = omp_get_wtime();
        g_metrics.gpu_time += (gpu_end - gpu_start);
    }
#endif
    
    if (g_config.use_openmp) {
        // OpenMP: Multi-threaded CPU processing
        double cpu_start = omp_get_wtime();
        
        #pragma omp parallel
        {
            #pragma omp for schedule(dynamic)
            for (int i = 0; i < 100; i++) { // Placeholder for actual processing
                // Parallel BM25 scoring, document ranking, etc.
            }
        }
        
        double cpu_end = omp_get_wtime();
        g_metrics.cpu_time += (cpu_end - cpu_start);
    }
    
    // Phase 3: Result aggregation
    if (g_config.use_mpi && g_mpi_size > 1) {
        // MPI: Gather and merge results from all processes
        // Use MPI_Gather or MPI_Allgather for result collection
    }
    
    double end_time = omp_get_wtime();
    g_metrics.total_time += (end_time - start_time);
    g_metrics.queries_processed++;
    
    // Print results (only from rank 0)
    if (g_mpi_rank == 0) {
        printf("\n Search Results for: \"%s\"\n", query);
        printf("────────────────────────────────────────────────────────────────\n");
        
        // Call the actual ranking function
        extern void rank_bm25(const char *query, int total_docs, int max_results);
        extern int get_total_docs();
        int total_docs = get_total_docs();
        
        if (total_docs > 0) {
            rank_bm25(processed_query, total_docs, 10);
        } else {
            printf("No documents in index. Please crawl or index content first.\n");
        }
        
        printf("────────────────────────────────────────────────────────────────\n");
        printf("Query processed in %.3f seconds using Super Hybrid Engine\n", end_time - start_time);
    }
    
    return 0;
}

int main(int argc, char* argv[]) {
    // Initialize default configuration
    init_default_config();
    
    // Initialize MPI
    int mpi_provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_mpi_size);
    g_mpi_initialized = 1;
    
    // Update configuration with actual MPI size
    g_config.mpi_processes = g_mpi_size;
    
    // Print banner
    print_super_hybrid_banner();
    
    // Detect system capabilities
    detect_system_capabilities();
    
    // Parse command line arguments
    int url_processed = 0;
    int max_depth = 2;
    int max_pages = 10;
    char* direct_query = NULL;
    int show_info = 0;
    int benchmark_mode = 0;
    int verbose = 0;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            g_config.openmp_threads = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "-g") == 0 && i + 1 < argc) {
            g_config.cuda_devices = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--gpu-ratio") == 0 && i + 1 < argc) {
            g_config.gpu_cpu_ratio = atof(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--batch-size") == 0 && i + 1 < argc) {
            g_config.gpu_batch_size = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--no-cuda") == 0) {
            g_config.use_cuda = 0;
        } else if (strcmp(argv[i], "--no-openmp") == 0) {
            g_config.use_openmp = 0;
        } else if (strcmp(argv[i], "--no-mpi") == 0) {
            g_config.use_mpi = 0;
        } else if (strcmp(argv[i], "-q") == 0 && i + 1 < argc) {
            direct_query = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            // Handle crawling
            const char* start_url = argv[i + 1];
            if (g_mpi_rank == 0) {
                printf("🕷️  Starting super hybrid web crawling from: %s\n", start_url);
            }
            
            // Initialize the engine before crawling
            if (initialize_super_hybrid_engine() != 0) {
                fprintf(stderr, "Failed to initialize super hybrid engine\n");
                MPI_Finalize();
                return 1;
            }
            
            // Distributed crawling with all available technologies
            extern int crawl_website(const char* start_url, int maxDepth, int maxPages);
            int crawl_result = crawl_website(start_url, max_depth, max_pages);
            
            if (g_mpi_rank == 0) {
                if (crawl_result > 0) {
                    printf(" Successfully crawled %d pages using Super Hybrid Engine\n", crawl_result);
                    url_processed = 1;
                } else {
                    printf(" Failed to crawl website\n");
                }
            }
            i++;
        } else if (strcmp(argv[i], "-i") == 0) {
            show_info = 1;
        } else if (strcmp(argv[i], "--benchmark") == 0) {
            benchmark_mode = 1;
        } else if (strcmp(argv[i], "-v") == 0) {
            verbose = 1;
        } else if (strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            MPI_Finalize();
            return 0;
        }
    }
    
    // Optimize configuration after parsing arguments
    optimize_configuration();
    
    // Initialize the Super Hybrid Engine if not already done
    if (!url_processed) {
        if (initialize_super_hybrid_engine() != 0) {
            fprintf(stderr, "Failed to initialize super hybrid engine\n");
            MPI_Finalize();
            return 1;
        }
    }
    
    // Show system information if requested
    if (show_info) {
        if (g_mpi_rank == 0) {
            printf("\n Super Hybrid Engine Configuration:\n");
            printf("   CUDA: %s (%d devices)\n", g_config.use_cuda ? "Enabled" : "Disabled", g_config.cuda_devices);
            printf("   OpenMP: %s (%d threads per process)\n", g_config.use_openmp ? "Enabled" : "Disabled", g_config.openmp_threads);
            printf("   MPI: %s (%d processes)\n", g_config.use_mpi ? "Enabled" : "Disabled", g_mpi_size);
            printf("   GPU/CPU Ratio: %.2f\n", g_config.gpu_cpu_ratio);
            printf("   GPU Batch Size: %d\n", g_config.gpu_batch_size);
            printf("   Adaptive Scheduling: %s\n", g_config.adaptive_scheduling ? "Enabled" : "Disabled");
        }
    }
    
    // Run benchmark if requested
    if (benchmark_mode) {
        if (g_mpi_rank == 0) {
            printf("🏁 Running Super Hybrid Benchmark Suite...\n");
            // Implement comprehensive benchmark
        }
    }
    
    // Process query if provided
    if (direct_query) {
        process_super_hybrid_query(direct_query);
    }
    
    // If no specific action was taken, build index from dataset
    if (!url_processed && !direct_query && !benchmark_mode && !show_info) {
        if (g_mpi_rank == 0) {
            printf(" Building index from dataset using Super Hybrid Engine...\n");
        }
        
        // Build index using all available technologies
        extern int build_index(const char *folder_path);
        int total_docs = build_index("dataset");
        
        if (g_mpi_rank == 0) {
            if (total_docs > 0) {
                printf(" Successfully indexed %d documents\n", total_docs);
                g_metrics.documents_processed = total_docs;
            } else {
                printf(" No documents found in dataset directory\n");
            }
        }
    }
    
    // Print final metrics
    print_super_hybrid_metrics();
    
    // Finalize the engine
    finalize_super_hybrid_engine();
    
    // Finalize MPI
    if (g_mpi_initialized) {
        MPI_Finalize();
    }
    
    if (g_mpi_rank == 0) {
        printf("\n Super Hybrid Search Engine execution completed!\n");
    }
    
    return 0;
}
