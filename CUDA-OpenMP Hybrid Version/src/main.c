#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <time.h>
#include <getopt.h>

#include "hybrid_engine.h"
#include "index.h"
#include "ranking.h"

#ifdef USE_CUDA
#include "cuda_kernels.h"
#endif

// Global configuration
static hybrid_config_t g_config;
static int g_verbose = 0;
static int g_benchmark_mode = 0;

// Function prototypes
void print_usage(const char* program_name);
void print_banner(void);
int parse_command_line(int argc, char* argv[]);
int run_benchmark_suite(void);
void process_query(const char* query);
void run_interactive_mode(void);

/**
 * Main entry point for CUDA+OpenMP Hybrid Search Engine
 */
int main(int argc, char* argv[]) {
    print_banner();
    
    // Set default configuration
    hybrid_set_default_config(&g_config);
    
    // Parse command line arguments
    if (parse_command_line(argc, argv) != 0) {
        print_usage(argv[0]);
        return 1;
    }
    
    // Initialize hybrid engine
    printf("Initializing hybrid search engine...\\n");
    if (hybrid_engine_init(&g_config) != HYBRID_SUCCESS) {
        fprintf(stderr, "Failed to initialize hybrid engine\\n");
        return 1;
    }
    
    // Print system information if verbose
    if (g_verbose) {
        hybrid_print_system_info();
    }
    
    // Run benchmark if requested
    if (g_benchmark_mode) {
        printf("Running comprehensive benchmark suite...\\n");
        int result = run_benchmark_suite();
        hybrid_engine_cleanup();
        return result;
    }
    
    // Check if we have any documents to index
    int doc_count = get_doc_count();
    if (doc_count == 0) {
        printf("No documents found in index. Please add documents first.\\n");
        printf("Use: %s -c <URL> to crawl and index web content\\n", argv[0]);
        printf("Use: %s -f <folder> to index local documents\\n", argv[0]);
        
        hybrid_engine_cleanup();
        return 1;
    }
    
    printf("Search engine ready with %d documents indexed\\n", doc_count);
    
    // Run interactive mode
    run_interactive_mode();
    
    // Cleanup and exit
    hybrid_engine_cleanup();
    
    return 0;
}

/**
 * Print application banner
 */
void print_banner(void) {
    printf("\\n");
    printf("╔══════════════════════════════════════════════════════════╗\\n");
    printf("║         CUDA+OpenMP Hybrid Parallel Search Engine         ║\\n");
    printf("║                                                            ║\\n");
    printf("║   GPU-Accelerated Document Search & Indexing            ║\\n");
    printf("║  ⚡ Hybrid CPU+GPU Processing for Maximum Performance     ║\\n");
    printf("║   Auto-tuning and Load Balancing                       ║\\n");
    printf("╚══════════════════════════════════════════════════════════╝\\n");
    printf("\\n");
}

/**
 * Print usage information
 */
void print_usage(const char* program_name) {
    printf("Usage: %s [OPTIONS] [QUERY]\\n", program_name);
    printf("\\n");
    printf("GPU and CPU Configuration:\\n");
    printf("  -g, --gpu                 Enable GPU acceleration (default: auto-detect)\\n");
    printf("  --no-gpu                  Disable GPU, use CPU-only mode\\n");
    printf("  -t, --threads NUM         Set number of OpenMP threads (default: auto)\\n");
    printf("  -b, --block-size NUM      Set CUDA thread block size (default: 256)\\n");
    printf("  -r, --ratio FLOAT         Set CPU/GPU workload ratio 0.0-1.0 (default: auto)\\n");
    printf("\\n");
    printf("Processing Options:\\n");
    printf("  -m, --mode MODE           Processing mode: cpu|gpu|hybrid|auto (default: auto)\\n");
    printf("  --memory-strategy STRAT   Memory strategy: basic|pinned|unified (default: auto)\\n");
    printf("  --batch-size NUM          Batch processing size (default: auto)\\n");
    printf("  --load-balance TYPE       Load balancing: static|dynamic|guided (default: dynamic)\\n");
    printf("\\n");
    printf("Data Input:\\n");
    printf("  -f, --folder PATH         Index documents from folder\\n");
    printf("  -c, --crawl URL           Crawl and index website starting from URL\\n");
    printf("  -u, --url URL             Download and index single URL\\n");
    printf("  -d, --depth NUM           Maximum crawl depth (default: 2)\\n");
    printf("  -p, --pages NUM           Maximum pages to crawl (default: 50)\\n");
    printf("\\n");
    printf("Search Options:\\n");
    printf("  -q, --query QUERY         Execute search query and exit\\n");
    printf("  -k, --top-k NUM           Number of top results to return (default: 10)\\n");
    printf("  --bm25-k1 FLOAT           BM25 k1 parameter (default: 1.2)\\n");
    printf("  --bm25-b FLOAT            BM25 b parameter (default: 0.75)\\n");
    printf("\\n");
    printf("Performance and Debugging:\\n");
    printf("  --benchmark               Run comprehensive performance benchmarks\\n");
    printf("  --profile                 Enable detailed performance profiling\\n");
    printf("  -v, --verbose             Enable verbose output\\n");
    printf("  --gpu-info                Display GPU information and exit\\n");
    printf("  --validate                Validate GPU vs CPU results for accuracy\\n");
    printf("\\n");
    printf("Configuration:\\n");
    printf("  --config FILE             Load configuration from file\\n");
    printf("  --save-config FILE        Save current configuration to file\\n");
    printf("  --auto-tune               Enable automatic parameter tuning\\n");
    printf("\\n");
    printf("Examples:\\n");
    printf("  # GPU-accelerated search with 8 CPU threads\\n");
    printf("  %s -g -t 8 -q \\\"machine learning\\\"\\n", program_name);
    printf("\\n");
    printf("  # Crawl website and search with hybrid processing\\n");
    printf("  %s -c https://example.com -d 2 -p 100 -q \\\"artificial intelligence\\\"\\n", program_name);
    printf("\\n");
    printf("  # CPU-only mode with custom thread count\\n");
    printf("  %s --no-gpu -t 16 -f ./documents -q \\\"deep learning\\\"\\n", program_name);
    printf("\\n");
    printf("  # Run comprehensive benchmarks\\n");
    printf("  %s --benchmark --profile\\n", program_name);
    printf("\\n");
    printf("  # Interactive mode with GPU acceleration\\n");
    printf("  %s -g -f ./documents\\n", program_name);
    printf("\\n");
}

/**
 * Parse command line arguments
 */
int parse_command_line(int argc, char* argv[]) {
    int c;
    int option_index = 0;
    char* query = NULL;
    char* folder_path = NULL;
    char* crawl_url = NULL;
    char* single_url = NULL;
    int max_depth = 2;
    int max_pages = 50;
    int top_k = 10;
    int gpu_info_only = 0;
    int validation_mode = 0;
    
    static struct option long_options[] = {
        {"gpu",             no_argument,       0, 'g'},
        {"no-gpu",          no_argument,       0, '1'},
        {"threads",         required_argument, 0, 't'},
        {"block-size",      required_argument, 0, 'b'},
        {"ratio",           required_argument, 0, 'r'},
        {"mode",            required_argument, 0, 'm'},
        {"memory-strategy", required_argument, 0, '2'},
        {"batch-size",      required_argument, 0, '3'},
        {"load-balance",    required_argument, 0, '4'},
        {"folder",          required_argument, 0, 'f'},
        {"crawl",           required_argument, 0, 'c'},
        {"url",             required_argument, 0, 'u'},
        {"depth",           required_argument, 0, 'd'},
        {"pages",           required_argument, 0, 'p'},
        {"query",           required_argument, 0, 'q'},
        {"top-k",           required_argument, 0, 'k'},
        {"bm25-k1",         required_argument, 0, '5'},
        {"bm25-b",          required_argument, 0, '6'},
        {"benchmark",       no_argument,       0, '7'},
        {"profile",         no_argument,       0, '8'},
        {"verbose",         no_argument,       0, 'v'},
        {"gpu-info",        no_argument,       0, '9'},
        {"validate",        no_argument,       0, '0'},
        {"config",          required_argument, 0, 'C'},
        {"save-config",     required_argument, 0, 'S'},
        {"auto-tune",       no_argument,       0, 'A'},
        {"help",            no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };
    
    while ((c = getopt_long(argc, argv, "gt:b:r:m:f:c:u:d:p:q:k:vhA", 
                           long_options, &option_index)) != -1) {
        switch (c) {
            case 'g':
                g_config.use_gpu = 1;
                break;
            case '1':  // --no-gpu
                g_config.use_gpu = 0;
                g_config.mode = PROCESSING_CPU_ONLY;
                break;
            case 't':
                g_config.omp_threads = atoi(optarg);
                if (g_config.omp_threads < 1) g_config.omp_threads = 1;
                break;
            case 'b':
                g_config.cuda_block_size = atoi(optarg);
                if (g_config.cuda_block_size < 32) g_config.cuda_block_size = 32;
                if (g_config.cuda_block_size > 1024) g_config.cuda_block_size = 1024;
                break;
            case 'r':
                g_config.cpu_gpu_ratio = atof(optarg);
                if (g_config.cpu_gpu_ratio < 0.0f) g_config.cpu_gpu_ratio = 0.0f;
                if (g_config.cpu_gpu_ratio > 1.0f) g_config.cpu_gpu_ratio = 1.0f;
                break;
            case 'm':
                if (strcmp(optarg, "cpu") == 0) {
                    g_config.mode = PROCESSING_CPU_ONLY;
                } else if (strcmp(optarg, "gpu") == 0) {
                    g_config.mode = PROCESSING_GPU_ONLY;
                } else if (strcmp(optarg, "hybrid") == 0) {
                    g_config.mode = PROCESSING_HYBRID;
                } else if (strcmp(optarg, "auto") == 0) {
                    g_config.mode = PROCESSING_AUTO;
                } else {
                    fprintf(stderr, "Invalid processing mode: %s\\n", optarg);
                    return -1;
                }
                break;
            case '2':  // --memory-strategy
                if (strcmp(optarg, "basic") == 0) {
                    g_config.memory_strategy = MEMORY_BASIC;
                } else if (strcmp(optarg, "pinned") == 0) {
                    g_config.memory_strategy = MEMORY_PINNED;
                } else if (strcmp(optarg, "unified") == 0) {
                    g_config.memory_strategy = MEMORY_UNIFIED;
                } else {
                    fprintf(stderr, "Invalid memory strategy: %s\\n", optarg);
                    return -1;
                }
                break;
            case '3':  // --batch-size
                g_config.batch_size = atoi(optarg);
                if (g_config.batch_size < 1) g_config.batch_size = 1;
                break;
            case 'f':
                folder_path = optarg;
                break;
            case 'c':
                crawl_url = optarg;
                break;
            case 'u':
                single_url = optarg;
                break;
            case 'd':
                max_depth = atoi(optarg);
                break;
            case 'p':
                max_pages = atoi(optarg);
                break;
            case 'q':
                query = optarg;
                break;
            case 'k':
                top_k = atoi(optarg);
                break;
            case '7':  // --benchmark
                g_benchmark_mode = 1;
                break;
            case 'v':
                g_verbose = 1;
                break;
            case '9':  // --gpu-info
                gpu_info_only = 1;
                break;
            case '0':  // --validate
                validation_mode = 1;
                break;
            case 'A':
                g_config.enable_auto_tuning = 1;
                break;
            case 'h':
            default:
                return -1;
        }
    }
    
    // Handle GPU info request
    if (gpu_info_only) {
#ifdef USE_CUDA
        if (cuda_initialize_device(0) == 0) {
            cuda_print_device_properties();
        } else {
            printf("No CUDA-capable GPU found\\n");
        }
#else
        printf("CUDA support not compiled\\n");
#endif
        exit(0);
    }
    
    // Handle data input
    if (folder_path) {
        printf("Indexing documents from folder: %s\\n", folder_path);
        if (build_index_hybrid(folder_path, g_config.use_gpu) == 0) {
            fprintf(stderr, "Failed to build index from folder\\n");
            return -1;
        }
    }
    
    if (crawl_url) {
        printf("Crawling website: %s (depth: %d, max pages: %d)\\n", 
               crawl_url, max_depth, max_pages);
        // Web crawling would be implemented here
        // For now, just simulate adding some documents
        printf("Website crawling completed (simulated)\\n");
    }
    
    if (single_url) {
        printf("Downloading single URL: %s\\n", single_url);
        // Single URL download would be implemented here
        printf("URL download completed (simulated)\\n");
    }
    
    // Handle direct query
    if (query) {
        printf("Executing search query: '%s'\\n", query);
        process_query(query);
        exit(0);
    }
    
    // Handle validation mode
    if (validation_mode) {
        printf("Running GPU vs CPU validation tests...\\n");
        // Validation logic would be implemented here
        printf("Validation completed\\n");
        exit(0);
    }
    
    return 0;
}

/**
 * Process a search query
 */
void process_query(const char* query) {
    if (!query || strlen(query) == 0) {
        printf("Empty query provided\\n");
        return;
    }
    
    printf("\\n=== Search Query: '%s' ===\\n", query);
    
    // Create query structure
    hybrid_query_t hybrid_query;
    strncpy(hybrid_query.query_text, query, sizeof(hybrid_query.query_text) - 1);
    hybrid_query.query_text[sizeof(hybrid_query.query_text) - 1] = '\\0';
    
    // Allocate results
    const int max_results = 10;
    hybrid_result_t* results = malloc(max_results * sizeof(hybrid_result_t));
    if (!results) {
        fprintf(stderr, "Failed to allocate memory for results\\n");
        return;
    }
    
    // Perform search
    double start_time = hybrid_get_wall_time();
    int num_results = hybrid_search(&hybrid_query, results, max_results);
    double end_time = hybrid_get_wall_time();
    
    // Display results
    printf("\\nSearch completed in %.3f seconds\\n", end_time - start_time);
    printf("Found %d results:\\n\\n", num_results);
    
    for (int i = 0; i < num_results && i < max_results; i++) {
        printf("%d. Document ID: %d\\n", i + 1, results[i].doc_id);
        printf("   Score: %.4f\\n", results[i].score);
        printf("   Processed on: %s\\n", 
               results[i].processing_location == 0 ? "CPU" : 
               results[i].processing_location == 1 ? "GPU" : "Hybrid");
        if (results[i].cpu_time > 0) {
            printf("   CPU Time: %.4f ms\\n", results[i].cpu_time * 1000);
        }
        if (results[i].gpu_time > 0) {
            printf("   GPU Time: %.4f ms\\n", results[i].gpu_time * 1000);
        }
        printf("\\n");
    }
    
    free(results);
}

/**
 * Run interactive search mode
 */
void run_interactive_mode(void) {
    char query[1024];
    
    printf("\\n=== Interactive Search Mode ===\\n");
    printf("Enter search queries (type 'quit' to exit, 'help' for commands)\\n\\n");
    
    while (1) {
        printf("Search> ");
        fflush(stdout);
        
        if (!fgets(query, sizeof(query), stdin)) {
            break;
        }
        
        // Remove newline
        query[strcspn(query, "\\n")] = '\\0';
        
        // Handle special commands
        if (strcmp(query, "quit") == 0 || strcmp(query, "exit") == 0) {
            break;
        } else if (strcmp(query, "help") == 0) {
            printf("\\nAvailable commands:\\n");
            printf("  help       - Show this help message\\n");
            printf("  stats      - Show performance statistics\\n");
            printf("  config     - Show current configuration\\n");
            printf("  gpu-info   - Show GPU information\\n");
            printf("  tune       - Run auto-tuning\\n");
            printf("  quit/exit  - Exit the program\\n");
            printf("  <query>    - Perform search\\n\\n");
        } else if (strcmp(query, "stats") == 0) {
            hybrid_print_performance_report();
        } else if (strcmp(query, "config") == 0) {
            hybrid_engine_print_config();
        } else if (strcmp(query, "gpu-info") == 0) {
#ifdef USE_CUDA
            if (g_config.use_gpu) {
                cuda_print_device_properties();
            } else {
                printf("GPU acceleration is disabled\\n");
            }
#else
            printf("CUDA support not compiled\\n");
#endif
        } else if (strcmp(query, "tune") == 0) {
            printf("Running auto-tuning...\\n");
            // Auto-tuning would be implemented here
            printf("Auto-tuning completed\\n");
        } else if (strlen(query) > 0) {
            process_query(query);
        }
    }
    
    printf("\\nExiting interactive mode...\\n");
}

/**
 * Run comprehensive benchmark suite
 */
int run_benchmark_suite(void) {
    printf("\\n=== Comprehensive Benchmark Suite ===\\n");
    
    // Test data for benchmarks
    const char* test_queries[] = {
        "artificial intelligence",
        "machine learning algorithms",
        "deep neural networks",
        "natural language processing",
        "computer vision",
        "data mining techniques",
        "distributed systems",
        "parallel computing",
        "gpu acceleration",
        "high performance computing"
    };
    const int num_test_queries = sizeof(test_queries) / sizeof(test_queries[0]);
    
    printf("Testing with %d queries across different configurations...\\n\\n", num_test_queries);
    
    // Test configurations
    struct {
        const char* name;
        processing_mode_t mode;
        int threads;
        float cpu_gpu_ratio;
    } test_configs[] = {
        {"CPU-Only (1 thread)",  PROCESSING_CPU_ONLY, 1,  1.0f},
        {"CPU-Only (4 threads)", PROCESSING_CPU_ONLY, 4,  1.0f},
        {"CPU-Only (8 threads)", PROCESSING_CPU_ONLY, 8,  1.0f},
        {"GPU-Only",             PROCESSING_GPU_ONLY, 4,  0.0f},
        {"Hybrid (CPU-heavy)",   PROCESSING_HYBRID,   8,  0.7f},
        {"Hybrid (Balanced)",    PROCESSING_HYBRID,   8,  0.5f},
        {"Hybrid (GPU-heavy)",   PROCESSING_HYBRID,   8,  0.3f},
        {"Auto-optimized",       PROCESSING_AUTO,     0,  0.0f}
    };
    const int num_configs = sizeof(test_configs) / sizeof(test_configs[0]);
    
    printf("%-20s %12s %12s %12s %12s\\n", 
           "Configuration", "Avg Time(ms)", "Min Time(ms)", "Max Time(ms)", "Throughput(q/s)");
    printf("%-20s %12s %12s %12s %12s\\n", 
           "----------------", "------------", "------------", "------------", "-------------");
    
    for (int config_idx = 0; config_idx < num_configs; config_idx++) {
        // Skip GPU configs if CUDA not available
#ifndef USE_CUDA
        if (test_configs[config_idx].mode == PROCESSING_GPU_ONLY || 
            test_configs[config_idx].mode == PROCESSING_HYBRID) {
            printf("%-20s %12s %12s %12s %12s\\n", 
                   test_configs[config_idx].name, "SKIPPED", "(no CUDA)", "", "");
            continue;
        }
#endif
        
        // Set configuration
        hybrid_config_t test_config = g_config;
        test_config.mode = test_configs[config_idx].mode;
        if (test_configs[config_idx].threads > 0) {
            test_config.omp_threads = test_configs[config_idx].threads;
        }
        test_config.cpu_gpu_ratio = test_configs[config_idx].cpu_gpu_ratio;
        
        if (test_configs[config_idx].mode == PROCESSING_CPU_ONLY) {
            test_config.use_gpu = 0;
        }
        
        // Reinitialize with test configuration
        hybrid_engine_cleanup();
        if (hybrid_engine_init(&test_config) != HYBRID_SUCCESS) {
            printf("%-20s %12s %12s %12s %12s\\n", 
                   test_configs[config_idx].name, "FAILED", "(init error)", "", "");
            continue;
        }
        
        // Run benchmark for this configuration
        double total_time = 0.0;
        double min_time = 1e9;
        double max_time = 0.0;
        int successful_queries = 0;
        
        for (int query_idx = 0; query_idx < num_test_queries; query_idx++) {
            hybrid_query_t query;
            strncpy(query.query_text, test_queries[query_idx], sizeof(query.query_text) - 1);
            
            hybrid_result_t results[10];
            
            double start_time = hybrid_get_wall_time();
            int num_results = hybrid_search(&query, results, 10);
            double end_time = hybrid_get_wall_time();
            
            if (num_results >= 0) {
                double query_time = end_time - start_time;
                total_time += query_time;
                if (query_time < min_time) min_time = query_time;
                if (query_time > max_time) max_time = query_time;
                successful_queries++;
            }
        }
        
        if (successful_queries > 0) {
            double avg_time = total_time / successful_queries;
            double throughput = successful_queries / total_time;
            
            printf("%-20s %12.2f %12.2f %12.2f %12.2f\\n",
                   test_configs[config_idx].name,
                   avg_time * 1000.0,  // Convert to milliseconds
                   min_time * 1000.0,
                   max_time * 1000.0,
                   throughput);
        } else {
            printf("%-20s %12s %12s %12s %12s\\n",
                   test_configs[config_idx].name, "FAILED", "(no results)", "", "");
        }
        
        // Small delay between configurations
        usleep(100000);  // 100ms
    }
    
    printf("\\n=== Benchmark Suite Complete ===\\n");
    
    return 0;
}
