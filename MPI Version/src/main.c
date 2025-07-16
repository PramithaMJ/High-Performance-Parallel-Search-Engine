#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "../include/parser.h"
#include "../include/index.h"
#include "../include/ranking.h"
#include "../include/crawler.h"
#include "../include/metrics.h"
#include "../include/benchmark.h"
#include "../include/load_balancer.h"
#include "../include/mpi_comm.h"
#include "../include/dist_index.h"
#include "../include/parallel_processor.h"

// Initialize stopwords
extern int is_stopword(const char *word); // Forward declaration

// External function declarations for web crawling
extern char* download_url(const char *url);

// Print usage instructions
void print_usage(const char* program_name) {
    printf("Options:\n");
    printf("  -np NUM    Number of MPI processes to use (for information only - use mpirun -np)\n");
    printf("  -u URL     Download and index content from URL\n");
    printf("  -c URL     Crawl website starting from URL (follows links)\n");
    printf("  -m USER    Crawl Medium profile for user (e.g., -m @username)\n");
    printf("  -d NUM     Maximum crawl depth (default: 2)\n");
    printf("  -p NUM     Maximum pages to crawl (default: 10)\n");
    printf("  -i         Print MPI process information\n");
    printf("  -h         Show this help message\n");
    printf("\n");
    printf("Examples:\n");
    printf(" -np 4 %s -c https://medium.com/@lpramithamj\n", program_name);
    printf(" -np 8 %s -m @lpramithamj -d 3 -p 25\n", program_name);
    printf(" -np 2 %s -c https://example.com -d 3 -p 20\n", program_name);
    printf("\n");
}

// Forward declaration for crawling function
extern int crawl_website(const char* start_url, int maxDepth, int maxPages);

int main(int argc, char* argv[])
{
    int rank, size;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Initialize the new components
    init_mpi_comm();
    init_load_balancer(size);
    
    // Set up global distributed index and parallel processor
    extern DistributedIndex* dist_index;
    extern ParallelProcessor* processor;
    dist_index = create_distributed_index(rank, size);
    processor = create_parallel_processor(rank, size);
    
    // Initialize metrics (only on master process)
    if (rank == 0) {
        init_metrics();
        
        // Start timing total execution
        start_timer();
        
        printf("MPI Search Engine started with %d processes\n", size);
    }
    
    // Process command line arguments
    int url_processed = 0;
    int max_depth = 2;  // Default crawl depth
    int max_pages = 10; // Default max pages to crawl
    int requested_num_processes = -1; // Number of processes requested with -np flag
    
    // First clear any existing index to make sure we rebuild it from scratch
    extern void clear_index(); // Forward declaration for the function we'll add
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-u") == 0 && i + 1 < argc) {
            // Download single URL content
            const char* url = argv[i + 1];
            printf("Downloading content from URL: %s\n", url);
            char* filepath = download_url(url);
            
            if (filepath) {
                printf("Successfully downloaded content to %s\n", filepath);
                url_processed = 1;
            } else {
                printf("Failed to download content from URL\n");
                return 1;
            }
            
            // Skip the URL parameter
            i++;
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            // Crawl website starting from URL
            const char* url = argv[i + 1];
            printf("Starting website crawl from URL: %s\n", url);
            
            // Special handling for Medium.com URLs
            if (strstr(url, "medium.com") != NULL) {
                printf("Detected Medium.com URL. Optimizing crawler settings for Medium...\n");
                
                // For Medium profile URLs, use more aggressive crawling
                if (strstr(url, "medium.com/@") != NULL) {
                    if (max_pages < 20) max_pages = 20; // Increase default for profiles
                    printf("Medium profile detected. Will crawl up to %d pages.\n", max_pages);
                }
            }
            
            int pages_crawled = crawl_website(url, max_depth, max_pages);
            
            if (pages_crawled > 0) {
                printf("Successfully crawled %d pages from %s\n", pages_crawled, url);
                url_processed = 1;
            } else {
                printf("Failed to crawl website from URL\n");
                return 1;
            }
            
            // Skip the URL parameter
            i++;
        } else if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            // Set maximum crawl depth
            max_depth = atoi(argv[i+1]);
            if (max_depth < 1) max_depth = 1;
            if (max_depth > 5) {
                printf("Warning: High crawl depth may take a long time. Limited to 5.\n");
                max_depth = 5;
            }
            i++;
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            // Set maximum pages to crawl
            max_pages = atoi(argv[i+1]);
            if (max_pages < 1) max_pages = 1;
            if (max_pages > 100) {
                printf("Warning: High page limit may take a long time. Limited to 100.\n");
                max_pages = 100;
            }
            i++;
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            // Special option for Medium profile crawling
            const char* username = argv[i + 1];
            char medium_url[256];
            
            // Check if it already has @ prefix
            if (username[0] == '@') {
                snprintf(medium_url, sizeof(medium_url), "https://medium.com/%s", username);
            } else {
                snprintf(medium_url, sizeof(medium_url), "https://medium.com/@%s", username);
            }
            
            printf("Crawling Medium profile: %s\n", medium_url);
            
            // Use higher limits for Medium profiles
            int profile_max_depth = 3;
            int profile_max_pages = 25;
            
            int pages_crawled = crawl_website(medium_url, profile_max_depth, profile_max_pages);
            
            if (pages_crawled > 0) {
                printf("Successfully crawled %d pages from Medium profile %s\n", pages_crawled, username);
                url_processed = 1;
            } else {
                printf("Failed to crawl Medium profile\n");
                return 1;
            }
            
            i++; // Skip the username parameter
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            // In MPI version, we don't set thread count here, but inform user
            printf("Note: In MPI version, use mpirun -np <NUM_PROCESSES> to control parallelism.\n");
            printf("The -t flag is ignored in MPI version.\n");
            i++;
        } else if (strcmp(argv[i], "-i") == 0) {
            // Print MPI information
            extern void print_thread_info();
            print_thread_info();
        } else if (strcmp(argv[i], "-np") == 0 && i + 1 < argc) {
            // Set the number of MPI processes
            requested_num_processes = atoi(argv[i+1]);
            if (requested_num_processes < 1) {
                if (rank == 0) {
                    printf("Warning: Invalid number of processes. Using available MPI processes (%d).\n", size);
                }
            } else if (requested_num_processes != size) {
                if (rank == 0) {
                    printf("WARNING: Requested %d processes but running with %d processes.\n", 
                           requested_num_processes, size);
                    printf("To run with %d processes, use: mpirun -np %d %s [other options]\n", 
                           requested_num_processes, requested_num_processes, argv[0]);
                    printf("Current execution will proceed with %d process(es).\n", size);
                }
            }
            i++;
        } else if (strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    // If we just processed a URL, let the user know we'll continue with search
    if (url_processed) {
        printf("\nURL content has been downloaded and will be included in the search index.\n");
        printf("Continuing with search engine startup...\n\n");
    }
    
    // Print information about the number of processes being used
    if (rank == 0) {
        printf("Running with %d MPI processes\n", size);
        printf("Initializing stopwords...\n");
    }
    
    is_stopword("test");
    
    if (rank == 0) {
        printf("Stopwords loaded.\n");
    }
    
    // Clear any existing index
    clear_index();
    
    if (rank == 0) {
        printf("Building index from dataset directory...\n");
    }
    
    int total_docs = build_index("dataset");
    
    // Synchronize all processes before continuing
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Indexed %d documents.\n", total_docs);
        printf("Search engine ready for queries.\n");
        printf("Enter your search query: ");
        fflush(stdout);
    }
    
    // Only rank 0 handles user input
    char user_query[256];
    if (rank == 0) {
        if (fgets(user_query, sizeof(user_query), stdin) == NULL) {
            strcpy(user_query, "default");
        }
        
        // Remove newline character if present
        int len = strlen(user_query);
        if (len > 0 && user_query[len-1] == '\n') {
            user_query[len-1] = '\0';
        }
        
        printf("\nSearching for: %s\n", user_query);
        printf("\nTop results (BM25):\n");
    }
    
    // Broadcast the query to all processes
    MPI_Bcast(user_query, 256, MPI_CHAR, 0, MPI_COMM_WORLD);
    
    // All processes participate in search, but only rank 0 shows results
    rank_bm25(user_query, total_docs, 10);
    
    // Synchronize all processes before calculating metrics
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Only the master process (rank 0) handles metrics and user interaction
    if (rank == 0) {
        // Calculate total execution time
        metrics.total_time = stop_timer();
        metrics.memory_usage_after = get_current_memory_usage();
        
        // Print all metrics
        print_metrics();
        
        // Load baseline metrics and calculate speedup
        init_baseline_metrics("../Serial Version/data/serial_metrics.csv");
        extern SpeedupMetrics speedup_metrics;  // Declare the external variable
        calculate_speedup(&speedup_metrics);
        
        // Option to save current metrics as new baseline
        char save_option;
        printf("\nSave current performance as new baseline? (y/n): ");
        fflush(stdout);
        if (scanf(" %c", &save_option) == 1) {
            if (save_option == 'y' || save_option == 'Y') {
                save_as_baseline("data/mpi_metrics.csv");
            }
        }
    }
    
    // Wait for all processes to complete before cleanup
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Clean up the new components
    free_parallel_processor(processor);
    free_distributed_index(dist_index);
    free_mpi_comm();
    free_load_balancer();
    
    // Finalize MPI
    MPI_Finalize();
    
    return 0;
}
