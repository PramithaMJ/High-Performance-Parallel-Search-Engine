#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>      // OpenMP for parallel processing
#include <mpi.h>      // MPI for distributed parallelism
#include <unistd.h>   // For gethostname
#include "../include/parser.h"
#include "../include/index.h"
#include "../include/ranking.h"
#include "../include/crawler.h"
#include "../include/metrics.h"
#include "../include/benchmark.h"

// Initialize stopwords
extern int is_stopword(const char *word); // Forward declaration

// External function declarations for web crawling
extern char* download_url(const char *url);

// Print usage instructions
void print_usage(const char* program_name) {
    printf("Usage: %s [options]\n", program_name);
    printf("Options:\n");
    printf("  -u URL     Download and index content from URL\n");
    printf("  -c URL     Crawl website starting from URL (follows links)\n");
    printf("  -m USER    Crawl Medium profile for user (e.g., -m @username)\n");
    printf("  -d NUM     Maximum crawl depth (default: 2)\n");
    printf("  -p NUM     Maximum pages to crawl (default: 10)\n");
    printf("  -np NUM    Set number of MPI processes\n");
    printf("  -t NUM     Set number of OpenMP threads for parallel processing\n");
    printf("  -q QUERY   Run search with the specified query\n");
    printf("  -i         Print OpenMP information (threads, etc.)\n");
    printf("  -h         Show this help message\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s -c https://medium.com/@lpramithamj\n", program_name);
    printf("  %s -m @lpramithamj\n", program_name);
    printf("  %s -c https://example.com -d 3 -p 20\n", program_name);
    printf("  %s -np 4 -t 8 -q \"artificial intelligence\"\n", program_name);
    printf("  %s -np 2 \"machine learning\"\n", program_name);
}

// Forward declaration for crawling function
extern int crawl_website(const char* start_url, int maxDepth, int maxPages);

int main(int argc, char* argv[])
{
    // MPI variables
    int mpi_provided;
    int mpi_rank = 0;
    int mpi_size = 1;
    int mpi_initialized = 0;
    
    // Check if MPI is already initialized
    MPI_Initialized(&mpi_initialized);
    
    if (!mpi_initialized) {
        // Initialize MPI with thread support
        MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_provided);
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    }
    
    // Initialize metrics
    init_metrics();
    
    // Start timing total execution (only on rank 0)
    if (mpi_rank == 0) {
        start_timer();
    }
    
    // Process command line arguments
    int url_processed = 0;
    int max_depth = 2;
    int max_pages = 10;
    int thread_count = 4;
    int mpi_procs_count = mpi_size;
    int total_docs = 0;
    int error_flag = 0;
    
    extern void clear_index();
    
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
                error_flag = 1;
                goto cleanup;
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
                    if (max_pages < 20) max_pages = 20;
                    printf("Medium profile detected. Will crawl up to %d pages.\n", max_pages);
                }
            }
            
            int pages_crawled = crawl_website(url, max_depth, max_pages);
            
            if (pages_crawled > 0) {
                printf("Successfully crawled %d pages from %s\n", pages_crawled, url);
                url_processed = 1;
            } else {
                printf("Failed to crawl website from URL\n");
                error_flag = 1;
                goto cleanup;
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
            const char* username = argv[i + 1];
            char medium_url[256];
            
            if (username[0] == '@') {
                snprintf(medium_url, sizeof(medium_url), "https://medium.com/%s", username);
            } else {
                snprintf(medium_url, sizeof(medium_url), "https://medium.com/@%s", username);
            }
            
            printf("Crawling Medium profile: %s\n", medium_url);
            
            // Use  limits for Medium profiles
            int profile_max_depth = 3;
            int profile_max_pages = 25;
            
            int pages_crawled = crawl_website(medium_url, profile_max_depth, profile_max_pages);
            
            if (pages_crawled > 0) {
                printf("Successfully crawled %d pages from Medium profile %s\n", pages_crawled, username);
                url_processed = 1;
            } else {
                printf("Failed to crawl Medium profile\n");
                error_flag = 1;
                goto cleanup;
            }
            
            i++;
        } else if (strcmp(argv[i], "-np") == 0 && i + 1 < argc) {
            mpi_procs_count = atoi(argv[i+1]);
            if (mpi_procs_count < 1) mpi_procs_count = 1;
            
            // can't change the actual number of MPI processes after MPI_Init
            if (mpi_rank == 0) {
                if (mpi_size != mpi_procs_count) {
                    printf("Note: MPI was already initialized with %d processes\n", mpi_size);
                    printf("The requested %d processes will be used for workload distribution.\n", mpi_procs_count);
                } else {
                    printf("Using %d MPI processes for distributed processing\n", mpi_procs_count);
                }
            }
            i++;
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            // Set number of OpenMP threads for parallel processing
            thread_count = atoi(argv[i+1]);
            if (thread_count < 1) thread_count = 1;
            
            // Set OpenMP thread count globally
            omp_set_num_threads(thread_count);
            
            // Disable dynamic adjustment for more consistent thread allocation
            omp_set_dynamic(0);
            
            // Enable nested parallelism if available
            omp_set_nested(1);
            
            if (mpi_rank == 0) {
                printf("Set OpenMP thread count to: %d (dynamic threads disabled)\n", thread_count);
            }
            i++;
        } else if (strcmp(argv[i], "-i") == 0) {
            extern void print_thread_info();
            print_thread_info();
        } else if (strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            goto cleanup;
        } else if (strcmp(argv[i], "-q") == 0 && i + 1 < argc) {
            const char* direct_query = argv[i + 1];
            
            if (!url_processed) {
                if (mpi_rank == 0) {
                    printf("Initializing stopwords...\n");
                }
                is_stopword("test");
                if (mpi_rank == 0) {
                    printf("Stopwords loaded.\n");
                    
                    clear_index();
                    
                    printf("Building index from dataset directory...\n");
                }
                
                total_docs = build_index("dataset");
                
                if (mpi_rank == 0) {
                    printf("Indexed %d documents.\n", total_docs);
                    
                    if (total_docs == 0) {
                        printf("No documents found in dataset directory.\n");
                        printf("Please download content first using -u, -c, or -m options.\n");
                        printf("Example: %s -c https://medium.com/@lpramithamj -q \"microservice\"\n", argv[0]);
                        error_flag = 1;
                        goto cleanup;
                    } else {
                        extern void print_all_index_terms();
                        print_all_index_terms();
                    }
                }
            }
            
            if (mpi_rank == 0) {
                printf("\nSearching for: \"%s\"\n", direct_query);
                if (total_docs > 0) {
                    printf("\nTop results (BM25):\n");
                } else {
                    printf("No documents to search. Please crawl some content first.\n");
                }
            }
            
            if (total_docs > 0) {
                rank_bm25(direct_query, total_docs, 10);
            }
            MPI_Barrier(MPI_COMM_WORLD);
            goto cleanup;
            
            i++;
        }
    }
    
    if (url_processed) {
        printf("\nURL content has been downloaded and will be included in the search index.\n");
        printf("Continuing with search engine startup...\n\n");
    }

    printf("Initializing stopwords...\n");
    is_stopword("test");
    printf("Stopwords loaded.\n");
    
    clear_index();
    
    printf("Building index from dataset directory...\n");
    total_docs = build_index("dataset");
    printf("Indexed %d documents.\n", total_docs);
    
    char user_query[256] = {0};
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Only rank 0 collects the query from the user
    if (mpi_rank == 0) {
        printf("\n==================================================\n");
        printf("                SEARCH ENGINE READY                \n");
        printf("==================================================\n\n");
        
        printf("Enter your search query: ");
        fflush(stdout);
        
        int c;
        while ((c = getchar()) != '\n' && c != EOF && c != '\0') { /* discard */ }
        
        // Read user input for search
        if (fgets(user_query, sizeof(user_query), stdin) == NULL) {
            printf("Error reading input. Please try again.\n");
            user_query[0] = '\0';
        } else {
            int len = strlen(user_query);
            if (len > 0 && user_query[len-1] == '\n') {
                user_query[len-1] = '\0';
            }
            printf("Processing query: \"%s\"\n", user_query);
        }
    }
    
    MPI_Bcast(user_query, sizeof(user_query), MPI_CHAR, 0, MPI_COMM_WORLD);
    
    // Check if query is empty
    if (strlen(user_query) == 0) {
        if (mpi_rank == 0) {
            printf("No search query entered. Please restart the program and enter a query.\n");
        }
    } else {
        if (mpi_rank == 0) {
            printf("\nSearching for: \"%s\"\n", user_query);
            printf("\nTop results (BM25):\n");
        }
        // Only search if we have documents
        if (total_docs > 0) {
            MPI_Barrier(MPI_COMM_WORLD);
            rank_bm25(user_query, total_docs, 10);
            if (mpi_rank == 0) {
                printf("\n==================================================\n");
                printf("              SEARCH COMPLETED                    \n");
                printf("==================================================\n");
            }
        } else if (mpi_rank == 0) {
            printf("No documents in index. Please crawl content first using -c, -m, or -u options.\n");
        }
    }
    
    // Calculate total execution time
    metrics.total_time = stop_timer();
    metrics.memory_usage_after = get_current_memory_usage();
    
    // Print all metrics
    print_metrics();
    
    // Load baseline metrics and calculate speedup
    init_baseline_metrics("data/serial_metrics.csv");
    extern SpeedupMetrics speedup_metrics;
    calculate_speedup(&speedup_metrics);
    
    if (mpi_rank == 0) {
        char save_option;
        char input_buffer[10];
        
        printf("\nSave current performance as new baseline? (y/n): ");
        fflush(stdout);
        
        if (fgets(input_buffer, sizeof(input_buffer), stdin) != NULL) {
            save_option = input_buffer[0];
            if (save_option == 'y' || save_option == 'Y') {
                save_as_baseline("data/hybrid_metrics.csv");
                printf("Performance metrics saved as new baseline.\n");
            }
        } else {
            printf("Failed to read input. Metrics not saved.\n");
        }
    }
    
cleanup:
    MPI_Barrier(MPI_COMM_WORLD);

    // --- Distributed/Threaded Process Summary ---
    char local_hostname[256];
    gethostname(local_hostname, sizeof(local_hostname));
    int local_threads = omp_get_max_threads();
    char all_hostnames[128][256];
    int all_threads[128];
    int max_procs = 128;
    
    MPI_Gather(local_hostname, 256, MPI_CHAR, all_hostnames, 256, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_threads, 1, MPI_INT, all_threads, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        printf("\n==================== DISTRIBUTED PROCESS SUMMARY ====================\n");
        printf("%-8s %-25s %-15s\n", "Rank", "Hostname", "OpenMP Threads");
        printf("--------------------------------------------------------------------\n");
        for (int i = 0; i < mpi_size && i < max_procs; i++) {
            printf("%-8d %-25s %-15d\n", i, all_hostnames[i], all_threads[i]);
        }
        printf("--------------------------------------------------------------------\n");
        printf("Total MPI Processes: %d\n", mpi_size);
        if (error_flag) {
            printf("\n[WARNING] One or more processes encountered an error.\n");
        }
    }

    // Finalize MPI if we initialized it
    if (mpi_initialized) {
        MPI_Finalize();
    }
    return error_flag ? 1 : 0;
}
