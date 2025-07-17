#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>      // OpenMP for parallel processing
#include <mpi.h>      // MPI for distributed parallelism
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
    int max_depth = 2;  // Default crawl depth
    int max_pages = 10; // Default max pages to crawl
    int thread_count = 4; // Default number of threads
    int mpi_procs_count = mpi_size; // Default number of MPI processes is already set
    int total_docs = 0; // Total documents in the index
    
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
        } else if (strcmp(argv[i], "-np") == 0 && i + 1 < argc) {
            // Set number of MPI processes
            mpi_procs_count = atoi(argv[i+1]);
            if (mpi_procs_count < 1) mpi_procs_count = 1;
            
            // Note: We can't change the actual number of MPI processes after MPI_Init
            // This is just for informational purposes in this context
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
            // Print OpenMP information
            extern void print_thread_info();
            print_thread_info();
        } else if (strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "-q") == 0 && i + 1 < argc) {
            // Direct query option from command line
            const char* direct_query = argv[i + 1];
            
            // Build the index if needed
            if (!url_processed) {
                if (mpi_rank == 0) {
                    printf("Initializing stopwords...\n");
                }
                is_stopword("test");
                if (mpi_rank == 0) {
                    printf("Stopwords loaded.\n");
                    
                    // Clear any existing index
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
                    } else {
                        // Print debug information about the index
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
                rank_bm25(direct_query, total_docs, 10); // Top 10 results
            }
            
            // All processes must call MPI_Finalize
            MPI_Barrier(MPI_COMM_WORLD); // Make sure all processes reach this point
            
            // Finalize MPI before exiting
            if (mpi_initialized) {
                MPI_Finalize();
            }
            
            return total_docs > 0 ? 0 : 1; // Exit with error code if no docs
            
            i++; // Skip the query parameter
        }
    }
    
    // If we just processed a URL, let the user know we'll continue with search
    if (url_processed) {
        printf("\nURL content has been downloaded and will be included in the search index.\n");
        printf("Continuing with search engine startup...\n\n");
    }

    printf("Initializing stopwords...\n");
    is_stopword("test");
    printf("Stopwords loaded.\n");
    
    // Clear any existing index
    clear_index();
    
    printf("Building index from dataset directory...\n");
    total_docs = build_index("dataset");
    printf("Indexed %d documents.\n", total_docs);
    
    // If we made it here, we can search
    // Ensure only rank 0 handles the interactive query input
    char user_query[256] = {0};
    
    // MPI barrier to ensure all processes are ready
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Only rank 0 collects the query from the user
    if (mpi_rank == 0) {
        // Add some visual separation after crawling output
        printf("\n==================================================\n");
        printf("                SEARCH ENGINE READY                \n");
        printf("==================================================\n\n");
        
        printf("Enter your search query: ");
        fflush(stdout); // Make sure the prompt is displayed immediately
        
        // Clear any input buffer before reading
        int c;
        while ((c = getchar()) != '\n' && c != EOF && c != '\0') { /* discard */ }
        
        // Read user input for search
        if (fgets(user_query, sizeof(user_query), stdin) == NULL) {
            printf("Error reading input. Please try again.\n");
            // Handle the error by setting an empty query
            user_query[0] = '\0';
        } else {
            // Remove newline character if present
            int len = strlen(user_query);
            if (len > 0 && user_query[len-1] == '\n') {
                user_query[len-1] = '\0';
            }
            printf("Processing query: \"%s\"\n", user_query);
        }
    }
    
    // Broadcast the query from rank 0 to all processes
    MPI_Bcast(user_query, sizeof(user_query), MPI_CHAR, 0, MPI_COMM_WORLD);
    
    // Check if query is empty
    if (strlen(user_query) == 0) {
        if (mpi_rank == 0) {
            printf("No search query entered. Please restart the program and enter a query.\n");
        }
    } else {
        // Only rank 0 displays the search messages
        if (mpi_rank == 0) {
            printf("\nSearching for: \"%s\"\n", user_query);
            printf("\nTop results (BM25):\n");
        }
        
        // Only search if we have documents
        if (total_docs > 0) {
            // Wait for all processes before searching
            MPI_Barrier(MPI_COMM_WORLD);
            rank_bm25(user_query, total_docs, 10); // Top 10 results
            
            // After search is complete, display completion message (only on rank 0)
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
    extern SpeedupMetrics speedup_metrics;  // Declare the external variable
    calculate_speedup(&speedup_metrics);
    
    // Option to save current metrics as new baseline (only for rank 0)
    if (mpi_rank == 0) {
        char save_option;
        char input_buffer[10];
        
        printf("\nSave current performance as new baseline? (y/n): ");
        fflush(stdout); // Make sure prompt is displayed immediately
        
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
    
    // All processes wait here before finishing
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Finalize MPI if we initialized it
    if (mpi_initialized) {
        MPI_Finalize();
    }
    
    return 0;
}
