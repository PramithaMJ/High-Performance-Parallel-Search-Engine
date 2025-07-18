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

extern int is_stopword(const char *word);

extern char* download_url(const char *url);

void print_usage(const char* program_name) {
    printf("MPI-Based Parallel Search Engine\n");
    printf("Usage: mpirun -np <NUM_PROCESSES> %s [OPTIONS]\n", program_name);
    printf("   OR: %s -np <NUM_PROCESSES> [OPTIONS]\n\n", program_name);
    printf("Options:\n");
    printf("  -np NUM    Number of MPI processes to use\n");
    printf("  -u URL     Download and index content from URL\n");
    printf("  -c URL     Crawl website starting from URL (follows links)\n");
    printf("  -m USER    Crawl Medium profile for user (e.g., -m @username)\n");
    printf("  @username  Shortcut for Medium profile (e.g., @lpramithamj)\n");
    printf("  -d NUM     Maximum crawl depth (default: 2)\n");
    printf("  -p NUM     Maximum pages to crawl (default: 10)\n");
    printf("  -i         Print MPI process information\n");
    printf("  -h         Show this help message\n");
    printf("\n");
    printf("Examples:\n");
    printf("  mpirun -np 4 %s -c https://medium.com/@lpramithamj\n", program_name);
    printf("  %s -np 8 -m @lpramithamj -d 3 -p 25\n", program_name);
    printf("  %s -np 6 @lpramithamj -d 2 -p 10\n", program_name);
    printf("\n");
    printf("Note: When using -np within the program, it will automatically launch mpirun.\n");
}

extern int crawl_website(const char* start_url, int maxDepth, int maxPages);

int main(int argc, char* argv[])
{
    int rank, size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    init_mpi_comm();
    init_load_balancer(size);
    
    extern DistributedIndex* dist_index;
    extern ParallelProcessor* processor;
    dist_index = create_distributed_index(rank, size);
    processor = create_parallel_processor(rank, size);
    
    if (rank == 0) {
        init_metrics();
        
        start_timer();
        
        printf("MPI Search Engine started with %d processes\n", size);
    }
    
    int url_processed = 0;
    int max_depth = 2;
    int max_pages = 10;
    int requested_processes = 1;
    char* medium_username = NULL;
    char* crawl_url = NULL;
    char* download_url_param = NULL;
    
    extern void clear_index();
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) {
            max_depth = atoi(argv[i+1]);
            if (max_depth < 1) max_depth = 1;
            if (max_depth > 5) {
                if (rank == 0) printf("Warning: High crawl depth may take a long time. Limited to 5.\n");
                max_depth = 5;
            }
            i++;
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            max_pages = atoi(argv[i+1]);
            if (max_pages < 1) max_pages = 1;
            if (max_pages > 100) {
                if (rank == 0) printf("Warning: High page limit may take a long time. Limited to 100.\n");
                max_pages = 100;
            }
            i++;
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            medium_username = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            crawl_url = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "-u") == 0 && i + 1 < argc) {
            download_url_param = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "-np") == 0 && i + 1 < argc) {
            // Handle -np flag for number of processes
            requested_processes = atoi(argv[i+1]);
            if (requested_processes < 1) requested_processes = 1;
            if (requested_processes > 16) {
                if (rank == 0) printf("Warning: High process count may be limited by system. Limited to 16.\n");
                requested_processes = 16;
            }
            
            if (size == 1 && requested_processes > 1) {
                if (rank == 0) {
                    printf("Detected -np %d flag. Automatically launching with MPI...\n", requested_processes);
                    
                    char mpi_command[2048];
                    snprintf(mpi_command, sizeof(mpi_command), "mpirun -np %d %s", requested_processes, argv[0]);
                    
                    // arguments to the command (skip -np and its value)
                    for (int j = 1; j < argc; j++) {
                        if (strcmp(argv[j], "-np") == 0) {
                            j++; 
                            continue;
                        }
                        strcat(mpi_command, " ");
                        strcat(mpi_command, argv[j]);
                    }
                    
                    printf("Executing: %s\n", mpi_command);
                    fflush(stdout);
                    
                    MPI_Finalize();
                    
                    int result = system(mpi_command);
                    exit(result >> 8);
                }
            }
            i++;
        } else if (argv[i][0] == '@' && strlen(argv[i]) > 1) {
            medium_username = argv[i] + 1;
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            if (rank == 0) {
                printf("Note: In MPI version, use mpirun -np <NUM_PROCESSES> to control parallelism.\n");
                printf("The -t flag is ignored in MPI version.\n");
            }
            i++;
        } else if (strcmp(argv[i], "-i") == 0) {
            extern void print_thread_info();
            print_thread_info();
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                print_usage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printf("Use -h for help.\n");
            }
        }
    }
    
    if (download_url_param) {
        if (rank == 0) printf("Downloading content from URL: %s\n", download_url_param);
        char* filepath = download_url(download_url_param);
        
        if (filepath) {
            if (rank == 0) printf("Successfully downloaded content to %s\n", filepath);
            url_processed = 1;
        } else {
            if (rank == 0) printf("Failed to download content from URL\n");
            MPI_Finalize();
            return 1;
        }
    }
    
    if (crawl_url) {
        if (rank == 0) {
            printf("Starting website crawl from URL: %s\n", crawl_url);
            
            if (strstr(crawl_url, "medium.com") != NULL) {
                printf("Detected Medium.com URL. Optimizing crawler settings for Medium...\n");
                
                if (strstr(crawl_url, "medium.com/@") != NULL) {
                    if (max_pages < 20) max_pages = 20; 
                    printf("Medium profile detected. Will crawl up to %d pages.\n", max_pages);
                }
            }
            
            printf("Using depth: %d, max pages: %d for crawling\n", max_depth, max_pages);
        }
        
        int pages_crawled = crawl_website(crawl_url, max_depth, max_pages);
        
        if (pages_crawled > 0) {
            if (rank == 0) printf("Successfully crawled %d pages from %s\n", pages_crawled, crawl_url);
            url_processed = 1;
        } else {
            if (rank == 0) printf("Failed to crawl website from URL\n");
            MPI_Finalize();
            return 1;
        }
    }
    
    if (medium_username) {
        char medium_url[256];
        
        if (medium_username[0] == '@') {
            snprintf(medium_url, sizeof(medium_url), "https://medium.com/%s", medium_username);
        } else {
            snprintf(medium_url, sizeof(medium_url), "https://medium.com/@%s", medium_username);
        }
        
        if (rank == 0) {
            printf("Crawling Medium profile: %s\n", medium_url);
            printf("Using depth: %d, max pages: %d for Medium profile crawling\n", max_depth, max_pages);
        }
        
        int pages_crawled = crawl_website(medium_url, max_depth, max_pages);
        
        if (pages_crawled > 0) {
            if (rank == 0) printf("Successfully crawled %d pages from Medium profile %s\n", pages_crawled, medium_username);
            url_processed = 1;
        } else {
            if (rank == 0) printf("Failed to crawl Medium profile\n");
            MPI_Finalize();
            return 1;
        }
    }
    
    if (url_processed) {
        printf("\nURL content has been downloaded and will be included in the search index.\n");
        printf("Continuing with search engine startup...\n\n");
    }
    
    if (rank == 0) {
        printf("Running with %d MPI processes\n", size);
        printf("Initializing stopwords...\n");
    }
    
    is_stopword("test");
    
    if (rank == 0) {
        printf("Stopwords loaded.\n");
    }
    
    clear_index();
    
    if (rank == 0) {
        printf("Building index from dataset directory...\n");
    }
    
    int total_docs = build_index("dataset");
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Indexed %d documents.\n", total_docs);
        printf("Search engine ready for queries.\n\n");
        fflush(stdout);
    }
    
    char user_query[256] = {0};
    int input_status = 0;
    
    // Make sure all processes are synchronized before getting input
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Enter your search query: ");
        fflush(stdout);
        
        printf("DEBUG: About to read input...\n");
        fflush(stdout);
        
        char *input_result = fgets(user_query, sizeof(user_query), stdin);
        
        printf("DEBUG: Input read attempt completed\n");
        fflush(stdout);
        
        if (input_result != NULL) {
            printf("DEBUG: Input received: '%s'\n", user_query);
            fflush(stdout);
            
            size_t len = strlen(user_query);
            if (len > 0 && user_query[len-1] == '\n') {
                user_query[len-1] = '\0';
            }
            
            printf("DEBUG: After newline removal: '%s'\n", user_query);
            fflush(stdout);
            
            input_status = 1;
        } else {
            printf("DEBUG: Input read failed\n");
            fflush(stdout);
            strcpy(user_query, "default");
            input_status = 0;
        }
        
        if (input_status) {
            printf("\nSearching for: %s\n", user_query);
            printf("\nTop results (BM25):\n");
        } else {
            printf("\nError reading input. Using default query: %s\n", user_query);
        }
        fflush(stdout);
    } 
    // Broadcast both the query and input status
    MPI_Bcast(&input_status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    MPI_Bcast(user_query, 256, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (!input_status) {
        MPI_Finalize();
        return 1;
    }
    
    rank_bm25(user_query, total_docs, 10);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        metrics.total_time = stop_timer();
        metrics.memory_usage_after = get_current_memory_usage();
        
        print_metrics();
        
        init_baseline_metrics("../Serial Version/data/serial_metrics.csv");
        extern SpeedupMetrics speedup_metrics; 
        calculate_speedup(&speedup_metrics);
        
        char save_option;
        printf("\nSave current performance as new baseline? (y/n): ");
        fflush(stdout);
        if (scanf(" %c", &save_option) == 1) {
            if (save_option == 'y' || save_option == 'Y') {
                save_as_baseline("data/mpi_metrics.csv");
            }
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    free_parallel_processor(processor);
    free_distributed_index(dist_index);
    free_mpi_comm();
    free_load_balancer();
    
    MPI_Finalize();
    
    return 0;
}
