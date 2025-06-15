#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
    printf("  -h         Show this help message\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s -c https://medium.com/@lpramithamj\n", program_name);
    printf("  %s -m @lpramithamj\n", program_name);
    printf("  %s -c https://example.com -d 3 -p 20\n", program_name);
}

// Forward declaration for crawling function
extern int crawl_website(const char* start_url, int maxDepth, int maxPages);

int main(int argc, char* argv[])
{
    // Initialize metrics
    init_metrics();
    
    // Start timing total execution
    start_timer();
    
    // Process command line arguments
    int url_processed = 0;
    int max_depth = 2;  // Default crawl depth
    int max_pages = 10; // Default max pages to crawl
    
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

    printf("Initializing stopwords...\n");
    is_stopword("test");
    printf("Stopwords loaded.\n");
    
    // Clear any existing index
    clear_index();
    
    printf("Building index from dataset directory...\n");
    int total_docs = build_index("dataset");
    printf("Indexed %d documents.\n", total_docs);
    
    // If we made it here, we can search
    printf("Search engine ready for queries.\n");
    
    // Read user input for search
    char user_query[256];
    printf("Enter your search query: ");
    fgets(user_query, sizeof(user_query), stdin);
    
    // Remove newline character if present
    int len = strlen(user_query);
    if (len > 0 && user_query[len-1] == '\n') {
        user_query[len-1] = '\0';
    }
    
    printf("\nSearching for: %s\n", user_query);
    printf("\nTop results (BM25):\n");
    rank_bm25(user_query, total_docs, 10); // Top 10 results
    
    // Calculate total execution time
    metrics.total_time = stop_timer();
    metrics.memory_usage_after = get_current_memory_usage();
    
    // Print all metrics
    print_metrics();
    
    // Load baseline metrics and calculate speedup
    init_baseline_metrics("data/serial_metrics.csv");
    extern SpeedupMetrics speedup_metrics;  // Declare the external variable
    calculate_speedup(&speedup_metrics);
    
    // Option to save current metrics as new baseline
    char save_option;
    printf("\nSave current performance as new baseline? (y/n): ");
    scanf("%c", &save_option);
    if (save_option == 'y' || save_option == 'Y') {
        save_as_baseline("data/serial_metrics.csv");
    }
    
    return 0;
}
