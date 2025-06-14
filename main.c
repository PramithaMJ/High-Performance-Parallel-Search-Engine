#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "parser.h"
#include "index.h"
#include "ranking.h"
#include "crawler.h"

// Initialize stopwords
extern int is_stopword(const char *word); // Forward declaration

// External function declarations for web crawling
extern char* download_url(const char *url);

// Print usage instructions
void print_usage(const char* program_name) {
    printf("Usage: %s [options]\n", program_name);
    printf("Options:\n");
    printf("  -u URL     Download and index content from URL\n");
    printf("  -h         Show this help message\n");
}

int main(int argc, char* argv[])
{
    // Process command line arguments
    int url_processed = 0;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-u") == 0 && i + 1 < argc) {
            // Download URL content
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
    printf("\nTop results:\n");
    rank_bm25(user_query, total_docs, 10);
    
    return 0;
    char query[256];
    fgets(query, sizeof(query), stdin);

    printf("\nTop results (BM25):\n");
    rank_bm25(query, total_docs, 10); // Top 10 results

    return 0;
}
