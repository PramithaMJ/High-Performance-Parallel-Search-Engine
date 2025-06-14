#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "parser.h"
#include "index.h"
#include "ranking.h"

// Initialize stopwords
extern int is_stopword(const char *word); // Forward declaration

// A minimal basic version of the program for debugging
int main()
{
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
    printf("\nTop results (BM25):\n");
    rank_bm25(user_query, total_docs, 10);
    
    return 0;
    char query[256];
    fgets(query, sizeof(query), stdin);

    printf("\nTop results (BM25):\n");
    rank_bm25(query, total_docs, 10); // Top 10 results

    return 0;
}
