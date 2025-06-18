#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Simplified versions of the functions from crawler.c for testing

#define MAX_URL_LENGTH 512

// Function to normalize a URL (remove tracking params, fragments, etc.)
static char* normalize_url(const char* url) {
    static char normalized[MAX_URL_LENGTH * 2];
    
    // Initialize to empty string
    normalized[0] = '\0';
    
    if (!url || strlen(url) == 0) {
        return normalized;
    }
    
    // Safe copy with null termination
    strncpy(normalized, url, MAX_URL_LENGTH * 2 - 1);
    normalized[MAX_URL_LENGTH * 2 - 1] = '\0';
    
    // Remove fragment identifiers (#)
    char* fragment = strchr(normalized, '#');
    if (fragment) *fragment = '\0';
    
    // Remove common tracking parameters
    char* query = strchr(normalized, '?');
    if (query) {
        // For medium.com, remove all query params as they're typically tracking
        if (strstr(normalized, "medium.com") != NULL) {
            *query = '\0';
        } else {
            // For other sites, try to keep important query params but remove common tracking ones
            if (strstr(query, "utm_") != NULL || 
                strstr(query, "fbclid=") != NULL || 
                strstr(query, "gclid=") != NULL) {
                *query = '\0';
            }
        }
    }
    
    // Ensure the URL doesn't end with a slash (for consistency)
    size_t len = strlen(normalized);
    if (len > 0 && normalized[len-1] == '/') {
        normalized[len-1] = '\0';
    }
    
    return normalized;
}

int main() {
    // Test array of URLs
    const char* test_urls[] = {
        "https://medium.com/@lpramithamj",
        "https://medium.com/@lpramithamj?source=search_post---------0",
        "https://medium.com/@lpramithamj/",
        "https://medium.com/@lpramithamj#content",
        "https://medium.com/@lpramithamj/article-title-123",
        NULL
    };
    
    printf("Testing URL normalization for Medium URLs:\n");
    printf("------------------------------------------\n");
    
    for (int i = 0; test_urls[i] != NULL; i++) {
        char* normalized = normalize_url(test_urls[i]);
        printf("Original: %s\n", test_urls[i]);
        printf("Normalized: %s\n\n", normalized);
    }

    // Test with NULL and empty URLs
    printf("Testing edge cases:\n");
    printf("------------------\n");
    
    char* result1 = normalize_url(NULL);
    printf("NULL URL normalized: '%s'\n", result1);
    
    char* result2 = normalize_url("");
    printf("Empty URL normalized: '%s'\n", result2);
    
    return 0;
}
