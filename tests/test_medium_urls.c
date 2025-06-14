#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_URL_LENGTH 512

// Forward declarations
static char* normalize_url(const char* url);
static int is_valid_crawl_url(const char* url, const char* base_domain);

// Function to normalize a URL (simplified version for testing)
static char* normalize_url(const char* url) {
    static __thread char normalized[MAX_URL_LENGTH * 2];
    
    // Always initialize to empty string first
    memset(normalized, 0, sizeof(normalized));
    
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
            
            // Further normalize Medium URLs to avoid duplicates
            // Handle URLs with @ symbol (profile URLs)
            char* at_symbol = strstr(normalized, "medium.com/@");
            if (at_symbol != NULL) {
                char* slash_after_username = strchr(at_symbol + 12, '/');
                // If it's a profile URL with path, but not an article
                if (slash_after_username != NULL && strstr(slash_after_username, "/p/") == NULL) {
                    // Keep only the profile part and remove additional paths
                    *slash_after_username = '\0';
                }
            }
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

// Simplified version for testing
static int is_valid_crawl_url(const char* url, const char* base_domain) {
    if (!url || strlen(url) == 0) {
        return 0;
    }
    
    // Check for medium.com URLs
    if (strstr(url, "medium.com") != NULL) {
        char* normalized_url = normalize_url(url);
        
        // Exclude specific Medium paths
        if (strstr(normalized_url, "medium.com/m/signin") != NULL || 
            strstr(normalized_url, "medium.com/m/signout") != NULL ||
            strstr(normalized_url, "medium.com/plans") != NULL ||
            strstr(normalized_url, "medium.com/m/callback") != NULL ||
            strstr(normalized_url, "medium.com/m/connect") != NULL ||
            strstr(normalized_url, "medium.com/m/login") != NULL ||
            strstr(normalized_url, "medium.com/m/account") != NULL ||
            strstr(normalized_url, "help.medium.com") != NULL ||
            strstr(normalized_url, "policy.medium.com") != NULL) {
            return 0;
        }
        
        return 1;
    }
    
    // For non-Medium URLs, allow only those from the same domain
    return (base_domain && strstr(url, base_domain) != NULL);
}

// Function to simulate crawling and test memory management
void simulate_crawler_processing() {
    char* urls[] = {
        "https://medium.com/@lpramithamj",
        "https://medium.com/@lpramithamj?source=search_post---------0",
        "https://medium.com/@lpramithamj/",
        "https://medium.com/@lpramithamj#content",
        "https://medium.com/@lpramithamj/article-title-123",
        "https://medium.com/m/signin?operation=register",
        "https://medium.com/plans?source=upgrade_membership",
        NULL
    };
    
    printf("Simulating crawler processing for Medium URLs:\n");
    printf("===============================================\n\n");
    
    for (int i = 0; urls[i] != NULL; i++) {
        const char* url = urls[i];
        char* normalized = normalize_url(url);
        int is_valid = is_valid_crawl_url(url, "medium.com");
        
        printf("URL: %s\n", url);
        printf("Normalized: %s\n", normalized);
        printf("Valid for crawling: %s\n", is_valid ? "YES" : "NO");
        
        // Simulate multiple normalizations of the same URL
        // This should not cause memory issues since we use static buffers
        char* normalized2 = normalize_url(url);
        char* normalized3 = normalize_url(url);
        
        printf("Normalized (2nd call): %s\n", normalized2);
        printf("Normalized addresses - 1st: %p, 2nd: %p, 3rd: %p\n", 
               (void*)normalized, (void*)normalized2, (void*)normalized3);
        
        // For thread-local static buffers, these should be the same address in the same thread
        if (normalized == normalized2 && normalized2 == normalized3) {
            printf("✓ Memory addresses match as expected (using thread-local static buffer)\n");
        } else {
            printf("✗ Memory addresses don't match! This might indicate a problem.\n");
        }
        
        printf("\n");
    }
}

int main() {
    // Seed the random number generator
    srand(time(NULL));
    
    printf("Test for Medium URL handling with fixes for double free issues\n\n");
    
    // Test URL normalization
    simulate_crawler_processing();
    
    printf("All tests completed.\n");
    return 0;
}
