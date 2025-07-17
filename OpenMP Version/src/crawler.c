#include "../include/crawler.h"
#include "../include/metrics.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <strings.h>
#include <curl/curl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>    // For directory operations
#include <time.h>      // For thread-specific random seeding
#include <math.h>      // For pow(), sqrt() in thread statistics
#include <limits.h>    // For INT_MAX constant

// On macOS we need special handling for OpenMP
#ifdef __APPLE__
  #include <TargetConditionals.h>
  #if TARGET_OS_MAC
    #include </opt/homebrew/opt/libomp/include/omp.h>
  #else
    #include <omp.h>
  #endif
#else
  #include <omp.h>    // OpenMP for parallel processing
#endif

// Define constants - these are moved to the top for global reference
#define MAX_URLS 1000
#define MAX_URL_LENGTH 512

// Forward declarations
static char* normalize_url(const char* url);
static char* extract_base_domain(const char* url);
static int has_visited(const char* url);
static void mark_visited(const char* url);
static void extract_links(const char* html, const char* base_url, char** urls, int* url_count, int max_urls);
static void show_crawling_progress(int thread_id, const char* message);
static void show_thread_distribution(int num_threads, int* thread_pages);

// Callback function for libcurl to write data to a file
// Structure to hold the downloaded data
struct MemoryStruct {
    char *memory;
    size_t size;
};

// Callback function for libcurl to write data to memory
static size_t write_data_callback(void *contents, size_t size, size_t nmemb, void *userp) {
    size_t realsize = size * nmemb;
    struct MemoryStruct *mem = (struct MemoryStruct *)userp;
    
    char *ptr = realloc(mem->memory, mem->size + realsize + 1);
    if(ptr == NULL) {
        fprintf(stderr, "Not enough memory (realloc returned NULL)\n");
        return 0;
    }
    
    mem->memory = ptr;
    memcpy(&(mem->memory[mem->size]), contents, realsize);
    mem->size += realsize;
    mem->memory[mem->size] = 0;
    
    return realsize;
}

// Write filtered content to a file
static size_t write_callback(void *contents, size_t size, size_t nmemb, void *userp) {
    FILE *file = (FILE *)userp;
    size_t written = fwrite(contents, size, nmemb, file);
    return written;
}

// Function to extract a filename from URL
static char* get_url_filename(const char* url) {
    // Start with a default name
    static char filename[256];
    
    // Try to get the last part of the URL after the last slash
    const char *last_slash = strrchr(url, '/');
    if (last_slash && strlen(last_slash) > 1) {
        // Remove query parameters
        char *query = strchr(last_slash + 1, '?');
        if (query) {
            int len = query - (last_slash + 1);
            if (len > 0 && len < 50) {
                strncpy(filename, last_slash + 1, len);
                filename[len] = '\0';
                return filename;
            }
        } else {
            // Use the last part of the URL
            if (strlen(last_slash + 1) > 0 && strlen(last_slash + 1) < 50) {
                strcpy(filename, last_slash + 1);
                return filename;
            }
        }
    }
    
    // Use a hash of the URL
    unsigned int hash = 0;
    for (int i = 0; url[i]; i++) {
        hash = 31 * hash + url[i];
    }
    sprintf(filename, "webpage_%u.txt", hash);
    return filename;
}

// Function to ensure the dataset directory exists
static void ensure_dataset_directory() {
    struct stat st = {0};
    if (stat("dataset", &st) == -1) {
        #if defined(_WIN32)
            mkdir("dataset");
        #else
            mkdir("dataset", 0700);
        #endif
    }
}

// Function to check if a string starts with another string
static int starts_with(const char* str, const char* prefix) {
    return strncasecmp(str, prefix, strlen(prefix)) == 0;
}

// Function to determine if text is likely machine-generated or useless
static int is_useful_content(const char* text, int length) {
    // Skip empty content
    if (length < 10) return 0;
    
    // Count certain characters that might indicate useful text
    int alpha_count = 0;
    int space_count = 0;
    int punct_count = 0;
    
    for (int i = 0; i < length && i < 200; i++) {
        if (isalpha(text[i])) alpha_count++;
        else if (isspace(text[i])) space_count++;
        else if (ispunct(text[i])) punct_count++;
    }
    
    // Heuristic: useful text usually has a good mix of letters, spaces, and some punctuation
    float alpha_ratio = (float)alpha_count / length;
    float space_ratio = (float)space_count / length;
    
    // Text should have a decent amount of alphabetic characters and spaces
    return (alpha_ratio > 0.4 && space_ratio > 0.05 && space_ratio < 0.3);
}

// A more robust HTML to text conversion with special handling for Medium articles
static void html_to_text(const char *html, FILE *output) {
    int in_tag = 0;
    int in_script = 0;
    int in_style = 0;
    int in_head = 0;
    int in_comment = 0;
    int space_needed = 0;
    int consecutive_spaces = 0;
    int article_start = 0;
    int article_end = 0;
    int content_written = 0;
    size_t html_len = strlen(html);
    
    // Buffer for collecting text from specific elements
    char text_buffer[10000] = {0};
    int buffer_pos = 0;
    int in_title = 0;
    int in_paragraph = 0;
    int in_heading = 0;
    
    // First, try to find the main article content for Medium pages
    const char* article_tag = NULL;
    if (strstr(html, "medium.com") != NULL) {
        // Look for article tag or main content section
        article_tag = strstr(html, "<article");
        
        if (!article_tag) {
            // Alternative: look for main section
            article_tag = strstr(html, "<section class=\"section-inner");
        }
        
        if (article_tag) {
            html = article_tag;
        }
    }
    
    // First, look for the body tag to skip header content if we didn't find article
    if (!article_tag) {
        const char *body_start = strstr(html, "<body");
        if (body_start) {
            html = body_start;
        }
    }
    
    for (size_t i = 0; html[i]; i++) {
        // Handle HTML comments
        if (i + 3 < html_len && !in_comment && !in_tag && strncmp(&html[i], "<!--", 4) == 0) {
            in_comment = 1;
            i += 3; // Skip past opening comment
            continue;
        } else if (in_comment && i + 2 < html_len && strncmp(&html[i], "-->", 3) == 0) {
            in_comment = 0;
            i += 2; // Skip past closing comment
            continue;
        }
        
        if (in_comment) {
            continue;
        }
        
        // Check for key HTML sections
        if (!in_tag && (i + 6 < html_len) && starts_with(&html[i], "<head>")) {
            in_head = 1;
            in_tag = 1;
            continue;
        }
        else if (in_head && (i + 7 < html_len) && starts_with(&html[i], "</head>")) {
            in_head = 0;
            in_tag = 1;
            i += 6;  // Skip to end of tag
            continue;
        }
        else if (!in_tag && (i + 8 < html_len) && starts_with(&html[i], "<script")) {
            in_script = 1;
            in_tag = 1;
        }
        else if (!in_tag && (i + 7 < html_len) && starts_with(&html[i], "<style")) {
            in_style = 1;
            in_tag = 1;
        }
        else if (in_script && (i + 9 < html_len) && starts_with(&html[i], "</script>")) {
            in_script = 0;
            i += 8;  // Skip to end of tag
            continue;
        }
        else if (in_style && (i + 8 < html_len) && starts_with(&html[i], "</style>")) {
            in_style = 0;
            i += 7;  // Skip to end of tag
            continue;
        }
        else if (!in_tag && (i + 7 < html_len) && starts_with(&html[i], "<title>")) {
            in_title = 1;
            buffer_pos = 0;
            i += 6;  // Skip past <title>
            continue;
        }
        else if (in_title && (i + 8 < html_len) && starts_with(&html[i], "</title>")) {
            in_title = 0;
            i += 7;  // Skip past </title>
            // Add title with emphasis
            if (buffer_pos > 0) {
                text_buffer[buffer_pos] = '\0';
                fprintf(output, "\n\n# %s\n\n", text_buffer);
                content_written = 1;
            }
            buffer_pos = 0;
            continue;
        }
        
        // Special handling for Medium blog articles
        else if (!in_tag && strstr(html, "medium.com") != NULL) {
            if ((i + 3 < html_len) && starts_with(&html[i], "<h1")) {
                in_heading = 1;
                buffer_pos = 0;
                i += 2;  // Skip past <h1
                in_tag = 1;
                continue;
            }
            else if (in_heading && (i + 5 < html_len) && starts_with(&html[i], "</h1>")) {
                in_heading = 0;
                i += 4;  // Skip past </h1>
                
                if (buffer_pos > 0) {
                    text_buffer[buffer_pos] = '\0';
                    fprintf(output, "\n\n# %s\n\n", text_buffer);
                    content_written = 1;
                }
                buffer_pos = 0;
                continue;
            }
            else if ((i + 3 < html_len) && starts_with(&html[i], "<h2")) {
                in_heading = 1;
                buffer_pos = 0;
                i += 2;  // Skip past <h2
                in_tag = 1;
                continue;
            }
            else if (in_heading && (i + 5 < html_len) && starts_with(&html[i], "</h2>")) {
                in_heading = 0;
                i += 4;  // Skip past </h2>
                
                if (buffer_pos > 0) {
                    text_buffer[buffer_pos] = '\0';
                    fprintf(output, "\n\n## %s\n\n", text_buffer);
                    content_written = 1;
                }
                buffer_pos = 0;
                continue;
            }
            else if ((i + 3 < html_len) && starts_with(&html[i], "<p>")) {
                in_paragraph = 1;
                buffer_pos = 0;
                i += 2;  // Skip past <p>
                continue;
            }
            else if (in_paragraph && (i + 4 < html_len) && starts_with(&html[i], "</p>")) {
                in_paragraph = 0;
                i += 3;  // Skip past </p>
                
                if (buffer_pos > 0) {
                    text_buffer[buffer_pos] = '\0';
                    if (is_useful_content(text_buffer, buffer_pos)) {
                        fprintf(output, "%s\n\n", text_buffer);
                        content_written = 1;
                    }
                }
                buffer_pos = 0;
                continue;
            }
        }
        
        // Skip content in head, script, and style sections
        if (in_head || in_script || in_style) {
            if (html[i] == '<') {
                in_tag = 1;
            } else if (in_tag && html[i] == '>') {
                in_tag = 0;
            }
            continue;
        }
        
        // Handle tags
        if (html[i] == '<') {
            in_tag = 1;
            
            // Check for specific tags that should add paragraph breaks
            if ((i + 4 < html_len) && (starts_with(&html[i], "<p>") || 
                                      starts_with(&html[i], "<br") ||
                                      starts_with(&html[i], "<li") ||
                                      starts_with(&html[i], "<h"))) {
                if (!in_title && !in_heading && !in_paragraph) {
                    fprintf(output, "\n\n");
                }
                consecutive_spaces = 0;
                space_needed = 0;
            }
            continue;
        }
        
        if (in_tag) {
            if (html[i] == '>') {
                in_tag = 0;
                space_needed = 1;
            }
            continue;
        }
        
        // Handle content in special elements that we're collecting in buffer
        if (in_title || in_heading || in_paragraph) {
            if (buffer_pos < sizeof(text_buffer) - 1) {
                // Convert common HTML entities within special elements
                if (html[i] == '&') {
                    if ((i + 5 < html_len) && strncmp(&html[i], "&amp;", 5) == 0) {
                        text_buffer[buffer_pos++] = '&';
                        i += 4;
                    } else if ((i + 4 < html_len) && strncmp(&html[i], "&lt;", 4) == 0) {
                        text_buffer[buffer_pos++] = '<';
                        i += 3;
                    } else if ((i + 4 < html_len) && strncmp(&html[i], "&gt;", 4) == 0) {
                        text_buffer[buffer_pos++] = '>';
                        i += 3;
                    } else if ((i + 6 < html_len) && strncmp(&html[i], "&quot;", 6) == 0) {
                        text_buffer[buffer_pos++] = '"';
                        i += 5;
                    } else if ((i + 6 < html_len) && strncmp(&html[i], "&nbsp;", 6) == 0) {
                        text_buffer[buffer_pos++] = ' ';
                        i += 5;
                    } else if ((i + 6 < html_len) && strncmp(&html[i], "&#039;", 6) == 0) {
                        text_buffer[buffer_pos++] = '\'';
                        i += 5;
                    } else {
                        // For other HTML entities, try to skip them
                        size_t j = i;
                        while (html[j] && html[j] != ';' && j - i < 10) j++;
                        if (html[j] == ';') {
                            i = j;
                        } else {
                            text_buffer[buffer_pos++] = html[i];
                        }
                    }
                } else if (isspace((unsigned char)html[i])) {
                    // Handle spaces in buffer
                    if (buffer_pos > 0 && !isspace((unsigned char)text_buffer[buffer_pos-1])) {
                        text_buffer[buffer_pos++] = ' ';
                    }
                } else {
                    text_buffer[buffer_pos++] = html[i];
                }
            }
            continue;
        }
        
        // Handle regular text content (outside special elements)
        if (isspace((unsigned char)html[i])) {
            if (consecutive_spaces == 0) {
                fputc(' ', output);
                consecutive_spaces = 1;
                content_written = 1;
            }
        } else {
            // Convert common HTML entities
            if (html[i] == '&') {
                if ((i + 5 < html_len) && strncmp(&html[i], "&amp;", 5) == 0) {
                    fputc('&', output);
                    i += 4;
                } else if ((i + 4 < html_len) && strncmp(&html[i], "&lt;", 4) == 0) {
                    fputc('<', output);
                    i += 3;
                } else if ((i + 4 < html_len) && strncmp(&html[i], "&gt;", 4) == 0) {
                    fputc('>', output);
                    i += 3;
                } else if ((i + 6 < html_len) && strncmp(&html[i], "&quot;", 6) == 0) {
                    fputc('"', output);
                    i += 5;
                } else if ((i + 6 < html_len) && strncmp(&html[i], "&nbsp;", 6) == 0) {
                    fputc(' ', output);
                    i += 5;
                } else if ((i + 6 < html_len) && strncmp(&html[i], "&#039;", 6) == 0) {
                    fputc('\'', output);
                    i += 5;
                } else {
                    // For other HTML entities, try to skip them
                    size_t j = i;
                    while (html[j] && html[j] != ';' && j - i < 10) j++;
                    if (html[j] == ';') {
                        i = j;
                    } else {
                        fputc(html[i], output);
                    }
                }
            } else {
                // Regular character
                fputc(html[i], output);
            }
            consecutive_spaces = 0;
            content_written = 1;
        }
    }
    
    // Add a note if no content was found
    if (!content_written) {
        fprintf(output, "No readable content could be extracted from this page.");
    }
}

#define MAX_URLS 1000
#define MAX_URL_LENGTH 512

// Store already visited URLs to avoid duplicates
static char visited_urls[MAX_URLS][MAX_URL_LENGTH];
static int visited_count = 0;
static omp_lock_t visited_lock; // Lock for thread-safe access to visited URLs

// Function to check if a URL has been visited
static int has_visited(const char* url) {
    if (!url) return 1;  // Treat NULL URLs as already visited
    
    // First normalize the URL for consistent comparison
    char* normalized = normalize_url(url);
    if (!normalized || normalized[0] == '\0') return 1; // Treat empty URLs as visited
    
    int result = 0;
    
    // Add lock to protect the read operation
    omp_set_lock(&visited_lock);
    
    // Safe comparison with visited URLs
    for (int i = 0; i < visited_count; i++) {
        if (visited_urls[i][0] != '\0' && strcmp(visited_urls[i], normalized) == 0) {
            result = 1;
            break;
        }
    }
    
    omp_unset_lock(&visited_lock);
    
    return result;
}

// Function to mark a URL as visited
static void mark_visited(const char* url) {
    if (!url) return;  // Don't try to mark NULL URLs
    
    // First normalize the URL for consistent storage
    char* normalized = normalize_url(url);
    if (!normalized || normalized[0] == '\0') return; // Don't mark empty URLs
    
    omp_set_lock(&visited_lock);
    
    // Check if already in our visited list
    for (int i = 0; i < visited_count; i++) {
        if (visited_urls[i][0] != '\0' && strcmp(visited_urls[i], normalized) == 0) {
            omp_unset_lock(&visited_lock);
            return; // Already marked
        }
    }
    
    // Add to visited list if space is available
    if (visited_count < MAX_URLS) {
        strncpy(visited_urls[visited_count], normalized, MAX_URL_LENGTH - 1);
        visited_urls[visited_count][MAX_URL_LENGTH - 1] = '\0';
        visited_count++;
    }
    
    omp_unset_lock(&visited_lock);
}

// Function to extract the base domain from a URL
static char* extract_base_domain(const char* url) {
    // Thread-local static buffer to avoid issues with multiple calls
    static __thread char domain[MAX_URL_LENGTH];
    
    if (!url || strlen(url) == 0) {
        domain[0] = '\0';
        return domain;
    }
    
    // Initialize the domain
    strncpy(domain, url, MAX_URL_LENGTH - 1);
    domain[MAX_URL_LENGTH - 1] = '\0';
    
    // Find the protocol part
    char* protocol = strstr(domain, "://");
    if (!protocol) return domain;
    
    // Find the domain part (after protocol)
    char* domain_start = protocol + 3;
    
    // Find the end of domain (first slash after protocol)
    char* path = strchr(domain_start, '/');
    if (path) *path = '\0';
    
    return domain;
}

// Function to normalize a URL (remove tracking params, fragments, etc.)
static char* normalize_url(const char* url) {
    // Use thread-local static buffer to avoid issues with multiple calls
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
        } else {
            // For other sites, try to keep important query params but remove common tracking ones
            // This is just a simple example - could be extended with more parameters
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

// Process a URL found in HTML and add it to the list if valid
// Enhanced with intelligent URL prioritization and validation
static void process_extracted_url(const char* url_text, int url_len, const char* base_url, const char* base_domain, 
                                char** urls, int* url_count, int max_urls) {
    if (url_len <= 0 || url_len >= MAX_URL_LENGTH || *url_count >= max_urls) 
        return;
        
    // Allocate and copy the URL
    char* new_url = malloc(MAX_URL_LENGTH);
    if (!new_url) return;
    
    strncpy(new_url, url_text, url_len);
    new_url[url_len] = '\0';
    
    // Skip irrelevant URLs and common non-content paths with expanded filters
    if (strncmp(new_url, "javascript:", 11) == 0 ||
        strncmp(new_url, "mailto:", 7) == 0 ||
        strncmp(new_url, "tel:", 4) == 0 ||
        strncmp(new_url, "data:", 5) == 0 ||
        strncmp(new_url, "file:", 5) == 0 ||
        strncmp(new_url, "ftp:", 4) == 0 ||
        strncmp(new_url, "#", 1) == 0 ||         // Skip page anchors
        strstr(new_url, "/cdn-cgi/") != NULL ||  // Cloudflare internal paths
        strstr(new_url, "/wp-json/") != NULL ||  // WordPress API paths
        strstr(new_url, "/wp-admin/") != NULL || // WordPress admin paths
        strstr(new_url, "/wp-content/") != NULL || // WordPress content paths (usually images)
        strstr(new_url, "/api/") != NULL ||      // API endpoints
        strstr(new_url, ".jpg") != NULL ||       // Common media files
        strstr(new_url, ".jpeg") != NULL ||
        strstr(new_url, ".png") != NULL ||
        strstr(new_url, ".gif") != NULL ||
        strstr(new_url, ".svg") != NULL ||
        strstr(new_url, ".ico") != NULL ||
        strstr(new_url, ".js") != NULL ||        // JavaScript files
        strstr(new_url, ".css") != NULL) {       // CSS files
        free(new_url);
        return;
    }
    
    // Special handling for Medium URLs with improved detection
    int is_medium_url = strstr(base_url, "medium.com") != NULL;
    int is_medium_profile = is_medium_url && strstr(base_url, "medium.com/@") != NULL;
    
    if (is_medium_url) {
        // Check for special Medium profile cases
        if (new_url[0] == '@') {
            // Convert @username to full URL
            char* absolute_url = malloc(MAX_URL_LENGTH * 2);
            if (!absolute_url) {
                free(new_url);
                return;
            }
            sprintf(absolute_url, "https://medium.com/%s", new_url);
            free(new_url);
            new_url = absolute_url;
        }
        
        // Skip Medium internal/utility pages that don't contain content
        if (strstr(new_url, "medium.com/m/signin") != NULL ||
            strstr(new_url, "medium.com/m/account") != NULL ||
            strstr(new_url, "medium.com/plans") != NULL ||
            strstr(new_url, "medium.com/about") != NULL ||
            strstr(new_url, "medium.com/creators") != NULL ||
            strstr(new_url, "medium.com/privacy") != NULL ||
            strstr(new_url, "medium.com/membership") != NULL) {
            free(new_url);
            return;
        }
    }
    
    // Handle relative URLs with improved path handling
    if (strncmp(new_url, "http", 4) != 0) {
        // Convert relative URL to absolute URL
        char* absolute_url = malloc(MAX_URL_LENGTH * 2);
        if (!absolute_url) {
            free(new_url);
            return;
        }
        
        if (new_url[0] == '/') {
            if (new_url[1] == '/') {
                // Protocol-relative URL (//example.com/path)
                // Extract protocol from base_url
                const char* protocol_end = strstr(base_url, "://");
                if (protocol_end) {
                    int protocol_len = protocol_end - base_url + 1; // Include the colon
                    strncpy(absolute_url, base_url, protocol_len);
                    absolute_url[protocol_len] = '\0';
                    strcat(absolute_url, new_url + 2); // Skip the //
                } else {
                    // Default to https if we can't determine
                    sprintf(absolute_url, "https:%s", new_url);
                }
            } else {
                // URL starts with /, so append to domain
                sprintf(absolute_url, "%s%s", base_domain, new_url);
            }
        } else {
            // URL is relative to current page
            strcpy(absolute_url, base_url);
            
            // Remove everything after the last slash in base_url
            char* last_slash = strrchr(absolute_url, '/');
            if (last_slash && last_slash != absolute_url + strlen(absolute_url) - 1) {
                *(last_slash + 1) = '\0';
            } else if (!last_slash) {
                // If no slash in the URL after domain, add one
                strcat(absolute_url, "/");
            }
            
            strcat(absolute_url, new_url);
        }
        
        free(new_url);
        new_url = absolute_url;
    }
    
    // Normalize the URL to avoid duplicates
    char* normalized_url = normalize_url(new_url);
    
    // Only duplicate valid normalized URLs
    char* final_url = NULL;
    if (normalized_url && normalized_url[0] != '\0') {
        final_url = strdup(normalized_url);
    }
    
    // Free the original URL - only if it's not NULL
    if (new_url) {
        free(new_url);
        new_url = NULL;
    }
    
    // Early return if no valid URL
    if (!final_url) return;
    
    // Check if the URL is valid and not already visited/queued
    int is_valid = 0;
    int is_duplicate = 0;
    int url_priority = 1;  // Default priority (higher = more important)
    
    // Check if it's already visited
    if (has_visited(final_url)) {
        is_duplicate = 1;
    } else {
        // Also check if it's already in our current extraction list
        for (int i = 0; i < *url_count; i++) {
            if (urls[i] && strcmp(urls[i], final_url) == 0) {
                is_duplicate = 1;
                break;
            }
        }
    }
    
    if (!is_duplicate) {
        // For Medium sites, use special handling
        if (strstr(base_url, "medium.com") != NULL) {
            if (strstr(final_url, "medium.com") != NULL) {
                // For Medium we allow all URLs within medium.com
                is_valid = 1;
                
                // Give higher priority to story URLs over profile pages
                if (strstr(final_url, "/tagged/") != NULL) {
                    url_priority = 3;  // Tag pages are valuable for crawling
                } else if (strstr(final_url, "/@") != NULL && strstr(final_url, "/followers") == NULL) {
                    // Profile URLs are high priority, except followers pages
                    url_priority = 4;
                } else if (strstr(final_url, "/p/") != NULL) {
                    // Story URLs are highest priority
                    url_priority = 5;
                }
            }
        } else if (base_domain && strstr(final_url, base_domain) != NULL) {
            // For other sites, use stricter domain checking but with enhanced prioritization
            is_valid = 1;
            
            // Prioritize content pages over navigational ones
            // Check for common content patterns
            if (strstr(final_url, "/article/") != NULL || 
                strstr(final_url, "/post/") != NULL || 
                strstr(final_url, "/blog/") != NULL ||
                strstr(final_url, "/story/") != NULL) {
                url_priority = 4;  // Content pages get high priority
            } else if (strstr(final_url, "/category/") != NULL ||
                      strstr(final_url, "/tag/") != NULL ||
                      strstr(final_url, "/topics/") != NULL) {
                url_priority = 3;  // Category/tag pages get medium priority
            } else if (strstr(final_url, "/page/") != NULL ||
                      strstr(final_url, "?page=") != NULL) {
                url_priority = 2;  // Pagination gets lower priority
            }
        }
    }
    
    // Add URL to the list or free it with enhanced priority-based positioning
    if (is_valid && !is_duplicate && *url_count < max_urls) {
        // Calculate URL diversity score to help thread distribution
        // URLs with different patterns will be distributed better across threads
        int url_diversity = 0;
        if (strstr(final_url, "/tag/") || strstr(final_url, "/topic/") || strstr(final_url, "/category/"))
            url_diversity = 2;  // These paths lead to diverse content
        else if (strstr(final_url, "/@") || strstr(final_url, "/author/"))
            url_diversity = 3;  // Author/profile pages have unique content
        
        // Combine priority and diversity for better insertion position
        int combined_score = url_priority + url_diversity;
        
        // If this is a high-priority URL, try to insert it strategically in the list
        // This improves thread workload distribution while maintaining priority
        if (combined_score > 3 && *url_count > 0) {
            // Use different insertion strategies based on combined score
            int insert_pos;
            
            if (combined_score >= 7) {
                insert_pos = 0;  // Top priority: place at beginning
            } else if (combined_score >= 5) {
                insert_pos = *url_count / 4;  // High priority: first quarter
            } else {
                insert_pos = *url_count / 2;  // Medium priority: middle
            }
            
            // Only move if we're not inserting at the end
            if (insert_pos < *url_count) {
                // Make room by shifting elements
                for (int i = *url_count; i > insert_pos; i--) {
                    urls[i] = urls[i-1];
                }
                
                // Insert at the calculated position
                urls[insert_pos] = final_url;
                (*url_count)++;
                return;
            }
        }
        
        // Default case: add to the end of the list
        urls[*url_count] = final_url;
        (*url_count)++;
    } else {
        free(final_url);
    }
}

// Function to extract links from HTML content with improved load balancing
static void extract_links(const char* html, const char* base_url, char** urls, int* url_count, int max_urls) {
    if (!html || !base_url || !urls || !url_count) return;
    
    // Skip extraction if we already have max urls
    if (*url_count >= max_urls) return;
    
    // Check if it's a Medium profile page (for special handling)
    int is_medium_profile = strstr(base_url, "medium.com/@") != NULL;
    
    // Normalize the base URL for generating absolute URLs
    char* base_domain = extract_base_domain(base_url);
    
    size_t html_len = strlen(html);
    int num_threads = omp_get_max_threads();
    omp_lock_t url_lock;
    omp_init_lock(&url_lock);
    
    // Adaptive thread count based on HTML size and content type
    size_t optimal_chunk_size = 50000; // Target ~50KB of HTML per thread for efficient processing
    
    // Calculate optimal number of threads based on HTML size
    int optimal_threads = (html_len / optimal_chunk_size) + 1;
    if (optimal_threads > num_threads) optimal_threads = num_threads;
    if (optimal_threads < 1) optimal_threads = 1;

    // Special handling for different content types
    if (is_medium_profile) {
        // Medium profiles need special handling with fewer threads due to their structure
        optimal_threads = (optimal_threads > 2) ? 2 : optimal_threads;
        printf("  Medium profile detected (%zu bytes), using %d threads for extraction\n", html_len, optimal_threads);
    } else if (strstr(base_url, "medium.com") != NULL) {
        // Other Medium pages - still need special handling but can use more threads
        optimal_threads = (optimal_threads > num_threads/2) ? num_threads/2 : optimal_threads;
        if (optimal_threads < 1) optimal_threads = 1;
        printf("  Medium page detected (%zu bytes), using %d threads for extraction\n", html_len, optimal_threads);
    } else {
        printf("  HTML size: %zu bytes, using %d threads for link extraction\n", html_len, optimal_threads);
    }
    
    // Use the optimal thread count for this extraction
    num_threads = optimal_threads;
    
    // Track URLs found per thread for better diagnostics
    int* urls_per_thread = calloc(num_threads, sizeof(int));
    if (!urls_per_thread) {
        fprintf(stderr, "Failed to allocate memory for URL distribution tracking\n");
        urls_per_thread = NULL;
    }
    
    // Pre-scan HTML to find approximate link count for better work distribution
    int estimated_link_count = 0;
    const char* scan_ptr = html;
    while ((scan_ptr = strstr(scan_ptr, "href=")) != NULL) {
        estimated_link_count++;
        scan_ptr += 5; // Move past 'href='
    }
    printf("  Estimated link count: %d\n", estimated_link_count);
    
    // Track time spent in extraction for performance tuning
    double start_time = omp_get_wtime();
    
    // Improved workload distribution with dynamic scheduling and better chunk sizing
    #pragma omp parallel num_threads(num_threads) shared(html, base_url, base_domain, urls, url_count, url_lock, urls_per_thread)
    {
        // Each thread processes a section of HTML with better boundary detection
        int thread_id = omp_get_thread_num();
        
        // Calculate chunk size based on estimated links for more even distribution
        size_t chunk_size = html_len / num_threads;
        size_t start_pos = thread_id * chunk_size;
        size_t end_pos = (thread_id == num_threads - 1) ? html_len : (thread_id + 1) * chunk_size;
        
        // For all threads except first, find a better starting point:
        // We look for an opening tag to get a cleaner HTML fragment
        if (thread_id > 0) {
            // First move backward a bit to find a good tag boundary if possible
            size_t look_back = start_pos > 500 ? 500 : start_pos;
            size_t potential_start = start_pos - look_back;
            
            // Find the first '<' after potential_start
            const char* tag_start = strchr(html + potential_start, '<');
            if (tag_start && tag_start < html + start_pos + 200) {
                // Use this clean boundary if it's reasonably close
                start_pos = tag_start - html;
            } else {
                // If we can't find a good boundary looking back, look forward
                while (start_pos < end_pos && html[start_pos] != '<')
                    start_pos++;
            }
        }
        
        // Thread-local buffer for URLs before adding to shared list
        char* local_urls[100];
        int local_count = 0;
        
        // Search for URLs in this thread's chunk
        const char* ptr = html + start_pos;
        const char* end_ptr = html + end_pos;
        
        // For Medium profiles, look for specific link patterns
        if (is_medium_profile) {
            // Medium profiles have links in different formats
            const char* article_patterns[] = {
                "href=\"/", 
                "href=\"https://medium.com/",
                "href=\"@",
                "href=\"https://",
                "data-action-value=\"https://medium.com/"
            };
            
            // Process each pattern in parallel for medium profiles
            #pragma omp for schedule(dynamic, 1)
            for (int i = 0; i < sizeof(article_patterns)/sizeof(article_patterns[0]); i++) {
                const char* search_ptr = ptr;
                while (search_ptr < end_ptr && local_count < 100) {
                    const char* pattern_start = strstr(search_ptr, article_patterns[i]);
                    if (!pattern_start || pattern_start >= end_ptr) break;
                    
                    // Find the starting position after the pattern
                    const char* url_start = pattern_start + strlen(article_patterns[i]);
                    if (strcmp(article_patterns[i], "href=\"/") == 0) {
                        // Special case for relative URLs
                        char rel_url[MAX_URL_LENGTH] = "https://medium.com";
                        strncat(rel_url, url_start - 1, MAX_URL_LENGTH - strlen(rel_url) - 1);
                        const char* quote_end = strchr(rel_url, '"');
                        if (quote_end) {
                            ((char*)quote_end)[0] = '\0'; // Remove end quote
                            
                            // Process the relative URL if it's valid
                            if (strlen(rel_url) > strlen("https://medium.com")) {
                                process_extracted_url(rel_url, strlen(rel_url), base_url, base_domain, 
                                                     local_urls, &local_count, 100);
                                if (urls_per_thread) urls_per_thread[thread_id]++;
                            }
                        }
                    } else if (strncmp(article_patterns[i], "href=\"@", 7) == 0) {
                        // Handle username references
                        char username_url[MAX_URL_LENGTH] = "https://medium.com/";
                        strncat(username_url, url_start - 1, MAX_URL_LENGTH - strlen(username_url) - 1);
                        const char* quote_end = strchr(username_url, '"');
                        if (quote_end) {
                            ((char*)quote_end)[0] = '\0'; // Remove end quote
                            process_extracted_url(username_url, strlen(username_url), base_url, base_domain, 
                                                local_urls, &local_count, 100);
                            if (urls_per_thread) urls_per_thread[thread_id]++;
                        }
                    } else {
                        // Regular URL handling
                        const char* url_end = strchr(url_start, '"');
                        if (url_end && url_end < end_ptr) {
                            int url_len = url_end - url_start;
                            if (url_len > 0 && url_len < MAX_URL_LENGTH) {
                                process_extracted_url(url_start, url_len, base_url, base_domain, 
                                                    local_urls, &local_count, 100);
                                if (urls_per_thread) urls_per_thread[thread_id]++;
                            }
                        }
                    }
                    
                    // Move past this URL for next iteration
                    search_ptr = url_start + 1;
                }
            }
        } else {
            // Normal (non-Medium) page link extraction
            // Find <a href="..."> links using plain string search for faster processing
            while (ptr < end_ptr && local_count < 100) {
                // Look for href attribute
                const char* href_double = strstr(ptr, "href=\"");
                const char* href_single = strstr(ptr, "href='");
                
                // Find the closer match
                const char* href_start = NULL;
                const char* href_end = NULL;
                
                // Choose the closest href attribute
                if (href_double && href_single) {
                    if (href_double < href_single) {
                        href_start = href_double + 6;
                        href_end = strchr(href_start, '"');
                    } else {
                        href_start = href_single + 6;
                        href_end = strchr(href_start, '\'');
                    }
                } else if (href_double) {
                    href_start = href_double + 6;
                    href_end = strchr(href_start, '"');
                } else if (href_single) {
                    href_start = href_single + 6;
                    href_end = strchr(href_start, '\'');
                }
                
                // If we found both start and end of the URL and it's within our chunk
                if (href_start && href_end && href_end < end_ptr) {
                    int url_len = href_end - href_start;
                    if (url_len > 0 && url_len < MAX_URL_LENGTH) {
                        process_extracted_url(href_start, url_len, base_url, base_domain, 
                                           local_urls, &local_count, 100);
                        if (urls_per_thread) urls_per_thread[thread_id]++;
                    }
                    ptr = href_end + 1;  // Move past this URL
                } else {
                    // If we found the start but not the end within this chunk
                    if (href_start && !href_end) {
                        ptr = end_ptr;  // End processing for this thread
                    } else {
                        ptr++;  // Move forward one character
                    }
                }
            }
        }
        
        // After processing, merge thread-local URLs into the global list with lock protection
        if (local_count > 0) {
            omp_set_lock(&url_lock);
            for (int i = 0; i < local_count && *url_count < max_urls; i++) {
                if (local_urls[i]) {
                    urls[*url_count] = local_urls[i];
                    (*url_count)++;
                    local_urls[i] = NULL; // Prevent double-free
                }
            }
            omp_unset_lock(&url_lock);
            
            // Free any remaining URLs that didn't make it to the global list
            for (int i = 0; i < local_count; i++) {
                if (local_urls[i]) {
                    free(local_urls[i]);
                    local_urls[i] = NULL;
                }
            }
        }
    }        // Print detailed URL distribution for diagnostics
    if (urls_per_thread) {
        double end_time = omp_get_wtime();
        double extraction_time = end_time - start_time;
        
        #pragma omp critical(output)
        {
            printf("URL distribution in extraction (by thread): ");
            int total_extracted = 0;
            int min_urls = INT_MAX;
            int max_urls = 0;
            int empty_threads = 0;
            
            for (int i = 0; i < num_threads; i++) {
                total_extracted += urls_per_thread[i];
                printf("%d ", urls_per_thread[i]);
                
                if (urls_per_thread[i] > max_urls) max_urls = urls_per_thread[i];
                if (urls_per_thread[i] < min_urls) min_urls = urls_per_thread[i];
                if (urls_per_thread[i] == 0) empty_threads++;
            }
            
            // Calculate load imbalance metrics
            double avg_urls = (double)total_extracted / num_threads;
            double imbalance = (max_urls > 0) ? ((double)max_urls - avg_urls) / max_urls : 0;
            
            printf("\n  Stats: Total: %d URLs | Time: %.3f sec | Threads: %d | ", 
                   total_extracted, extraction_time, num_threads);
            printf("Min/Avg/Max: %d/%.1f/%d | Imbalance: %.1f%% | Empty threads: %d\n",
                   min_urls, avg_urls, max_urls, imbalance * 100, empty_threads);
        }
        free(urls_per_thread);
    }
    
    // Clean up
    omp_destroy_lock(&url_lock);
}

// Function to extract page title from HTML
static char* extract_title(const char* html) {
    static char title[256];
    memset(title, 0, sizeof(title));
    
    // Find start of title tag
    const char* title_start = strstr(html, "<title");
    if (!title_start) return title;
    
    // Find end of title tag opening
    title_start = strchr(title_start, '>');
    if (!title_start) return title;
    title_start++; // Move past '>'
    
    // Find end of title content
    const char* title_end = strstr(title_start, "</title>");
    if (!title_end) return title;
    
    // Calculate length and copy title
    size_t title_len = title_end - title_start;
    if (title_len > 0 && title_len < sizeof(title) - 1) {
        strncpy(title, title_start, title_len);
        title[title_len] = '\0';
        
        // Convert HTML entities in title
        char* amp = strstr(title, "&amp;");
        while (amp) {
            *amp = '&';
            memmove(amp + 1, amp + 5, strlen(amp + 5) + 1);
            amp = strstr(amp + 1, "&amp;");
        }
        
        // Do the same for other common entities
        char* lt = strstr(title, "&lt;");
        while (lt) {
            *lt = '<';
            memmove(lt + 1, lt + 4, strlen(lt + 4) + 1);
            lt = strstr(lt + 1, "&lt;");
        }
        
        char* gt = strstr(title, "&gt;");
        while (gt) {
            *gt = '>';
            memmove(gt + 1, gt + 4, strlen(gt + 4) + 1);
            gt = strstr(gt + 1, "&gt;");
        }
    }
    
    return title;
}

// Function to get a better filename for medium URLs
static char* get_medium_filename(const char* url, const char* html) {
    static char filename[256];
    
    // Extract title from HTML if possible
    char* title = extract_title(html);
    if (strlen(title) > 0) {
        // Convert title to a valid filename
        char safe_title[256];
        int j = 0;
        
        for (int i = 0; i < strlen(title) && j < sizeof(safe_title) - 5; i++) {
            char c = title[i];
            if (isalnum(c) || c == ' ' || c == '-' || c == '_') {
                safe_title[j++] = (c == ' ') ? '_' : tolower(c);
            }
        }
        safe_title[j] = '\0';
        
        // Make sure we have something usable
        if (strlen(safe_title) > 0) {
            snprintf(filename, sizeof(filename), "medium_%s.txt", safe_title);
            return filename;
        }
    }
    
    // Fallback: use username for profiles
    if (strstr(url, "medium.com/@") != NULL) {
        const char* username = strstr(url, "@") + 1;
        char safe_username[100] = {0};
        
        // Copy until end of username (next slash or end of string)
        int i;
        for (i = 0; username[i] && username[i] != '/' && username[i] != '?' && i < 99; i++) {
            safe_username[i] = username[i];
        }
        safe_username[i] = '\0';
        
        if (strlen(safe_username) > 0) {
            snprintf(filename, sizeof(filename), "medium_profile_%s.txt", safe_username);
            return filename;
        }
    }
    
    // Ultimate fallback: use default URL hash
    return get_url_filename(url);
}

// Function to determine content type from URL and headers
static int is_html_content(const char* url, const char* content_type) {
    // Check URL extension first
    const char* ext = strrchr(url, '.');
    if (ext) {
        if (strcasecmp(ext, ".jpg") == 0 || strcasecmp(ext, ".jpeg") == 0 || 
            strcasecmp(ext, ".png") == 0 || strcasecmp(ext, ".gif") == 0 || 
            strcasecmp(ext, ".css") == 0 || strcasecmp(ext, ".js") == 0 ||
            strcasecmp(ext, ".pdf") == 0) {
            return 0;
        }
    }
    
    // If we have content type, check it
    if (content_type) {
        if (strstr(content_type, "text/html") || strstr(content_type, "application/xhtml+xml")) {
            return 1;
        }
        if (strstr(content_type, "image/") || strstr(content_type, "application/pdf") ||
            strstr(content_type, "application/javascript") || strstr(content_type, "text/css")) {
            return 0;
        }
    }
    
    // Default to treating it as HTML
    return 1;
}

// Function to download a URL and save it to the dataset directory
char* download_url(const char* url) {
    CURL *curl;
    CURLcode res;
    FILE *file;
    static char filepath[512];
    char* filename;
    struct MemoryStruct chunk;
    char content_type[256] = {0};
    
    // Initialize memory chunk
    chunk.memory = malloc(1);
    chunk.size = 0;
    
    curl = curl_easy_init();
    if (!curl) {
        fprintf(stderr, "Failed to initialize curl\n");
        free(chunk.memory);
        return NULL;
    }
    
    // Set up curl options to download to memory first
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&chunk);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "Mozilla/5.0 (compatible; SearchEngine/1.0)");
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L); // For simplicity
    
    // Add headers to mimic a browser request
    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Accept: text/html,application/xhtml+xml,application/xml");
    headers = curl_slist_append(headers, "Accept-Language: en-US,en;q=0.9");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    
    // Capture content-type header
    curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, write_data_callback);
    curl_easy_setopt(curl, CURLOPT_HEADERDATA, &content_type);
    
    // Perform the request
    printf("Downloading %s...\n", url);
    res = curl_easy_perform(curl);
    
    // Get content type from CURL
    char *ct = NULL;
    curl_easy_getinfo(curl, CURLINFO_CONTENT_TYPE, &ct);
    if (ct) {
        strncpy(content_type, ct, sizeof(content_type) - 1);
    }
    
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    
    if (res != CURLE_OK) {
        fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
        free(chunk.memory);
        return NULL;
    }
    
    // Check if content is HTML and has enough content
    if (!is_html_content(url, content_type) || chunk.size < 100) {
        printf("Skipping non-HTML content or too small content (size: %zu, type: %s)\n", 
               chunk.size, content_type[0] ? content_type : "unknown");
        free(chunk.memory);
        return NULL;
    }
    
    // Ensure dataset directory exists
    ensure_dataset_directory();
    
    // Create a filename based on the content
    if (strstr(url, "medium.com") != NULL) {
        filename = get_medium_filename(url, chunk.memory);
    } else {
        filename = get_url_filename(url);
    }
    
    snprintf(filepath, sizeof(filepath), "dataset/%s", filename);
    
    // Open file for writing
    file = fopen(filepath, "wb");
    if (!file) {
        fprintf(stderr, "Failed to open file for writing: %s\n", filepath);
        free(chunk.memory);
        return NULL;
    }
    
    // Write URL at the top of the file for reference
    fprintf(file, "Source URL: %s\n\n", url);
    
    // Convert HTML to text and save to file
    printf("Processing HTML content (%zu bytes)...\n", chunk.size);
    html_to_text(chunk.memory, file);
    
    // Clean up
    fclose(file);
    free(chunk.memory);
    
    printf("Downloaded and processed to %s\n", filepath);
    return filepath;
}

// Function to check if a URL is valid for crawling
static int is_valid_crawl_url(const char* url, const char* base_domain) {
    // Skip empty URLs
    if (!url || strlen(url) == 0) {
        return 0;
    }
    
    // Must be HTTP or HTTPS
    if (strncmp(url, "http://", 7) != 0 && strncmp(url, "https://", 8) != 0) {
        return 0;
    }
    
    // Skip common file types that are not useful for text search
    const char* file_extensions[] = {
        ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".ico", ".tiff", 
        ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx",
        ".zip", ".rar", ".tar", ".gz", ".mp3", ".mp4", ".avi", ".mov",
        ".css", ".js", ".json", ".xml"
    };
    
    for (int i = 0; i < sizeof(file_extensions)/sizeof(file_extensions[0]); i++) {
        if (strcasestr(url, file_extensions[i])) {
            return 0;
        }
    }
    
    // For medium.com URLs, add special handling
    if (strstr(url, "medium.com") != NULL) {
        // Exclude specific Medium paths that cause issues
        if (strstr(url, "medium.com/m/signin") != NULL || 
            strstr(url, "medium.com/m/signout") != NULL ||
            strstr(url, "medium.com/plans") != NULL ||
            strstr(url, "help.medium.com") != NULL ||
            strstr(url, "policy.medium.com") != NULL ||
            strstr(url, "statuspage.medium.com") != NULL ||
            strstr(url, "medium.com/about") != NULL ||
            strstr(url, "medium.com/jobs") != NULL ||
            strstr(url, "medium.com/_/graphql") != NULL ||
            strstr(url, "cdn-client.medium.com") != NULL) {
            return 0;
        }
        
        // Allow specific Medium paths
        if (strstr(url, "medium.com/@") != NULL ||       // Profile pages
            strstr(url, "/p/") != NULL ||                // Article pages
            strstr(url, "/tag/") != NULL ||              // Tag pages
            strstr(url, "/topics/") != NULL ||           // Topic pages
            strstr(url, "medium.com/") != NULL) {        // Publication pages
            return 1;
        }
    } else if (base_domain != NULL && strstr(url, base_domain) != NULL) {
        // For other domains, require that the URL contains our base domain
        return 1;
    }
    
    return 0;
}

// Function to recursively crawl a website starting from a URL
int crawl_website(const char* start_url, int maxDepth, int maxPages) {
    if (!start_url || maxDepth < 1 || maxPages < 1) return 0;
    
    // Start measuring crawling time
    start_timer();

    // Initialize OpenMP locks for thread safety
    omp_init_lock(&visited_lock);

    // URL queue with thread-safe locks
    char* queue[MAX_URLS];
    int depth[MAX_URLS];  // Track depth of each URL
    
    omp_lock_t queue_lock; // Lock for queue operations
    omp_init_lock(&queue_lock);
    
    int front = 0, rear = 0;
    
    // Shared atomic counters for better thread synchronization
    int total_pages_crawled = 0;
    int total_failed_downloads = 0;
    
    // Normalize and add start URL to queue
    char* normalized_start_url = normalize_url(start_url);
    if (!normalized_start_url || normalized_start_url[0] == '\0') {
        printf("Invalid starting URL: %s\n", start_url);
        return 0;
    }
    
    // Safely add to queue
    queue[rear] = strdup(normalized_start_url);
    if (!queue[rear]) {
        printf("Memory allocation failed for starting URL\n");
        return 0;
    }
    
    depth[rear] = 1;
    rear = (rear + 1) % MAX_URLS;
    
    // Mark as visited
    mark_visited(normalized_start_url);
    
    printf("Starting crawl from: %s (max depth: %d, max pages: %d)\n", normalized_start_url, maxDepth, maxPages);
    
    // Extract base domain from start_url
    char* base_domain = extract_base_domain(start_url);
    printf("Base domain for crawling: %s\n", base_domain);
    
    // Initialize curl globally (once)
    curl_global_init(CURL_GLOBAL_DEFAULT);
    
    // Get the number of threads set by the user in main.c
    int num_threads = omp_get_max_threads();
    printf("Crawling with %d parallel threads (configured via OMP_NUM_THREADS or -t option)\n", num_threads);
    
    // Initialize the thread state flags for better coordination
    int active_threads = num_threads;
    int* thread_active = calloc(num_threads, sizeof(int));
    if (!thread_active) {
        fprintf(stderr, "Failed to allocate memory for thread activity tracking\n");
        return 0;
    }
    
    // Initialize all threads as active
    for (int i = 0; i < num_threads; i++) {
        thread_active[i] = 1;
    }
    
    // Use the number of threads set by the user in main.c
    printf("Using %d OpenMP threads for crawling\n", num_threads);
    
    // Array to track pages crawled per thread for statistics
    int* thread_pages = calloc(num_threads, sizeof(int));
    if (!thread_pages) {
        fprintf(stderr, "Failed to allocate memory for thread statistics\n");
        free(thread_active);
        return 0;
    }
    
    // Enhanced queue seeding with multiple strategies for balanced workload
    if (maxPages >= num_threads) {
        // Strategy 1: Create multiple initial entry points with URL fragments 
        int initial_urls = num_threads * 2;  // Create more initial URLs than threads
        if (initial_urls > maxPages/2) initial_urls = maxPages/2;
        
        // Create variations of the initial URL with different fragments
        // This helps threads start with different entry points
        for (int i = 0; i < initial_urls; i++) {
            char seed_url[MAX_URL_LENGTH];
            
            // Use different URL fragment patterns to maximize diversity
            if (i % 3 == 0) {
                snprintf(seed_url, MAX_URL_LENGTH, "%s#section%d", normalized_start_url, i);
            } else if (i % 3 == 1) {
                snprintf(seed_url, MAX_URL_LENGTH, "%s?t=%d", normalized_start_url, i);
            } else {
                snprintf(seed_url, MAX_URL_LENGTH, "%s#thread%d", normalized_start_url, i);
            }
            
            char* url_copy = strdup(seed_url);
            if (url_copy) {
                queue[rear] = url_copy;
                depth[rear] = 1;
                rear = (rear + 1) % MAX_URLS;
                printf("Seeded queue with variant of start URL: %s\n", seed_url);
            }
        }
        
        // Strategy 2: For Medium sites, also seed with known URL patterns that will yield good crawl results
        if (strstr(normalized_start_url, "medium.com") != NULL) {
            const char* medium_paths[] = {
                "/latest", "/popular", "/tagged/programming", "/tagged/technology"
            };
            
            char base_medium_url[MAX_URL_LENGTH];
            if (strstr(normalized_start_url, "/@")) {
                // For profile URLs, keep the profile part
                strncpy(base_medium_url, normalized_start_url, MAX_URL_LENGTH - 1);
                base_medium_url[MAX_URL_LENGTH - 1] = '\0';
            } else {
                // For other Medium URLs, just use the domain
                strcpy(base_medium_url, "https://medium.com");
            }
            
            // Add a few Medium-specific paths
            int added_medium = 0;
            for (int i = 0; i < sizeof(medium_paths)/sizeof(medium_paths[0]) && added_medium < 4; i++) {
                char medium_seed[MAX_URL_LENGTH];
                snprintf(medium_seed, MAX_URL_LENGTH, "%s%s", base_medium_url, medium_paths[i]);
                
                char* url_copy = strdup(medium_seed);
                if (url_copy) {
                    queue[rear] = url_copy;
                    depth[rear] = 1;
                    rear = (rear + 1) % MAX_URLS;
                    printf("Seeded queue with Medium-specific URL: %s\n", medium_seed);
                    added_medium++;
                }
            }
        }
        
        printf("Queue seeded with %d initial URLs to ensure balanced thread distribution\n", 
              (rear - front + MAX_URLS) % MAX_URLS);
    }
    
    // Start crawling in parallel with explicitly fixed number of threads and improved coordination
    #pragma omp parallel num_threads(num_threads) shared(queue, depth, front, rear, queue_lock, \
                                                         total_pages_crawled, total_failed_downloads, \
                                                         thread_active, active_threads, thread_pages)
    {
        // Thread-local variables for better performance
        int thread_id = omp_get_thread_num();
        int local_pages_crawled = 0;
        int local_failed_downloads = 0;
        int consecutive_empty = 0;  // Track consecutive empty queue encounters
        
        // Print initial thread info
        show_crawling_progress(thread_id, "Thread started and ready to crawl");
        
        // Set random seed differently for each thread to avoid synchronization issues
        // Plus use a better seed with microsecond precision if available
        struct timespec ts;
        unsigned int thread_seed;
        if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
            thread_seed = ts.tv_nsec + thread_id * 100000;
        } else {
            thread_seed = time(NULL) + thread_id * 1000;
        }
        srand(thread_seed);
        
        // Each thread processes URLs until all work is done or max pages reached
        while (total_pages_crawled < maxPages) {
            char* current_url = NULL;
            int current_depth = 0;
            int should_continue = 0;
            int queue_empty = 0;
            
            // Critical section for getting a URL from the queue - only one thread at a time
            #pragma omp critical(queue_access)
            {
                if (front != rear && total_pages_crawled < maxPages && total_failed_downloads < 10) {
                    // Get a URL from the queue
                    current_url = queue[front];
                    current_depth = depth[front];
                    front = (front + 1) % MAX_URLS;
                    should_continue = 1;
                    
                    // Mark this thread as active and reset empty counter
                    thread_active[thread_id] = 1;
                    consecutive_empty = 0;
                } else {
                    queue_empty = 1;
                }
            }
            
            // If there's no URL to process, implement improved waiting strategy with backoff
            if (queue_empty) {
                thread_active[thread_id] = 0;  // Mark as inactive
                consecutive_empty++;           // Increment empty counter
                
                // Check if any other thread is still active or has work in queue
                int any_active = 0;
                int queue_has_items = 0;
                int threads_extracting_links = 0;
                
                #pragma omp critical(queue_access)
                {
                    queue_has_items = (front != rear);
                    
                    if (queue_has_items) {
                        // Queue still has items, reactivate this thread
                        thread_active[thread_id] = 1;
                        any_active = 1;
                        consecutive_empty = 0;
                    } else {
                        // Check if any other thread is active and currently processing
                        for (int i = 0; i < num_threads; i++) {
                            if (thread_active[i]) {
                                any_active = 1;
                                
                                // Check if this thread is likely extracting links (useful for debugging)
                                if (thread_pages[i] > 0) {
                                    threads_extracting_links++;
                                }
                                
                                break;
                            }
                        }
                    }
                    
                    // Print diagnostic info when threads are waiting but there might still be work
                    if (consecutive_empty > 5 && any_active && !queue_has_items) {
                        printf("[Thread %d] Waiting: queue empty but %d threads still active (%d extracting links)\n", 
                              thread_id, any_active, threads_extracting_links);
                    }
                }
                
                // If we've reached the maximum number of pages, exit
                if (total_pages_crawled >= maxPages) {
                    break;
                }
                
                // If the queue is empty and no threads are active, or we've downloaded enough pages
                if (!any_active && !queue_has_items) {
                    #pragma omp critical(output)
                    {
                        printf("[Thread %d] Exiting: no more work to do\n", thread_id);
                    }
                    break; // All threads are inactive and queue is empty
                }
                
                // Advanced adaptive wait strategy with dynamic adjustment based on thread history
                // This reduces CPU usage during low work periods but reacts quickly when work appears
                int wait_time = 50000;  // Base: 50ms
                
                // Adjust wait time based on consecutive empty attempts and queue history
                if (consecutive_empty > 20) {
                    wait_time = 750000; // 750ms for very long idle periods
                } else if (consecutive_empty > 10) {
                    wait_time = 400000; // 400ms after 10 consecutive empty attempts
                } else if (consecutive_empty > 5) {
                    wait_time = 150000; // 150ms after 5 consecutive empty attempts
                }
                
                // Adjust wait time based on thread ID to distribute wakup times
                // Even-numbered threads wait a bit less to check more frequently
                if (thread_id % 2 == 0 && wait_time > 100000) {
                    wait_time = wait_time * 3 / 4;  // 25% shorter wait for even threads
                }
                
                // If this thread has processed more pages than others, it's likely more efficient
                // and should check more often
                if (thread_pages[thread_id] > 0) {
                    int avg_pages = 0;
                    for (int i = 0; i < num_threads; i++) {
                        avg_pages += thread_pages[i];
                    }
                    avg_pages /= num_threads;
                    
                    if (thread_pages[thread_id] > avg_pages * 1.5) {
                        // This thread is 50% more productive than average, let it check more often
                        wait_time = wait_time / 2;
                    }
                }
                
                // Wait with a slight variation between threads to avoid synchronization
                usleep(wait_time + (thread_id * 15000));
                continue;
            }
            
            if (!should_continue) {
                continue; // Try again if somehow we got here without a URL
            }
            
            // Skip invalid URLs or already processed ones
            if (!is_valid_crawl_url(current_url, base_domain)) {
                printf("  Thread %d: Skipping invalid URL: %s\n", thread_id, current_url);
                free(current_url);
                continue;
            }
            
            // Process the current URL with better progress reporting
            char progress_msg[256];
            snprintf(progress_msg, sizeof(progress_msg), "Crawling: %s (depth %d/%d)", 
                    current_url, current_depth, maxDepth);
            show_crawling_progress(thread_id, progress_msg);
            
            // Download the URL content
            struct MemoryStruct chunk;
            chunk.memory = malloc(1);
            chunk.size = 0;
        
            CURL* curl = curl_easy_init();
            if (curl) {
                curl_easy_setopt(curl, CURLOPT_URL, current_url);
                curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data_callback);
                curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void*)&chunk);
                curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
                curl_easy_setopt(curl, CURLOPT_USERAGENT, "Mozilla/5.0 (compatible; SearchEngine-Crawler/1.1)");
                curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
                curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);  // Disable SSL verification for simplicity
                
                // Add headers to mimic a browser request
                struct curl_slist *headers = NULL;
                headers = curl_slist_append(headers, "Accept: text/html,application/xhtml+xml,application/xml");
                headers = curl_slist_append(headers, "Accept-Language: en-US,en;q=0.9");
                curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
                
                CURLcode res = curl_easy_perform(curl);
                curl_slist_free_all(headers);
                curl_easy_cleanup(curl);
                
                if (res == CURLE_OK && chunk.size > 100) {  // Ensure we have some content
                    if (chunk.memory == NULL) {
                        fprintf(stderr, "  Downloaded content is NULL for %s\n", current_url);
                        local_failed_downloads++;
                    } else {
                        // Check if this URL has been processed by another thread while we were downloading
                        int already_processed = 0;
                        char* normalized_current = normalize_url(current_url);
                        
                        // Verify this URL hasn't already been processed in this session
                        char potential_filename[512];
                        
                        if (strstr(current_url, "medium.com") != NULL) {
                            // For Medium URLs, check general pattern
                            DIR* dir = opendir("dataset");
                            if (dir) {
                                struct dirent* entry;
                                while ((entry = readdir(dir))) {
                                    if (strstr(entry->d_name, "medium_") && 
                                        (strstr(entry->d_name, strrchr(normalized_current, '@') ? strrchr(normalized_current, '@') + 1 : "") ||
                                         strstr(entry->d_name, normalized_current))) {
                                        already_processed = 1;
                                        break;
                                    }
                                }
                                closedir(dir);
                            }
                        } else {
                            // For other URLs, try the standard filename
                            snprintf(potential_filename, sizeof(potential_filename), "dataset/%s", get_url_filename(current_url));
                            if (access(potential_filename, F_OK) == 0) {
                                already_processed = 1;
                            }
                        }
                        
                        if (already_processed) {
                            char msg[256];
                            snprintf(msg, sizeof(msg), "Already downloaded URL to dataset, skipping: %s", current_url);
                            show_crawling_progress(thread_id, msg);
                            // Count as success
                            local_pages_crawled++;
                            thread_pages[thread_id]++;  // Track per-thread page count
                        } else {
                            // Download and process the URL
                            char* filename = download_url(current_url);
                            if (filename) {
                                char msg[256];
                                snprintf(msg, sizeof(msg), "Downloaded to %s (%zu bytes)", filename, chunk.size);
                                show_crawling_progress(thread_id, msg);
                                local_pages_crawled++;
                                thread_pages[thread_id]++;  // Track per-thread page count
                                local_failed_downloads = 0;  // Reset consecutive failure counter
                            } else {
                                local_failed_downloads++;
                                char msg[256];
                                snprintf(msg, sizeof(msg), "Failed to save content from: %s", current_url);
                                show_crawling_progress(thread_id, msg);
                            }
                        }
                        
                        // If we have not reached max depth, extract links and add to queue
                        if (current_depth < maxDepth) {
                            char* extracted_urls[MAX_URLS];
                            int url_count = 0;
                            
                            // Extract links from the page
                            extract_links(chunk.memory, current_url, extracted_urls, &url_count, MAX_URLS);
                            char msg[256];
                            snprintf(msg, sizeof(msg), "Found %d links in %s", url_count, current_url);
                            show_crawling_progress(thread_id, msg);
                            
                            // First filter URLs outside the critical section to minimize lock contention
                            int valid_urls = 0;
                            char* valid_url_list[MAX_URLS];
                            int valid_url_depths[MAX_URLS];
                            
                            // Pre-process URLs to minimize critical section time
                            for (int i = 0; i < url_count && valid_urls < MAX_URLS; i++) {
                                if (!extracted_urls[i]) continue;
                                
                                // Pre-check for validity without locking
                                if (!is_valid_crawl_url(extracted_urls[i], base_domain)) {
                                    free(extracted_urls[i]);
                                    extracted_urls[i] = NULL;
                                    continue;
                                }
                                
                                valid_url_list[valid_urls] = extracted_urls[i];
                                valid_url_depths[valid_urls] = current_depth + 1;
                                valid_urls++;
                                extracted_urls[i] = NULL; // Prevent double free
                            }
                            
                            // Add new URLs to the queue with proper synchronization
                            if (valid_urls > 0) {
                                // Reactivate all threads since we have new work
                                for (int i = 0; i < num_threads; i++) {
                                    thread_active[i] = 1;
                                }
                                
                                // Enhanced URL distribution with intelligent workload distribution and queue rebalancing
                                // Using thread performance metrics to guide URL distribution
                                #pragma omp critical(queue_access)
                                {
                                    int added_urls = 0;
                                    int max_to_add = 20;  // Default limit URLs per page to prevent single-thread dominance
                                    
                                    // Calculate how many URLs to add based on current queue size and thread activity
                                    int current_queue_size = (rear - front + MAX_URLS) % MAX_URLS;
                                    int active_thread_count = 0;
                                    
                                    // Count active threads for better workload estimation
                                    for (int i = 0; i < num_threads; i++) {
                                        if (thread_active[i]) active_thread_count++;
                                    }
                                    
                                    // Dynamic URL addition based on queue state and thread activity
                                    if (current_queue_size < active_thread_count * 2) {
                                        // Queue almost empty relative to active threads - add more URLs
                                        max_to_add = valid_urls > 75 ? 75 : valid_urls;
                                        printf("  Queue low (%d items) with %d active threads - adding up to %d URLs\n", 
                                              current_queue_size, active_thread_count, max_to_add);
                                    } else if (current_queue_size < num_threads) {
                                        // Some URLs but not enough for all threads
                                        max_to_add = valid_urls > 40 ? 40 : valid_urls;
                                    }
                                    
                                    // Add URLs up to the limit
                                    for (int i = 0; i < valid_urls && 
                                         rear != (front - 1 + MAX_URLS) % MAX_URLS && 
                                         added_urls < max_to_add; i++) {
                                         
                                        if (!valid_url_list[i]) continue;
                                        
                                        // Final check for visited within critical section
                                        if (has_visited(valid_url_list[i])) {
                                            free(valid_url_list[i]);
                                            valid_url_list[i] = NULL;
                                            continue;
                                        }
                                        
                                        queue[rear] = valid_url_list[i];
                                        depth[rear] = valid_url_depths[i];
                                        rear = (rear + 1) % MAX_URLS;
                                        mark_visited(valid_url_list[i]);
                                        
                                        char queue_msg[256];
                                        snprintf(queue_msg, sizeof(queue_msg), "Thread %d queued: %s", thread_id, valid_url_list[i]);
                                        printf("  %s\n", queue_msg);
                                        
                                        added_urls++;
                                        valid_url_list[i] = NULL;
                                    }
                                    
                                    // Print queue status after adding URLs
                                    if (added_urls > 0) {
                                        printf("  [Queue status] Size: %d, Added: %d URLs from thread %d\n", 
                                              (rear - front + MAX_URLS) % MAX_URLS,
                                              added_urls, thread_id);
                                    }
                                }
                                
                                // Free any remaining valid URLs that weren't added to queue
                                for (int i = 0; i < valid_urls; i++) {
                                    if (valid_url_list[i] != NULL) {
                                        free(valid_url_list[i]);
                                        valid_url_list[i] = NULL;
                                    }
                                }
                            }
                            
                            // Free any remaining extracted URLs
                            for (int i = 0; i < url_count; i++) {
                                if (extracted_urls[i] != NULL) {
                                    free(extracted_urls[i]);
                                    extracted_urls[i] = NULL;
                                }
                            }
                        }
                    }
                } else {
                    fprintf(stderr, "  Failed to download or process content from: %s (size: %zu bytes)\n", 
                            current_url, chunk.size);
                    local_failed_downloads++;
                }
                
                // Safely free the memory chunk
                if (chunk.memory) {
                    free(chunk.memory);
                    chunk.memory = NULL;
                }
            } else {
                local_failed_downloads++;
                fprintf(stderr, "  Failed to initialize curl for: %s\n", current_url);
            }
            
            free(current_url);
            
            // Update global counters with atomic operations
            #pragma omp atomic
            total_pages_crawled += local_pages_crawled;
            
            #pragma omp atomic
            total_failed_downloads += local_failed_downloads;
            
            // Reset local counters after updating global ones
            local_pages_crawled = 0;
            local_failed_downloads = 0;
            
            // Improved thread coordination - print periodic status to show active threads
            if (rand() % 100 < 10) {
                #pragma omp critical(output)
                {
                    int active_count = 0;
                    for (int i = 0; i < num_threads; i++) {
                        if (thread_active[i]) active_count++;
                    }
                    printf("Thread status: %d/%d active, %d pages crawled\n", 
                          active_count, num_threads, total_pages_crawled);
                }
            }
            
            // Add a small delay between requests to be nice to servers (200-500ms)
            // Use a variable delay based on thread ID to prevent synchronization of threads
            int delay_ms = (rand() % 300 + 200) + (thread_id * 50);
            usleep(delay_ms * 1000);
            
            // Better load balancing for threads - dynamically adjust delays based on queue size
            #pragma omp critical(queue_access)
            {
                int queue_size = (rear - front + MAX_URLS) % MAX_URLS;
                // If queue is getting smaller than half the number of threads,
                // add additional delay to some threads to allow better distribution
                if (queue_size < num_threads / 2 && num_threads > 4) {
                    // Slow down odd-numbered threads when queue is small
                    // This helps prevent thread starvation by letting some threads work more
                    if (thread_id % 2 == 1) {
                        usleep(100000); // Extra 100ms for odd-numbered threads
                    }
                } else if (queue_size > num_threads * 4) {
                    // When queue is large, have all threads work at full speed
                    // No additional delay
                }
            }
        }
        
        // Print final report for this thread
        #pragma omp critical(output)
        {
            printf("[Thread %d] Finished crawling with %d pages processed\n", 
                  thread_id, thread_pages[thread_id]);
        }
    }
    
    // Show thread distribution statistics for better insights
    show_thread_distribution(num_threads, thread_pages);
    
    // Clean up any remaining URLs in the queue
    while (front != rear) {
        if (queue[front] != NULL) {
            free(queue[front]);
            queue[front] = NULL;
        }
        front = (front + 1) % MAX_URLS;
    }
    
    // Free the thread activity tracking array
    free(thread_active);
    free(thread_pages);
    
    // Clean up curl global state
    curl_global_cleanup();
    
    // Clean up OpenMP locks
    omp_destroy_lock(&visited_lock);
    omp_destroy_lock(&queue_lock);
    
    // Record crawling time
    metrics.crawling_time = stop_timer();
    
    printf("\nCrawling completed. Crawled %d pages in %.2f ms.\n", 
           total_pages_crawled, metrics.crawling_time);
    return total_pages_crawled;
}

// Function to display a progress indicator for crawling
static void show_crawling_progress(int thread_id, const char* message) {
    // Using critical section to prevent output garbling
    #pragma omp critical(output)
    {
        printf("[Thread %d] %s\n", thread_id, message);
    }
}

// Function to display thread activity statistics with enhanced metrics
static void show_thread_distribution(int num_threads, int* thread_pages) {
    printf("\nThread workload distribution:\n");
    printf("----------------------------\n");
    
    int total = 0;
    int min_pages = INT_MAX;
    int max_pages = 0;
    int empty_threads = 0;
    
    // First pass: calculate totals and extremes
    for (int i = 0; i < num_threads; i++) {
        total += thread_pages[i];
        
        if (thread_pages[i] < min_pages) min_pages = thread_pages[i];
        if (thread_pages[i] > max_pages) max_pages = thread_pages[i];
        if (thread_pages[i] == 0) empty_threads++;
    }
    
    // Calculate averages and imbalance metrics
    double avg_pages = (total > 0 && num_threads > 0) ? ((double)total / num_threads) : 0;
    double load_imbalance = (avg_pages > 0 && max_pages > 0) ? 
                           ((max_pages - avg_pages) / max_pages) * 100.0 : 0;
    double coefficient_of_variation = 0.0;
    
    if (avg_pages > 0) {
        double variance = 0.0;
        for (int i = 0; i < num_threads; i++) {
            variance += pow(thread_pages[i] - avg_pages, 2);
        }
        variance /= num_threads;
        coefficient_of_variation = sqrt(variance) / avg_pages * 100.0;
    }
    
    // Display individual thread statistics
    for (int i = 0; i < num_threads; i++) {
        double thread_percent = (total > 0) ? (thread_pages[i] * 100.0 / total) : 0.0;
        printf("Thread %d: %d pages (%.1f%%)", i, thread_pages[i], thread_percent);
        
        // Highlight threads that are significantly above or below average
        if (thread_pages[i] > avg_pages * 1.5 && thread_pages[i] > 3)
            printf(" [OVERLOADED]");
        else if (thread_pages[i] < avg_pages * 0.5 && avg_pages > 3)
            printf(" [UNDERUTILIZED]");
        else if (thread_pages[i] == 0)
            printf(" [IDLE]");
        
        printf("\n");
    }
    
    // Display summary statistics
    printf("\nSummary Statistics:\n");
    printf("Total: %d pages | Threads: %d | Avg: %.1f pages/thread\n", 
           total, num_threads, avg_pages);
    printf("Min/Max: %d/%d pages | Imbalance: %.1f%% | CoV: %.1f%%\n", 
           min_pages, max_pages, load_imbalance, coefficient_of_variation);
    
    if (empty_threads > 0) {
        printf("WARNING: %d thread(s) processed no pages!\n", empty_threads);
    }
    
    printf("----------------------------\n");
}
