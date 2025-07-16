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
#include <dirent.h>
#include <mpi.h>

#define MAX_URLS 1000
#define MAX_URL_LENGTH 512

static char* normalize_url(const char* url);
static char* extract_base_domain(const char* url);
static int has_visited(const char* url);
static void mark_visited(const char* url);
static void extract_links(const char* html, const char* base_url, char** urls, int* url_count, int max_urls);

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
                i += 2;  // Skip past <h1/>
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
// MPI variables
static int mpi_rank = 0; // Process rank
static int mpi_size = 1; // Number of processes
static int mpi_initialized = 0; // Flag to check if MPI is initialized

// Initialize MPI info
static void init_mpi_crawler_info() {
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    mpi_initialized = 1;
}

// Function to check if a URL has been visited
static int has_visited(const char* url) {
    if (!url) return 1;  // Treat NULL URLs as already visited
    
    // First normalize the URL for consistent comparison
    char* normalized = normalize_url(url);
    if (!normalized || normalized[0] == '\0') return 1; // Treat empty URLs as visited
    
    int result = 0;
    
    // Add lock to protect the read operation
    // omp_set_lock(&visited_lock);
    
    // Safe comparison with visited URLs
    for (int i = 0; i < visited_count; i++) {
        if (visited_urls[i][0] != '\0' && strcmp(visited_urls[i], normalized) == 0) {
            result = 1;
            break;
        }
    }
    
    // omp_unset_lock(&visited_lock);
    
    return result;
}

// Function to mark a URL as visited
static void mark_visited(const char* url) {
    if (!url) return;  // Don't try to mark NULL URLs
    
    // First normalize the URL for consistent storage
    char* normalized = normalize_url(url);
    if (!normalized || normalized[0] == '\0') return; // Don't mark empty URLs
    
    // omp_set_lock(&visited_lock);
    
    // Check if already in our visited list
    for (int i = 0; i < visited_count; i++) {
        if (visited_urls[i][0] != '\0' && strcmp(visited_urls[i], normalized) == 0) {
            // omp_unset_lock(&visited_lock);
            return; // Already marked
        }
    }
    
    // Add to visited list if space is available
    if (visited_count < MAX_URLS) {
        strncpy(visited_urls[visited_count], normalized, MAX_URL_LENGTH - 1);
        visited_urls[visited_count][MAX_URL_LENGTH - 1] = '\0';
        visited_count++;
    }
    
    // omp_unset_lock(&visited_lock);
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
static void process_extracted_url(const char* url_text, int url_len, const char* base_url, const char* base_domain, 
                                char** urls, int* url_count, int max_urls) {
    if (url_len <= 0 || url_len >= MAX_URL_LENGTH || *url_count >= max_urls) 
        return;
        
    // Allocate and copy the URL
    char* new_url = malloc(MAX_URL_LENGTH);
    if (!new_url) return;
    
    strncpy(new_url, url_text, url_len);
    new_url[url_len] = '\0';
    
    // Skip javascript: urls, mailto: links, tel: links, data: URIs
    if (strncmp(new_url, "javascript:", 11) == 0 ||
        strncmp(new_url, "mailto:", 7) == 0 ||
        strncmp(new_url, "tel:", 4) == 0 ||
        strncmp(new_url, "data:", 5) == 0 ||
        strncmp(new_url, "#", 1) == 0) {  // Skip page anchors
        free(new_url);
        return;
    }
    
    // Special handling for Medium URLs
    int is_medium_url = strstr(base_url, "medium.com") != NULL;
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
    }
    
    // Handle relative URLs
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
        if (strstr(base_url, "medium.com") != NULL && strstr(final_url, "medium.com") != NULL) {
            // For Medium we allow all URLs within medium.com
            is_valid = 1;
        } else if (base_domain && strstr(final_url, base_domain) != NULL) {
            // For other sites, use stricter domain checking
            is_valid = 1;
        }
    }
    
    // Add URL to the list or free it
    if (is_valid && !is_duplicate && *url_count < max_urls) {
        urls[*url_count] = final_url;
        (*url_count)++;
    } else {
        free(final_url);
    }
}

// Function to extract links from HTML content
static void extract_links(const char* html, const char* base_url, char** urls, int* url_count, int max_urls) {
    if (!html || !base_url || !urls || !url_count || max_urls <= 0) return;
    
    const char* base_domain = extract_base_domain(base_url);
    if (!base_domain || base_domain[0] == '\0') return;
    
    *url_count = 0;  // Initialize count
    
    // Special handling for Medium profiles
    int is_medium_profile = strstr(base_url, "medium.com/@") != NULL;
    
    // Determine HTML length to divide work among threads
    size_t html_len = strlen(html);
    // Get MPI rank and size
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    // For Medium profiles, use fewer processes
    int active_processes = mpi_size;
    if (is_medium_profile && active_processes > 2) {
        active_processes = 2; // Use fewer processes to avoid splitting link structures
        if (mpi_rank == 0) {
            printf("  Using %d processes for Medium profile link extraction\n", active_processes);
        }
    }
    
    // Only processes with rank < active_processes participate
    if (mpi_rank < active_processes) {
        // Each process handles a section of HTML
        size_t chunk_size = html_len / active_processes;
        size_t start_pos = mpi_rank * chunk_size;
        size_t end_pos = (mpi_rank == active_processes - 1) ? html_len : (mpi_rank + 1) * chunk_size;
        
        // For all processes except first, find the first '<' character to get a clean start point
        if (mpi_rank > 0) {
            while (start_pos < end_pos && html[start_pos] != '<')
                start_pos++;
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
                        }
                    } else {
                        // Regular URL handling
                        const char* url_end = strchr(url_start, '"');
                        if (url_end && url_end < end_ptr) {
                            int url_len = url_end - url_start;
                            if (url_len > 0 && url_len < MAX_URL_LENGTH) {
                                process_extracted_url(url_start, url_len, base_url, base_domain, 
                                                    local_urls, &local_count, 100);
                            }
                        }
                    }
                    // Move past this URL for next iteration
                    search_ptr = url_start + 1;
                }
            }
        } else {
            // Regular URL extraction for non-Medium profiles
            while (ptr < end_ptr && local_count < 100) {
                // Look for href attributes
                const char* href_start = NULL;
                const char* href_end = NULL;
                
                // Search for "href=" in both quote styles
                const char* href_double = strstr(ptr, "href=\"");
                const char* href_single = strstr(ptr, "href='");
                
                if (!href_double && !href_single) break;
                
                if (href_double && href_single) {
                    // Use whichever comes first
                    if (href_double < href_single) {
                        href_start = href_double + 6;  // Skip past href="
                        href_end = strchr(href_start, '"');
                    } else {
                        href_start = href_single + 6;  // Skip past href='
                        href_end = strchr(href_start, '\'');
                    }
                } else if (href_double) {
                    href_start = href_double + 6;
                    href_end = strchr(href_start, '"');
                } else {
                    href_start = href_single + 6;
                    href_end = strchr(href_start, '\'');
                }
                
                // If we found both start and end of the URL and it's within our chunk
                if (href_start && href_end && href_end < end_ptr) {
                    int url_len = href_end - href_start;
                    if (url_len > 0 && url_len < MAX_URL_LENGTH) {
                        process_extracted_url(href_start, url_len, base_url, base_domain, 
                                            local_urls, &local_count, 100);
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
            // omp_set_lock(&url_lock);
            for (int i = 0; i < local_count && *url_count < max_urls; i++) {
                if (local_urls[i]) {
                    urls[*url_count] = local_urls[i];
                    (*url_count)++;
                    local_urls[i] = NULL; // Prevent double-free
                }
            }
            // omp_unset_lock(&url_lock);
            
            // Free any remaining URLs that didn't make it to the global list
            for (int i = 0; i < local_count; i++) {
                if (local_urls[i]) {
                    free(local_urls[i]);
                    local_urls[i] = NULL;
                }
            }
        }
    }
    
    // Clean up
    // omp_destroy_lock(&url_lock);
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

// MPI work sharing structure
typedef struct {
    char url[MAX_URL_LENGTH];
    int depth;
    int assigned_rank;
    int processed;
} MPIWorkItem;

// Global work queue shared among all MPI processes
static MPIWorkItem mpi_work_queue[MAX_URLS];
static int mpi_queue_size = 0;
static int global_pages_crawled = 0;
static int work_distribution_round = 0;

// Traditional queue variables for compatibility
static char* queue[MAX_URLS];
static int depth[MAX_URLS];
static int front = 0;
static int rear = 0;
static int pages_crawled = 0;
static int failed_downloads = 0;

// Function to distribute work among MPI processes
static void distribute_mpi_work(const char* start_url, int maxDepth, int maxPages) {
    if (mpi_rank == 0) {
        // Master process initializes the work queue
        mpi_queue_size = 0;
        global_pages_crawled = 0;
        
        // Add initial URL to work queue
        strcpy(mpi_work_queue[0].url, start_url);
        mpi_work_queue[0].depth = 1;
        mpi_work_queue[0].assigned_rank = 0;
        mpi_work_queue[0].processed = 0;
        mpi_queue_size = 1;
        
        printf("Master process initialized work queue with %s\n", start_url);
    }
    
    // Broadcast the initial work queue to all processes
    MPI_Bcast(&mpi_queue_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(mpi_work_queue, MAX_URLS * sizeof(MPIWorkItem), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&global_pages_crawled, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

// Function to get next work item for this MPI process
static int get_mpi_work_item(char* url_out, int* depth_out, int maxPages) {
    // Check if we've reached the page limit
    if (global_pages_crawled >= maxPages) {
        return 0; // No more work
    }
    
    // Look for unassigned work items
    for (int i = 0; i < mpi_queue_size; i++) {
        if (!mpi_work_queue[i].processed) {
            // Assign work in round-robin fashion
            int target_rank = work_distribution_round % mpi_size;
            
            if (mpi_rank == target_rank) {
                strcpy(url_out, mpi_work_queue[i].url);
                *depth_out = mpi_work_queue[i].depth;
                mpi_work_queue[i].assigned_rank = mpi_rank;
                mpi_work_queue[i].processed = 1;
                work_distribution_round++;
                
                printf("Process %d took work item %d: %s (depth %d)\n", 
                       mpi_rank, i, url_out, *depth_out);
                return 1; // Got work
            }
            work_distribution_round++;
        }
    }
    
    return 0; // No work available for this process
}

// Function to add new URLs to the shared work queue
static void add_mpi_work_items(char** urls, int* depths, int url_count, int maxDepth) {
    if (url_count == 0) return;
    
    int added_count = 0;
    for (int i = 0; i < url_count && mpi_queue_size < MAX_URLS && added_count < 50; i++) {
        if (urls[i] && depths[i] <= maxDepth) {
            // Check if URL is already in queue
            int already_exists = 0;
            for (int j = 0; j < mpi_queue_size; j++) {
                if (strcmp(mpi_work_queue[j].url, urls[i]) == 0) {
                    already_exists = 1;
                    break;
                }
            }
            
            if (!already_exists && !has_visited(urls[i])) {
                strcpy(mpi_work_queue[mpi_queue_size].url, urls[i]);
                mpi_work_queue[mpi_queue_size].depth = depths[i];
                mpi_work_queue[mpi_queue_size].assigned_rank = -1;
                mpi_work_queue[mpi_queue_size].processed = 0;
                mpi_queue_size++;
                added_count++;
                
                printf("Process %d added to queue: %s (depth %d)\n", 
                       mpi_rank, urls[i], depths[i]);
            }
        }
    }
    
    if (added_count > 0) {
        printf("Process %d added %d new URLs to shared queue (total: %d)\n", 
               mpi_rank, added_count, mpi_queue_size);
    }
}

// Function to synchronize work queues across all MPI processes
static void sync_mpi_work_queues() {
    // Each process contributes its local queue updates
    MPIWorkItem temp_queue[MAX_URLS];
    int temp_queue_size = mpi_queue_size;
    
    // Gather queue sizes from all processes
    int all_queue_sizes[mpi_size];
    MPI_Allgather(&temp_queue_size, 1, MPI_INT, all_queue_sizes, 1, MPI_INT, MPI_COMM_WORLD);
    
    // Find the process with the largest queue (most up-to-date)
    int max_size = 0;
    int master_process = 0;
    for (int i = 0; i < mpi_size; i++) {
        if (all_queue_sizes[i] > max_size) {
            max_size = all_queue_sizes[i];
            master_process = i;
        }
    }
    
    // Broadcast the most up-to-date queue from the master process
    if (max_size > 0) {
        MPI_Bcast(&mpi_queue_size, 1, MPI_INT, master_process, MPI_COMM_WORLD);
        MPI_Bcast(mpi_work_queue, MAX_URLS * sizeof(MPIWorkItem), MPI_BYTE, master_process, MPI_COMM_WORLD);
    }
}

// Function to recursively crawl a website starting from a URL using MPI parallelization
int crawl_website(const char* start_url, int maxDepth, int maxPages) {
    // Start measuring crawling time
    if (mpi_rank == 0) {
        start_timer();
    }
    
    // Initialize MPI info if not already initialized
    if (!mpi_initialized) {
        init_mpi_crawler_info();
    }
    
    // Reset the visited URLs on all processes
    visited_count = 0;
    
    // Normalize start URL
    char* normalized_start_url = normalize_url(start_url);
    if (!normalized_start_url || normalized_start_url[0] == '\0') {
        if (mpi_rank == 0) {
            printf("Invalid starting URL: %s\n", start_url);
        }
        return 0;
    }
    
    // Extract base domain from start_url
    char* base_domain = extract_base_domain(start_url);
    
    if (mpi_rank == 0) {
        printf("Starting crawl from: %s (max depth: %d, max pages: %d)\n", 
               normalized_start_url, maxDepth, maxPages);
        printf("Base domain for crawling: %s\n", base_domain);
        printf("Crawling with %d MPI processes\n", mpi_size);
    }
    
    // Initialize curl globally (once per process)
    curl_global_init(CURL_GLOBAL_DEFAULT);
    
    // Distribute initial work among MPI processes
    distribute_mpi_work(normalized_start_url, maxDepth, maxPages);
    
    // Mark initial URL as visited
    mark_visited(normalized_start_url);
    
    // MPI parallel crawling loop
    int local_pages_crawled = 0;
    int local_failed_downloads = 0;
    int sync_counter = 0;
    
    while (global_pages_crawled < maxPages) {
        char current_url[MAX_URL_LENGTH];
        int current_depth;
        
        // Synchronize work queues every few iterations
        if (sync_counter % 5 == 0) {
            sync_mpi_work_queues();
            
            // Update global page count
            int local_count = local_pages_crawled;
            MPI_Allreduce(&local_count, &global_pages_crawled, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        }
        sync_counter++;
        
        // Get next work item for this process
        if (!get_mpi_work_item(current_url, &current_depth, maxPages)) {
            // No work available, wait a bit and try again
            usleep(100000); // 100ms
            continue;
        }
        
        // Check if URL is valid for crawling
        if (!is_valid_crawl_url(current_url, base_domain)) {
            printf("Process %d: Skipping invalid URL: %s\n", mpi_rank, current_url);
            continue;
        }
        
        printf("Process %d crawling [%d/%d]: %s (depth %d/%d)\n", 
               mpi_rank, global_pages_crawled + 1, maxPages, current_url, current_depth, maxDepth);
            
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
            char* filename = download_url(current_url);
            if (filename) {
                printf("  Process %d downloaded to %s (%zu bytes)\n", mpi_rank, filename, chunk.size);
                local_pages_crawled++;
                
                // If we have not reached max depth, extract links and add to queue
                if (current_depth < maxDepth) {
                    char* extracted_urls[MAX_URLS];
                    int url_count = 0;
                    
                    // Extract links from HTML
                    extract_links(chunk.memory, current_url, extracted_urls, &url_count, MAX_URLS);
                    printf("  Process %d found %d links\n", mpi_rank, url_count);
                    
                    // Add new URLs to the MPI work queue
                    if (url_count > 0) {
                        int* depths = malloc(url_count * sizeof(int));
                        for (int i = 0; i < url_count; i++) {
                            depths[i] = current_depth + 1;
                        }
                        add_mpi_work_items(extracted_urls, depths, url_count, maxDepth);
                        free(depths);
                        
                        // Free extracted URLs
                        for (int i = 0; i < url_count; i++) {
                            if (extracted_urls[i]) {
                                free(extracted_urls[i]);
                            }
                        }
                    }
                }
            } else {
                local_failed_downloads++;
                printf("  Process %d failed to save content from: %s\n", mpi_rank, current_url);
            }
        } else {
            fprintf(stderr, "  Process %d failed to download content from: %s (size: %zu bytes)\n", 
                    mpi_rank, current_url, chunk.size);
            local_failed_downloads++;
        }
        
        // Safely free the memory chunk
        if (chunk.memory) {
            free(chunk.memory);
            chunk.memory = NULL;
        }
    } else {
        local_failed_downloads++;
        fprintf(stderr, "  Process %d failed to initialize curl for: %s\n", mpi_rank, current_url);
    }
    
    // Add a small delay between requests to be nice to servers (200-500ms)
    usleep((rand() % 300 + 200) * 1000);
} // End of while loop

// Clean up any remaining URLs in the queue
while (front != rear) {
    if (queue[front] != NULL) {
        free(queue[front]);
        queue[front] = NULL;
    }
    front = (front + 1) % MAX_URLS;
}

// Clean up curl global state
curl_global_cleanup();

// Synchronize all processes before final cleanup
MPI_Barrier(MPI_COMM_WORLD);

// Calculate final global page count
MPI_Allreduce(&local_pages_crawled, &global_pages_crawled, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

// Record crawling time (only on master process)
if (mpi_rank == 0) {
    extern SearchEngineMetrics metrics;
    metrics.crawling_time = stop_timer();
    
    printf("\nCrawling completed. Crawled %d pages in %.2f ms.\n", 
           global_pages_crawled, metrics.crawling_time);
}

return global_pages_crawled;
}
