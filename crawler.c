#include "crawler.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <strings.h>  // For strcasecmp
#include <curl/curl.h>
#include <sys/stat.h>
#include <unistd.h>

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

// Function to check if a URL has been visited
static int has_visited(const char* url) {
    for (int i = 0; i < visited_count; i++) {
        if (strcmp(visited_urls[i], url) == 0) {
            return 1;
        }
    }
    return 0;
}

// Function to mark a URL as visited
static void mark_visited(const char* url) {
    if (visited_count < MAX_URLS) {
        strncpy(visited_urls[visited_count], url, MAX_URL_LENGTH - 1);
        visited_urls[visited_count][MAX_URL_LENGTH - 1] = '\0';
        visited_count++;
    }
}

// Function to extract the base domain from a URL
static char* extract_base_domain(const char* url) {
    static char domain[MAX_URL_LENGTH];
    
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
    static char normalized[MAX_URL_LENGTH * 2];
    strncpy(normalized, url, MAX_URL_LENGTH * 2 - 1);
    normalized[MAX_URL_LENGTH * 2 - 1] = '\0';
    
    // Remove fragment identifiers (#)
    char* fragment = strchr(normalized, '#');
    if (fragment) *fragment = '\0';
    
    // Remove common tracking parameters
    // This is a simplified approach - a more comprehensive solution would parse the URL properly
    char* query = strchr(normalized, '?');
    if (query) {
        // For simplicity, we'll just keep essential parameters
        // For medium.com, often the important part is before the query string
        if (strstr(normalized, "medium.com") != NULL) {
            *query = '\0';  // For medium.com, remove all query params
        }
    }
    
    // Ensure the URL doesn't end with a slash (for consistency)
    size_t len = strlen(normalized);
    if (len > 0 && normalized[len-1] == '/') {
        normalized[len-1] = '\0';
    }
    
    return normalized;
}

// Function to extract links from HTML content
static void extract_links(const char* html, const char* base_url, char** urls, int* url_count, int max_urls) {
    const char* ptr = html;
    const char* href;
    const char* base_domain = extract_base_domain(base_url);
    
    // Search for both href=" and href=' patterns
    while (*ptr && *url_count < max_urls) {
        // Look for anchor tags with href attributes (handle both quote styles)
        const char* href_double = strstr(ptr, "href=\"");
        const char* href_single = strstr(ptr, "href='");
        
        // Find which quote style appears first
        if (href_double && href_single) {
            href = (href_double < href_single) ? href_double : href_single;
            char quote = (href == href_double) ? '\"' : '\'';
            ptr = href + 6; // Move past "href=" and the quote
            const char* end = strchr(ptr, quote);
            
            if (end) {
                int url_len = end - ptr;
                if (url_len > 0 && url_len < MAX_URL_LENGTH) {
                    char* new_url = malloc(MAX_URL_LENGTH);
                    if (!new_url) continue;
                    
                    // Copy the URL
                    strncpy(new_url, ptr, url_len);
                    new_url[url_len] = '\0';
                    
                    // Skip javascript: urls, mailto: links, tel: links, data: URIs
                    if (strncmp(new_url, "javascript:", 11) == 0 ||
                        strncmp(new_url, "mailto:", 7) == 0 ||
                        strncmp(new_url, "tel:", 4) == 0 ||
                        strncmp(new_url, "data:", 5) == 0) {
                        free(new_url);
                        ptr = end + 1;
                        continue;
                    }
                    
                    // Handle relative URLs
                    if (strncmp(new_url, "http", 4) != 0) {
                        // Skip URLs starting with # (page anchors)
                        if (new_url[0] == '#') {
                            free(new_url);
                            ptr = end + 1;
                            continue;
                        }
                        
                        // Convert relative URL to absolute URL
                        char* absolute_url = malloc(MAX_URL_LENGTH * 2);
                        if (!absolute_url) {
                            free(new_url);
                            ptr = end + 1;
                            continue;
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
                    free(new_url);
                    new_url = strdup(normalized_url);
                    
                    // Special handling for medium.com URLs
                    if (strstr(new_url, "medium.com") != NULL) {
                        // Check if it's a user profile or article
                        if (strstr(new_url, "medium.com/@") != NULL || 
                            strstr(new_url, "medium.com/p/") != NULL ||
                            strstr(new_url, "medium.com/tag/") != NULL) {
                            // It's valid, keep it
                        }
                        // Else could add more filters here
                    }
                    
                    // Check if the URL belongs to the same site
                    // For medium.com, allow crawling within medium.com
                    if (strstr(base_url, "medium.com") != NULL && strstr(new_url, "medium.com") != NULL) {
                        urls[*url_count] = new_url;
                        (*url_count)++;
                    } 
                    // For other sites, use stricter domain checking
                    else if (strstr(new_url, base_domain) != NULL) {
                        urls[*url_count] = new_url;
                        (*url_count)++;
                    } else {
                        free(new_url);
                    }
                }
                
                ptr = end + 1;
            } else {
                ptr++; // No end quote found, move forward
            }
        } else if (href_double) {
            href = href_double;
            ptr = href + 6;
            const char* end = strchr(ptr, '\"');
            // Process as before for double quotes
            // Similar processing as above
            if (end) {
                int url_len = end - ptr;
                if (url_len > 0 && url_len < MAX_URL_LENGTH) {
                    // Process URL (similar to above)
                    char* new_url = malloc(MAX_URL_LENGTH);
                    if (!new_url) {
                        ptr = end + 1;
                        continue;
                    }
                    
                    // Copy the URL
                    strncpy(new_url, ptr, url_len);
                    new_url[url_len] = '\0';
                    
                    // Skip javascript:, mailto:, etc.
                    if (strncmp(new_url, "javascript:", 11) == 0 ||
                        strncmp(new_url, "mailto:", 7) == 0 ||
                        strncmp(new_url, "tel:", 4) == 0 ||
                        strncmp(new_url, "data:", 5) == 0) {
                        free(new_url);
                        ptr = end + 1;
                        continue;
                    }
                    
                    // Process relative URLs (similar to above)
                    if (strncmp(new_url, "http", 4) != 0) {
                        // Same relative URL processing as above
                        if (new_url[0] == '#') {
                            free(new_url);
                            ptr = end + 1;
                            continue;
                        }
                        
                        char* absolute_url = malloc(MAX_URL_LENGTH * 2);
                        if (!absolute_url) {
                            free(new_url);
                            ptr = end + 1;
                            continue;
                        }
                        
                        // Same URL resolution as above
                        if (new_url[0] == '/') {
                            if (new_url[1] == '/') {
                                // Protocol-relative URL
                                const char* protocol_end = strstr(base_url, "://");
                                if (protocol_end) {
                                    int protocol_len = protocol_end - base_url + 1;
                                    strncpy(absolute_url, base_url, protocol_len);
                                    absolute_url[protocol_len] = '\0';
                                    strcat(absolute_url, new_url + 2);
                                } else {
                                    sprintf(absolute_url, "https:%s", new_url);
                                }
                            } else {
                                sprintf(absolute_url, "%s%s", base_domain, new_url);
                            }
                        } else {
                            // Relative to current page
                            strcpy(absolute_url, base_url);
                            char* last_slash = strrchr(absolute_url, '/');
                            if (last_slash && last_slash != absolute_url + strlen(absolute_url) - 1) {
                                *(last_slash + 1) = '\0';
                            } else if (!last_slash) {
                                strcat(absolute_url, "/");
                            }
                            strcat(absolute_url, new_url);
                        }
                        
                        free(new_url);
                        new_url = absolute_url;
                    }
                    
                    // Normalize URL
                    char* normalized_url = normalize_url(new_url);
                    free(new_url);
                    new_url = strdup(normalized_url);
                    
                    // Domain check (same as above)
                    if (strstr(base_url, "medium.com") != NULL && strstr(new_url, "medium.com") != NULL) {
                        urls[*url_count] = new_url;
                        (*url_count)++;
                    } else if (strstr(new_url, base_domain) != NULL) {
                        urls[*url_count] = new_url;
                        (*url_count)++;
                    } else {
                        free(new_url);
                    }
                }
                ptr = end + 1;
            } else {
                ptr++;
            }
        } else if (href_single) {
            href = href_single;
            ptr = href + 6;
            const char* end = strchr(ptr, '\'');
            // Process as before for single quotes
            // Similar processing as above
            if (end) {
                // Similar URL processing as above
                ptr = end + 1;
            } else {
                ptr++;
            }
        } else {
            // No more href found
            break;
        }
    }
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
        // Allow profile pages and article pages
        if (strstr(url, "medium.com/@") != NULL ||       // Profile pages
            strstr(url, "/p/") != NULL ||                // Article pages
            strstr(url, "/tag/") != NULL ||              // Tag pages
            strstr(url, "/topics/") != NULL ||           // Topic pages
            strstr(url, "medium.com/") != NULL) {        // Publication pages
            return 1;
        }
    } else if (strstr(url, base_domain) != NULL) {
        // For other domains, require that the URL contains our base domain
        return 1;
    }
    
    return 0;
}

// Function to recursively crawl a website starting from a URL
int crawl_website(const char* start_url, int maxDepth, int maxPages) {
    // Reset the visited URLs
    visited_count = 0;
    
    // Initialize the queue with the start URL
    char* queue[MAX_URLS];
    int depth[MAX_URLS];  // Track depth of each URL
    int front = 0, rear = 0, pages_crawled = 0;
    int failed_downloads = 0;
    
    // Normalize and add start URL to queue
    char* normalized_start_url = normalize_url(start_url);
    queue[rear] = strdup(normalized_start_url);
    depth[rear] = 1;
    rear = (rear + 1) % MAX_URLS;
    
    mark_visited(normalized_start_url);
    
    printf("Starting crawl from: %s (max depth: %d, max pages: %d)\n", normalized_start_url, maxDepth, maxPages);
    
    // Extract base domain from start_url
    char* base_domain = extract_base_domain(start_url);
    printf("Base domain for crawling: %s\n", base_domain);
    
    // Initialize curl globally (once)
    curl_global_init(CURL_GLOBAL_DEFAULT);
    
    // Start crawling
    while (front != rear && pages_crawled < maxPages && failed_downloads < 10) {
        // Get a URL from the queue
        char* current_url = queue[front];
        int current_depth = depth[front];
        front = (front + 1) % MAX_URLS;
        
        // Skip invalid URLs or already processed ones
        if (!is_valid_crawl_url(current_url, base_domain)) {
            printf("  Skipping invalid URL: %s\n", current_url);
            free(current_url);
            continue;
        }
        
        printf("\nCrawling [%d/%d]: %s (depth %d/%d)\n", pages_crawled + 1, maxPages, current_url, current_depth, maxDepth);
        
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
                // Save the content to a file
                char* filename = download_url(current_url);
                if (filename) {
                    printf("  Downloaded to %s (%zu bytes)\n", filename, chunk.size);
                    pages_crawled++;
                    failed_downloads = 0;  // Reset consecutive failure counter
                    
                    // If we have not reached max depth, extract links and add to queue
                    if (current_depth < maxDepth) {
                        char* extracted_urls[MAX_URLS];
                        int url_count = 0;
                        
                        // Pass the current URL as base URL for relative link resolution
                        extract_links(chunk.memory, current_url, extracted_urls, &url_count, MAX_URLS);
                        printf("  Found %d links\n", url_count);
                        
                        // Add new URLs to the queue
                        int added_urls = 0;
                        for (int i = 0; i < url_count && rear != (front - 1 + MAX_URLS) % MAX_URLS && added_urls < 20; i++) {
                            if (!has_visited(extracted_urls[i]) && is_valid_crawl_url(extracted_urls[i], base_domain)) {
                                queue[rear] = extracted_urls[i];
                                depth[rear] = current_depth + 1;
                                rear = (rear + 1) % MAX_URLS;
                                
                                mark_visited(extracted_urls[i]);
                                printf("  Queued: %s\n", extracted_urls[i]);
                                added_urls++;
                            } else {
                                free(extracted_urls[i]); // Free if already visited or invalid
                            }
                        }
                        
                        // Free any remaining URLs
                        for (int i = added_urls; i < url_count; i++) {
                            free(extracted_urls[i]);
                        }
                    }
                } else {
                    failed_downloads++;
                    printf("  Failed to save content from: %s\n", current_url);
                }
            } else {
                failed_downloads++;
                fprintf(stderr, "  Failed to download: %s - %s\n", current_url, 
                        (res != CURLE_OK) ? curl_easy_strerror(res) : "Content too small");
            }
            
            free(chunk.memory);
        } else {
            failed_downloads++;
            fprintf(stderr, "  Failed to initialize curl for: %s\n", current_url);
        }
        
        // Add a small delay between requests to be nice to servers (200-500ms)
        usleep((rand() % 300 + 200) * 1000);
        
        free(current_url);
    }
    
    // Clean up curl global state
    curl_global_cleanup();
    
    printf("\nCrawling completed. Crawled %d pages.\n", pages_crawled);
    return pages_crawled;
}
