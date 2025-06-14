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

// A more robust HTML to text conversion
static void html_to_text(const char *html, FILE *output) {
    int in_tag = 0;
    int in_script = 0;
    int in_style = 0;
    int in_head = 0;
    int space_needed = 0;
    int consecutive_spaces = 0;
    size_t html_len = strlen(html);
    
    // First, look for the body tag to skip header content
    const char *body_start = strstr(html, "<body");
    if (body_start) {
        html = body_start;
    }
    
    for (size_t i = 0; html[i]; i++) {
        // Check for key HTML sections
        if (!in_tag && (i + 6 < html_len) && strncasecmp(&html[i], "<head>", 6) == 0) {
            in_head = 1;
            in_tag = 1;
            continue;
        }
        else if (in_head && (i + 7 < html_len) && strncasecmp(&html[i], "</head>", 7) == 0) {
            in_head = 0;
            in_tag = 1;
            i += 6;  // Skip to end of tag
            continue;
        }
        else if (!in_tag && (i + 8 < html_len) && strncasecmp(&html[i], "<script", 7) == 0) {
            in_script = 1;
            in_tag = 1;
        }
        else if (!in_tag && (i + 7 < html_len) && strncasecmp(&html[i], "<style", 6) == 0) {
            in_style = 1;
            in_tag = 1;
        }
        else if (in_script && (i + 9 < html_len) && strncasecmp(&html[i], "</script>", 9) == 0) {
            in_script = 0;
            i += 8;  // Skip to end of tag
            continue;
        }
        else if (in_style && (i + 8 < html_len) && strncasecmp(&html[i], "</style>", 8) == 0) {
            in_style = 0;
            i += 7;  // Skip to end of tag
            continue;
        }
        
        // Skip content in head, script and style sections
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
            if ((i + 4 < html_len) && (strncasecmp(&html[i], "<p>", 3) == 0 || 
                                       strncasecmp(&html[i], "<br", 3) == 0 ||
                                       strncasecmp(&html[i], "<li", 3) == 0 ||
                                       strncasecmp(&html[i], "<h", 2) == 0)) {
                fprintf(output, "\n\n");
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
        
        // Handle regular text content
        if (isspace((unsigned char)html[i])) {
            if (consecutive_spaces == 0) {
                fputc(' ', output);
                consecutive_spaces = 1;
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
        }
    }
}

// Function to download a URL and save it to the dataset directory
char* download_url(const char* url) {
    CURL *curl;
    CURLcode res;
    FILE *file;
    static char filepath[512];
    char* filename;
    struct MemoryStruct chunk;
    
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
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "SearchEngine/1.0");
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
    
    // Perform the request
    printf("Downloading %s...\n", url);
    res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    
    if (res != CURLE_OK) {
        fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
        free(chunk.memory);
        return NULL;
    }
    
    // Ensure dataset directory exists
    ensure_dataset_directory();
    
    // Create a unique filename from the URL
    filename = get_url_filename(url);
    snprintf(filepath, sizeof(filepath), "dataset/%s", filename);
    
    // Open file for writing
    file = fopen(filepath, "wb");
    if (!file) {
        fprintf(stderr, "Failed to open file for writing: %s\n", filepath);
        free(chunk.memory);
        return NULL;
    }
    
    // Convert HTML to text and save to file
    printf("Processing HTML content...\n");
    html_to_text(chunk.memory, file);
    
    // Clean up
    fclose(file);
    free(chunk.memory);
    
    printf("Downloaded and processed to %s\n", filepath);
    return filepath;
}
