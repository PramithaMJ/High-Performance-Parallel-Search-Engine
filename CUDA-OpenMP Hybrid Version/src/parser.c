#include "parser.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <sys/stat.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#define F_OK 0
#define access _access
#else
#include <unistd.h>
#include <dirent.h>
#endif

// Global variables
parser_config_t g_parser_config = {
    .remove_punctuation = 1,
    .convert_lowercase = 1,
    .remove_stopwords = 1,
    .min_token_length = 2,
    .max_token_length = MAX_TOKEN_LENGTH - 1,
    .max_tokens_per_document = MAX_TOKENS_PER_DOC,
    .enable_parallel_parsing = 1,
    .use_gpu_tokenization = 1
};

parser_stats_t g_parser_stats = {0};

// Thread safety variables
#ifdef _OPENMP
static omp_lock_t parser_lock;
static int lock_initialized = 0;
#endif

// Common stop words
const char* stop_words[] = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
    "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "will", "with",
    NULL
};

// Initialize parser locks
void init_parser_locks(void) {
#ifdef _OPENMP
    if (!lock_initialized) {
        omp_init_lock(&parser_lock);
        lock_initialized = 1;
    }
#endif
}

// Destroy parser locks
void destroy_parser_locks(void) {
#ifdef _OPENMP
    if (lock_initialized) {
        omp_destroy_lock(&parser_lock);
        lock_initialized = 0;
    }
#endif
}

// Lock parser resources
void lock_parser_resources(void) {
#ifdef _OPENMP
    if (lock_initialized) {
        omp_set_lock(&parser_lock);
    }
#endif
}

// Unlock parser resources
void unlock_parser_resources(void) {
#ifdef _OPENMP
    if (lock_initialized) {
        omp_unset_lock(&parser_lock);
    }
#endif
}

// Check if token is a stop word
int is_stop_word(const char* token) {
    for (int i = 0; stop_words[i] != NULL; i++) {
        if (strcmp(token, stop_words[i]) == 0) {
            return 1;
        }
    }
    return 0;
}

// Convert string to lowercase
void to_lowercase(char *str) {
    if (!str) return;
    
    for (int i = 0; str[i]; i++) {
        str[i] = tolower(str[i]);
    }
}

// Parallel lowercase conversion
void to_lowercase_parallel(char *str, size_t length) {
    if (!str || length == 0) return;
    
#ifdef _OPENMP
    #pragma omp parallel for
    for (size_t i = 0; i < length; i++) {
        str[i] = tolower(str[i]);
    }
#else
    to_lowercase(str);
#endif
}

// Remove punctuation from string
void remove_punctuation(char *str) {
    if (!str) return;
    
    int i, j = 0;
    for (i = 0; str[i]; i++) {
        if (isalnum(str[i]) || isspace(str[i])) {
            str[j++] = str[i];
        }
    }
    str[j] = '\0';
}

// Remove extra whitespace
void remove_extra_whitespace(char *str) {
    if (!str) return;
    
    int i, j = 0;
    int space_found = 0;
    
    for (i = 0; str[i]; i++) {
        if (isspace(str[i])) {
            if (!space_found) {
                str[j++] = ' ';
                space_found = 1;
            }
        } else {
            str[j++] = str[i];
            space_found = 0;
        }
    }
    
    // Remove trailing space
    if (j > 0 && str[j-1] == ' ') {
        j--;
    }
    str[j] = '\0';
}

// Check if token is valid
int is_valid_token(const char* token) {
    if (!token || strlen(token) < g_parser_config.min_token_length) {
        return 0;
    }
    
    if (strlen(token) > g_parser_config.max_token_length) {
        return 0;
    }
    
    if (g_parser_config.remove_stopwords && is_stop_word(token)) {
        return 0;
    }
    
    // Check if token contains at least one letter
    int has_letter = 0;
    for (int i = 0; token[i]; i++) {
        if (isalpha(token[i])) {
            has_letter = 1;
            break;
        }
    }
    
    return has_letter;
}

// Read file content into memory
char* read_file_content(const char* filepath, size_t* file_size) {
    FILE* file = fopen(filepath, "rb");
    if (!file) {
        fprintf(stderr, "Error: Cannot open file %s\n", filepath);
        return NULL;
    }
    
    // Get file size
    fseek(file, 0, SEEK_END);
    *file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    // Check file size limit
    if (*file_size > MAX_FILE_SIZE) {
        fprintf(stderr, "Error: File %s is too large (%zu bytes)\n", filepath, *file_size);
        fclose(file);
        return NULL;
    }
    
    // Allocate memory for content
    char* content = malloc(*file_size + 1);
    if (!content) {
        fprintf(stderr, "Error: Memory allocation failed for file %s\n", filepath);
        fclose(file);
        return NULL;
    }
    
    // Read file content
    size_t bytes_read = fread(content, 1, *file_size, file);
    if (bytes_read != *file_size) {
        fprintf(stderr, "Error: Failed to read complete file %s\n", filepath);
        free(content);
        fclose(file);
        return NULL;
    }
    
    content[*file_size] = '\0';
    fclose(file);
    
    return content;
}

// Tokenize content into tokens array
int tokenize_content(const char* content, size_t length, token_t* tokens, int max_tokens) {
    if (!content || !tokens || max_tokens <= 0) {
        return 0;
    }
    
    char* content_copy = malloc(length + 1);
    if (!content_copy) {
        return 0;
    }
    
    strcpy(content_copy, content);
    
    // Apply preprocessing
    if (g_parser_config.convert_lowercase) {
        to_lowercase(content_copy);
    }
    
    if (g_parser_config.remove_punctuation) {
        remove_punctuation(content_copy);
    }
    
    remove_extra_whitespace(content_copy);
    
    // Tokenize
    int token_count = 0;
    char* token = strtok(content_copy, " \t\n\r");
    int position = 0;
    
    while (token && token_count < max_tokens) {
        if (is_valid_token(token)) {
            strncpy(tokens[token_count].text, token, MAX_TOKEN_LENGTH - 1);
            tokens[token_count].text[MAX_TOKEN_LENGTH - 1] = '\0';
            tokens[token_count].frequency = 1;
            tokens[token_count].position = position;
            tokens[token_count].weight = 1.0f;
            token_count++;
        }
        token = strtok(NULL, " \t\n\r");
        position++;
    }
    
    free(content_copy);
    return token_count;
}

// Create parse result structure
parse_result_t* create_parse_result(int doc_id) {
    parse_result_t* result = malloc(sizeof(parse_result_t));
    if (!result) {
        return NULL;
    }
    
    result->doc_id = doc_id;
    result->content = NULL;
    result->content_length = 0;
    result->tokens = malloc(MAX_TOKENS_PER_DOC * sizeof(token_t));
    result->token_count = 0;
    result->parsing_time = 0.0f;
    result->parsed_on_gpu = 0;
    
    if (!result->tokens) {
        free(result);
        return NULL;
    }
    
    return result;
}

// Free parse result structure
void free_parse_result(parse_result_t* result) {
    if (!result) return;
    
    if (result->content) {
        free(result->content);
    }
    
    if (result->tokens) {
        free(result->tokens);
    }
    
    free(result);
}

// Parse content and return result
parse_result_t* parse_content(const char* content, size_t length, int doc_id) {
    if (!content || length == 0) {
        return NULL;
    }
    
    clock_t start_time = clock();
    
    parse_result_t* result = create_parse_result(doc_id);
    if (!result) {
        return NULL;
    }
    
    // Store content copy
    result->content = malloc(length + 1);
    if (!result->content) {
        free_parse_result(result);
        return NULL;
    }
    strcpy(result->content, content);
    result->content_length = length;
    
    // Tokenize content
    result->token_count = tokenize_content(content, length, result->tokens, MAX_TOKENS_PER_DOC);
    
    clock_t end_time = clock();
    result->parsing_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    
    // Update statistics
    lock_parser_resources();
    g_parser_stats.documents_processed++;
    g_parser_stats.total_tokens_extracted += result->token_count;
    g_parser_stats.total_parsing_time += result->parsing_time;
    g_parser_stats.total_content_size += length;
    g_parser_stats.cpu_parsing_count++;
    unlock_parser_resources();
    
    return result;
}

// Parse content with parallel processing
parse_result_t* parse_content_parallel(const char* content, size_t length, int doc_id) {
    if (!g_parser_config.enable_parallel_parsing) {
        return parse_content(content, length, doc_id);
    }
    
#ifdef USE_CUDA
    if (g_parser_config.use_gpu_tokenization) {
        return parse_content_gpu(content, length, doc_id);
    }
#endif
    
    // For now, use regular parsing - parallel tokenization would require more complex implementation
    return parse_content(content, length, doc_id);
}

// Parse file
int parse_file(const char *filepath, int doc_id) {
    if (!filepath || access(filepath, F_OK) != 0) {
        return PARSER_ERROR_FILE_NOT_FOUND;
    }
    
    size_t file_size;
    char* content = read_file_content(filepath, &file_size);
    if (!content) {
        return PARSER_ERROR_MEMORY;
    }
    
    parse_result_t* result = parse_content(content, file_size, doc_id);
    free(content);
    
    if (!result) {
        return PARSER_ERROR_MEMORY;
    }
    
    // Store result in document database (would need integration with index system)
    // For now, just update statistics and free result
    
    lock_parser_resources();
    g_parser_stats.files_parsed++;
    unlock_parser_resources();
    
    free_parse_result(result);
    return PARSER_SUCCESS;
}

// Parse file with hybrid GPU/CPU approach
int parse_file_hybrid(const char *filepath, int doc_id, int use_gpu) {
    if (!filepath || access(filepath, F_OK) != 0) {
        return PARSER_ERROR_FILE_NOT_FOUND;
    }
    
    size_t file_size;
    char* content = read_file_content(filepath, &file_size);
    if (!content) {
        return PARSER_ERROR_MEMORY;
    }
    
    parse_result_t* result;
    
#ifdef USE_CUDA
    if (use_gpu && g_parser_config.use_gpu_tokenization) {
        result = parse_content_gpu(content, file_size, doc_id);
    } else {
        result = parse_content(content, file_size, doc_id);
    }
#else
    result = parse_content(content, file_size, doc_id);
#endif
    
    free(content);
    
    if (!result) {
        return PARSER_ERROR_MEMORY;
    }
    
    lock_parser_resources();
    g_parser_stats.files_parsed++;
    unlock_parser_resources();
    
    free_parse_result(result);
    return PARSER_SUCCESS;
}

// Detect document format
document_format_t detect_document_format(const char* filepath) {
    if (!filepath) return FORMAT_UNKNOWN;
    
    char* ext = get_file_extension(filepath);
    if (!ext) return FORMAT_PLAIN_TEXT;
    
    // Convert extension to lowercase
    to_lowercase(ext);
    
    if (strcmp(ext, "html") == 0 || strcmp(ext, "htm") == 0) {
        return FORMAT_HTML;
    } else if (strcmp(ext, "xml") == 0) {
        return FORMAT_XML;
    } else if (strcmp(ext, "json") == 0) {
        return FORMAT_JSON;
    } else if (strcmp(ext, "csv") == 0) {
        return FORMAT_CSV;
    } else if (strcmp(ext, "pdf") == 0) {
        return FORMAT_PDF;
    } else if (strcmp(ext, "doc") == 0 || strcmp(ext, "docx") == 0) {
        return FORMAT_DOC;
    }
    
    return FORMAT_PLAIN_TEXT;
}

// Get file extension
char* get_file_extension(const char* filepath) {
    if (!filepath) return NULL;
    
    char* dot = strrchr(filepath, '.');
    if (!dot || dot == filepath) return NULL;
    
    return dot + 1;
}

// Get parser configuration
parser_config_t* get_parser_config(void) {
    return &g_parser_config;
}

// Set parser configuration
void set_parser_config(parser_config_t* config) {
    if (config) {
        g_parser_config = *config;
    }
}

// Get parser statistics
parser_stats_t* get_parser_statistics(void) {
    lock_parser_resources();
    
    // Calculate derived statistics
    if (g_parser_stats.documents_processed > 0) {
        g_parser_stats.avg_parsing_time_per_doc = 
            g_parser_stats.total_parsing_time / g_parser_stats.documents_processed;
    }
    
    if (g_parser_stats.cpu_parsing_count > 0 && g_parser_stats.gpu_parsing_count > 0) {
        double cpu_avg_time = g_parser_stats.total_parsing_time / g_parser_stats.cpu_parsing_count;
        double gpu_avg_time = g_parser_stats.total_parsing_time / g_parser_stats.gpu_parsing_count;
        if (gpu_avg_time > 0) {
            g_parser_stats.gpu_acceleration_factor = cpu_avg_time / gpu_avg_time;
        }
    }
    
    unlock_parser_resources();
    return &g_parser_stats;
}

// Reset parser statistics
void reset_parser_statistics(void) {
    lock_parser_resources();
    memset(&g_parser_stats, 0, sizeof(parser_stats_t));
    unlock_parser_resources();
}

// Print performance report
void print_parser_performance_report(void) {
    parser_stats_t* stats = get_parser_statistics();
    
    printf("\n=== Parser Performance Report ===\n");
    printf("Files parsed: %d\n", stats->files_parsed);
    printf("Documents processed: %d\n", stats->documents_processed);
    printf("Total tokens extracted: %ld\n", stats->total_tokens_extracted);
    printf("Total content size: %zu bytes\n", stats->total_content_size);
    printf("Total parsing time: %.3f seconds\n", stats->total_parsing_time);
    printf("Average parsing time per document: %.3f ms\n", 
           stats->avg_parsing_time_per_doc * 1000);
    printf("CPU parsing operations: %d\n", stats->cpu_parsing_count);
    printf("GPU parsing operations: %d\n", stats->gpu_parsing_count);
    if (stats->gpu_acceleration_factor > 0) {
        printf("GPU acceleration factor: %.2fx\n", stats->gpu_acceleration_factor);
    }
    printf("================================\n\n");
}

// Get error string
const char* parser_get_error_string(parser_error_t error) {
    switch (error) {
        case PARSER_SUCCESS: return "Success";
        case PARSER_ERROR_FILE_NOT_FOUND: return "File not found";
        case PARSER_ERROR_FILE_TOO_LARGE: return "File too large";
        case PARSER_ERROR_MEMORY: return "Memory allocation error";
        case PARSER_ERROR_FORMAT: return "Unsupported file format";
        case PARSER_ERROR_GPU: return "GPU processing error";
        default: return "Unknown error";
    }
}

// Validate parse result
int validate_parse_result(parse_result_t* result) {
    if (!result) return 0;
    
    if (result->doc_id < 0) return 0;
    if (!result->tokens && result->token_count > 0) return 0;
    if (result->token_count < 0 || result->token_count > MAX_TOKENS_PER_DOC) return 0;
    if (result->parsing_time < 0) return 0;
    
    return 1;
}

// Print debug information
void print_parse_debug_info(parse_result_t* result) {
    if (!result) {
        printf("Parse result is NULL\n");
        return;
    }
    
    printf("=== Parse Debug Info ===\n");
    printf("Document ID: %d\n", result->doc_id);
    printf("Content length: %zu\n", result->content_length);
    printf("Token count: %d\n", result->token_count);
    printf("Parsing time: %.3f ms\n", result->parsing_time * 1000);
    printf("Parsed on GPU: %s\n", result->parsed_on_gpu ? "Yes" : "No");
    
    if (result->token_count > 0 && result->tokens) {
        printf("First 10 tokens:\n");
        int max_tokens = (result->token_count < 10) ? result->token_count : 10;
        for (int i = 0; i < max_tokens; i++) {
            printf("  %d: '%s' (freq: %d, pos: %d, weight: %.2f)\n",
                   i, result->tokens[i].text, result->tokens[i].frequency,
                   result->tokens[i].position, result->tokens[i].weight);
        }
    }
    printf("=======================\n");
}

// Cleanup parser resources
void cleanup_parser_resources(void) {
    destroy_parser_locks();
    reset_parser_statistics();
}

// Initialize parser system
void initialize_parser_system(void) {
    init_parser_locks();
    reset_parser_statistics();
    
#ifdef USE_CUDA
    // Initialize CUDA context for parsing if available
    // This would be implemented in cuda_kernels.cu
#endif
}

// Count tokens in content (utility function)
int count_tokens_in_content(const char* content, size_t length) {
    if (!content || length == 0) return 0;
    
    int token_count = 0;
    int in_token = 0;
    
    for (size_t i = 0; i < length; i++) {
        if (isspace(content[i])) {
            in_token = 0;
        } else {
            if (!in_token) {
                token_count++;
                in_token = 1;
            }
        }
    }
    
    return token_count;
}

// Estimate memory requirement for parsing
size_t estimate_parsing_memory_requirement(const char* filepath) {
    if (!filepath) return 0;
    
    struct stat st;
    if (stat(filepath, &st) != 0) return 0;
    
    size_t file_size = st.st_size;
    size_t estimated_tokens = file_size / 5;  // Rough estimate: 5 chars per token
    
    if (estimated_tokens > MAX_TOKENS_PER_DOC) {
        estimated_tokens = MAX_TOKENS_PER_DOC;
    }
    
    size_t memory_needed = file_size +  // Content storage
                          (estimated_tokens * sizeof(token_t)) +  // Token storage
                          sizeof(parse_result_t) +  // Result structure
                          1024;  // Additional overhead
    
    return memory_needed;
}

// Check if parsing is supported for file format
int is_parsing_supported_format(const char* filepath) {
    document_format_t format = detect_document_format(filepath);
    
    switch (format) {
        case FORMAT_PLAIN_TEXT:
        case FORMAT_HTML:
        case FORMAT_XML:
        case FORMAT_JSON:
        case FORMAT_CSV:
            return 1;
        case FORMAT_PDF:
        case FORMAT_DOC:
            // These would require additional libraries
            return 0;
        default:
            return 0;
    }
}
