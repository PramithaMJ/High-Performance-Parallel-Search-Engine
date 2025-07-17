#ifndef PARSER_H
#define PARSER_H

#include "hybrid_engine.h"

#ifdef USE_CUDA
#include "cuda_kernels.h"
#endif

#define MAX_FILE_SIZE (10 * 1024 * 1024)  // 10MB max file size
#define MAX_LINE_LENGTH 4096
#define MAX_TOKEN_LENGTH 128
#define MAX_TOKENS_PER_DOC 10000

// Parsing configuration
typedef struct {
    int remove_punctuation;
    int convert_lowercase;
    int remove_stopwords;
    int min_token_length;
    int max_token_length;
    int max_tokens_per_document;
    int enable_parallel_parsing;
    int use_gpu_tokenization;
} parser_config_t;

// Token structure for efficient processing
typedef struct __attribute__((aligned(8))) {
    char text[MAX_TOKEN_LENGTH];
    int frequency;
    int position;
    float weight;
} token_t;

// Document parsing result
typedef struct {
    int doc_id;
    char* content;
    size_t content_length;
    token_t* tokens;
    int token_count;
    float parsing_time;
    int parsed_on_gpu;
} parse_result_t;

// Global parser configuration
extern parser_config_t g_parser_config;

// Core parsing functions
int parse_file(const char *filepath, int doc_id);
int parse_file_parallel(const char *filepath, int doc_id);
int parse_file_hybrid(const char *filepath, int doc_id, int use_gpu);

// Content parsing functions
parse_result_t* parse_content(const char* content, size_t length, int doc_id);
parse_result_t* parse_content_parallel(const char* content, size_t length, int doc_id);

// Tokenization functions
void tokenize(char *text, int doc_id);
void tokenize_parallel(char *text, int doc_id, int num_threads);
int tokenize_content(const char* content, size_t length, token_t* tokens, int max_tokens);

#ifdef USE_CUDA
// GPU-accelerated parsing functions
parse_result_t* parse_content_gpu(const char* content, size_t length, int doc_id);
int tokenize_content_gpu(const char* content, size_t length, token_t* tokens, int max_tokens);
void transfer_tokens_to_gpu(token_t* host_tokens, int count);
void transfer_tokens_from_gpu(token_t* host_tokens, int count);
#endif

// Text preprocessing functions
void to_lowercase(char *str);
void to_lowercase_parallel(char *str, size_t length);
void remove_punctuation(char *str);
void remove_extra_whitespace(char *str);
int is_valid_token(const char* token);

// File I/O functions
char* read_file_content(const char* filepath, size_t* file_size);
char* read_file_content_parallel(const char* filepath, size_t* file_size);
int write_parsed_content(const char* filepath, parse_result_t* result);

// Batch processing functions
int parse_multiple_files(char** filepaths, int num_files, parse_result_t** results);
int parse_directory(const char* directory_path, parse_result_t** results, int* result_count);
int parse_directory_parallel(const char* directory_path, parse_result_t** results, int* result_count);

// HTML/Web content parsing
int parse_html_content(const char* html, char** text_content);
int extract_text_from_html(const char* html, char* text_buffer, size_t buffer_size);
int parse_url_content(const char* url, parse_result_t** result);

// Document format detection and parsing
typedef enum {
    FORMAT_UNKNOWN = 0,
    FORMAT_PLAIN_TEXT,
    FORMAT_HTML,
    FORMAT_XML,
    FORMAT_JSON,
    FORMAT_CSV,
    FORMAT_PDF,
    FORMAT_DOC
} document_format_t;

document_format_t detect_document_format(const char* filepath);
document_format_t detect_content_format(const char* content, size_t length);
int parse_document_by_format(const char* filepath, document_format_t format, parse_result_t** result);

// Performance optimization
void optimize_parsing_performance(void);
void precompile_parsing_patterns(void);
void cache_frequent_tokens(void);

// Memory management
parse_result_t* create_parse_result(int doc_id);
void free_parse_result(parse_result_t* result);
void cleanup_parser_resources(void);

// Configuration functions
void set_parser_config(parser_config_t* config);
parser_config_t* get_parser_config(void);
void load_parser_config_from_file(const char* config_file);
void save_parser_config_to_file(const char* config_file);

// Statistics and monitoring
typedef struct {
    int files_parsed;
    int documents_processed;
    long total_tokens_extracted;
    double total_parsing_time;
    double avg_parsing_time_per_doc;
    size_t total_content_size;
    int gpu_parsing_count;
    int cpu_parsing_count;
    float gpu_acceleration_factor;
} parser_stats_t;

extern parser_stats_t g_parser_stats;

parser_stats_t* get_parser_statistics(void);
void reset_parser_statistics(void);
void print_parser_performance_report(void);

// Error handling
typedef enum {
    PARSER_SUCCESS = 0,
    PARSER_ERROR_FILE_NOT_FOUND = -1,
    PARSER_ERROR_FILE_TOO_LARGE = -2,
    PARSER_ERROR_MEMORY = -3,
    PARSER_ERROR_FORMAT = -4,
    PARSER_ERROR_GPU = -5
} parser_error_t;

const char* parser_get_error_string(parser_error_t error);

// Thread safety
void init_parser_locks(void);
void destroy_parser_locks(void);
void lock_parser_resources(void);
void unlock_parser_resources(void);

// Advanced features
int parse_with_metadata_extraction(const char* filepath, parse_result_t** result);
int parse_with_language_detection(const char* content, parse_result_t** result);
int parse_with_encoding_detection(const char* filepath, parse_result_t** result);

// Validation and debugging
int validate_parse_result(parse_result_t* result);
void print_parse_debug_info(parse_result_t* result);
int compare_cpu_gpu_parsing_results(const char* content, float tolerance);

// Utility functions
size_t estimate_parsing_memory_requirement(const char* filepath);
int is_parsing_supported_format(const char* filepath);
char* get_file_extension(const char* filepath);
int count_tokens_in_content(const char* content, size_t length);

#endif // PARSER_H
