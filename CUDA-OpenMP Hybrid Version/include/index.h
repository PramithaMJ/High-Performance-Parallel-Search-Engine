#ifndef INDEX_H
#define INDEX_H

#include "hybrid_engine.h"

#ifdef USE_CUDA
#include "cuda_kernels.h"
#endif

#define MAX_FILENAME_LEN 256
#define MAX_TERM_LENGTH 64
#define MAX_POSTINGS 1000
#define MAX_INDEX_SIZE 10000

// Document structure to store file names and metadata
typedef struct {
    char filename[MAX_FILENAME_LEN];
    int processed_on_gpu;  // Flag for hybrid processing tracking
    size_t content_hash;   // Hash for duplicate detection
    float processing_time; // Time taken to process this document
} Document;

// Posting structure with GPU alignment
typedef struct __attribute__((aligned(8))) {
    int doc_id;
    int freq;
    float tf_score;        // Pre-calculated TF score for faster GPU access
} Posting;

// Inverted index structure optimized for GPU transfer
typedef struct __attribute__((aligned(16))) {
    char term[MAX_TERM_LENGTH];
    Posting postings[MAX_POSTINGS];
    int posting_count;
    float idf_score;       // Pre-calculated IDF score
    int gpu_memory_offset; // Offset in GPU memory (if loaded)
    int last_access_time;  // For memory management
} InvertedIndex;

// GPU-optimized index structure
typedef struct {
    float* gpu_tf_scores;    // GPU array of TF scores
    float* gpu_idf_scores;   // GPU array of IDF scores
    int* gpu_doc_ids;        // GPU array of document IDs
    int* gpu_term_offsets;   // GPU array of term offsets
    char* gpu_terms;         // GPU array of terms (concatenated)
    int gpu_memory_allocated; // Flag indicating GPU memory state
    size_t gpu_memory_size;   // Size of allocated GPU memory
} GPUIndex;

// Global variables
extern InvertedIndex index_data[MAX_INDEX_SIZE];
extern int index_size;
extern Document documents[1000];
extern int doc_count;
extern GPUIndex gpu_index;

// Core indexing functions
int build_index(const char *folder_path);
int build_index_hybrid(const char *folder_path, int use_gpu);
void add_token(const char *token, int doc_id);
void add_token_gpu_optimized(const char *token, int doc_id, float tf_score);

// Document management
int get_doc_length(int doc_id);
int get_doc_count(void);
const char* get_doc_filename(int doc_id);
void synchronize_document_filenames(void);

// Index operations
void print_index(void);
void print_all_index_terms(void);
void clear_index(void);
void clear_gpu_index(void);

// GPU-specific functions
#ifdef USE_CUDA
int transfer_index_to_gpu(void);
int transfer_index_from_gpu(void);
void update_gpu_index_async(void);
int is_index_on_gpu(void);
void optimize_gpu_index_layout(void);
#endif

// Hybrid processing functions
int build_index_parallel_hybrid(const char *folder_path);
void distribute_indexing_workload(const char *folder_path, int num_cpu_threads);
int auto_select_processing_mode(int num_documents, size_t total_size);

// Performance optimization
void precompute_tf_idf_scores(void);
void optimize_index_layout(void);
int compress_index(float compression_ratio);
void index_memory_prefetch(int doc_start, int doc_end);

// Search optimization functions
int parallel_search_term(const char *term, Posting **results, int *result_count);
int gpu_search_term(const char *term, Posting **results, int *result_count);
int hybrid_search_term(const char *term, Posting **results, int *result_count);

// Index statistics and monitoring
typedef struct {
    int total_terms;
    int total_postings;
    float avg_postings_per_term;
    size_t memory_usage_cpu;
    size_t memory_usage_gpu;
    float index_density;
    double build_time_cpu;
    double build_time_gpu;
    float compression_ratio;
} IndexStats;

IndexStats* get_index_statistics(void);
void print_index_performance_report(void);

// Thread safety for hybrid processing
void init_index_locks(void);
void destroy_index_locks(void);
void lock_index_read(void);
void unlock_index_read(void);
void lock_index_write(void);
void unlock_index_write(void);

// Memory management
size_t get_index_memory_usage(void);
int optimize_index_memory_layout(void);
void cleanup_index_resources(void);

// Validation and debugging
int validate_index_integrity(void);
void print_index_debug_info(void);
int compare_cpu_gpu_index_results(const char *term);

// Configuration
typedef struct {
    int use_gpu;
    int gpu_batch_size;
    int cpu_threads;
    float gpu_memory_limit_gb;
    int enable_compression;
    int enable_prefetching;
    int auto_optimize;
} IndexConfig;

extern IndexConfig g_index_config;

void set_index_config(IndexConfig* config);
IndexConfig* get_index_config(void);
void load_index_config_from_file(const char* filename);

#endif // INDEX_H
