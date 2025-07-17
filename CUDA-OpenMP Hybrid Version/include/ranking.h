#ifndef RANKING_H
#define RANKING_H

#include "hybrid_engine.h"
#include "index.h"

#ifdef USE_CUDA
#include "cuda_kernels.h"
#endif

#define BM25_K1 1.2f
#define BM25_B 0.75f
#define MAX_QUERY_TERMS 32
#define MAX_RESULTS 1000

// BM25 parameters structure
typedef struct {
    float k1;              // Term frequency saturation parameter
    float b;               // Length normalization parameter
    float avg_doc_length;  // Average document length in collection
    int total_docs;        // Total number of documents
    int use_gpu;           // Whether to use GPU acceleration
} BM25Params;

// Query term structure for GPU processing
typedef struct __attribute__((aligned(8))) {
    char term[MAX_TERM_LENGTH];
    float idf;             // Inverse document frequency
    float query_tf;        // Query term frequency
    float weight;          // Query term weight
    int doc_frequency;     // Number of documents containing this term
} QueryTerm;

// Result structure optimized for sorting
typedef struct __attribute__((aligned(16))) {
    int doc_id;
    float score;
    float cpu_time;        // Processing time on CPU
    float gpu_time;        // Processing time on GPU
    int processing_mode;   // 0=CPU, 1=GPU, 2=Hybrid
} SearchResult;

// Batch processing structure for GPU
typedef struct {
    QueryTerm* terms;
    int num_terms;
    SearchResult* results;
    int num_results;
    float* partial_scores; // Intermediate scores for GPU reduction
    int* doc_mask;         // Document inclusion mask
} BatchQuery;

// Global BM25 configuration
extern BM25Params g_bm25_params;

// Core ranking functions
void rank_bm25(const char *query, int total_docs, int top_k);
void rank_bm25_hybrid(const char *query, int total_docs, int top_k, int use_gpu);
void rank_bm25_batch(char** queries, int num_queries, int total_docs, int top_k);

// CPU-optimized OpenMP functions
void rank_bm25_cpu_parallel(const char *query, int total_docs, int top_k);
float calculate_bm25_score_cpu(const char* term, int doc_id, BM25Params* params);
void parallel_term_scoring_cpu(QueryTerm* terms, int num_terms, 
                               SearchResult* results, int num_docs);

// GPU-accelerated functions
#ifdef USE_CUDA
void rank_bm25_gpu(const char *query, int total_docs, int top_k);
float calculate_bm25_score_gpu(const char* term, int doc_id, BM25Params* params);
void gpu_batch_scoring(QueryTerm* terms, int num_terms, 
                       SearchResult* results, int num_docs);
int transfer_ranking_data_to_gpu(QueryTerm* terms, int num_terms);
void gpu_parallel_sort_results(SearchResult* results, int num_results);
#endif

// Hybrid processing functions
void rank_bm25_auto_hybrid(const char *query, int total_docs, int top_k);
int select_optimal_processing_mode(const char *query, int total_docs);
void hybrid_workload_distribution(QueryTerm* terms, int num_terms, 
                                  int* cpu_terms, int* gpu_terms);

// Query preprocessing
int parse_query_terms(const char* query, QueryTerm* terms, int max_terms);
void calculate_query_term_weights(QueryTerm* terms, int num_terms);
void normalize_query_scores(QueryTerm* terms, int num_terms);

// Result processing and sorting
void sort_results_parallel(SearchResult* results, int num_results);
void merge_partial_results(SearchResult* cpu_results, int cpu_count,
                          SearchResult* gpu_results, int gpu_count,
                          SearchResult* final_results, int max_results);
void deduplicate_results(SearchResult* results, int* num_results);

// Performance optimization
void precompute_idf_scores(void);
void cache_frequent_queries(const char* query, SearchResult* results, int num_results);
SearchResult* lookup_cached_query(const char* query, int* num_results);
void optimize_ranking_memory_layout(void);

// Statistics and monitoring
typedef struct {
    double total_ranking_time;
    double cpu_ranking_time;
    double gpu_ranking_time;
    double query_parsing_time;
    double result_sorting_time;
    int queries_processed;
    int cache_hits;
    int cache_misses;
    float gpu_acceleration_factor;
    float parallel_efficiency;
} RankingStats;

extern RankingStats g_ranking_stats;

RankingStats* get_ranking_statistics(void);
void reset_ranking_statistics(void);
void print_ranking_performance_report(void);

// Configuration and tuning
void set_bm25_parameters(float k1, float b);
void auto_tune_bm25_parameters(void);
void load_ranking_config(const char* config_file);
void save_ranking_config(const char* config_file);

// Advanced ranking features
float calculate_tf_idf_score(const char* term, int doc_id);
float calculate_cosine_similarity(QueryTerm* query_terms, int num_query_terms, int doc_id);
void rank_with_query_expansion(const char* query, char** expanded_terms, 
                               int num_expanded, int total_docs, int top_k);

// Multi-threaded result processing
void parallel_result_aggregation(SearchResult** partial_results, int* result_counts,
                                int num_partitions, SearchResult* final_results, 
                                int max_results);

// GPU memory management for ranking
#ifdef USE_CUDA
void allocate_gpu_ranking_memory(int max_docs, int max_terms);
void deallocate_gpu_ranking_memory(void);
int is_ranking_data_on_gpu(void);
void prefetch_ranking_data_gpu(QueryTerm* terms, int num_terms);
#endif

// Thread safety
void init_ranking_locks(void);
void destroy_ranking_locks(void);

// Validation and debugging
int validate_ranking_results(SearchResult* results, int num_results);
void print_ranking_debug_info(const char* query, SearchResult* results, int num_results);
int compare_cpu_gpu_ranking_results(const char* query, float tolerance);

// Query optimization
char* optimize_query_string(const char* original_query);
void remove_stopwords_from_query(char* query);
void stem_query_terms(QueryTerm* terms, int num_terms);

#endif // RANKING_H
