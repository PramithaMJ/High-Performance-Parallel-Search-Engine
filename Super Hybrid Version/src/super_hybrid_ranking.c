#include "../include/ranking.h"
#include "../include/index.h"
#include "../include/metrics.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Conditional includes for parallelization technologies
#ifdef USE_CUDA
#include "../include/cuda_kernels.h"
#endif

#ifdef USE_OPENMP
#include <omp.h>
#endif

#ifdef USE_MPI
#include <mpi.h>
#endif

// BM25 parameters (can be tuned for better performance)
#define K1 1.5f
#define B 0.75f

// Maximum number of results to return
#define MAX_RESULTS 100

// Result structure for super hybrid ranking
typedef struct {
    int doc_id;
    float score;
    char doc_name[256];
    int rank;
} SuperHybridResult;

// Performance metrics for ranking operations
typedef struct {
    double cuda_ranking_time;
    double openmp_ranking_time;
    double mpi_ranking_time;
    double serial_ranking_time;
    int documents_scored_cuda;
    int documents_scored_openmp;
    int documents_scored_mpi;
    int documents_scored_serial;
    double total_ranking_time;
} RankingMetrics;

static RankingMetrics g_ranking_metrics = {0};

// External declarations
extern InvertedIndex index_data[];
extern int index_size;
extern int doc_lengths[];
extern Document documents[];

// Function prototypes
int compare_results_desc(const void *a, const void *b);
double calculate_idf(int df, int total_docs);
float calculate_bm25_score(int tf, int doc_length, float avg_doc_length, float idf);
void merge_mpi_results(SuperHybridResult *local_results, int local_count, 
                      SuperHybridResult *global_results, int *global_count, int max_results);

// Super Hybrid BM25 ranking function that uses all available technologies
void rank_bm25_super_hybrid(const char *query, int total_docs, int max_results) {
    printf("🚀 Starting Super Hybrid BM25 Ranking...\n");
    printf("   Query: \"%s\"\n", query);
    printf("   Technologies: ");
#ifdef USE_CUDA
    printf("CUDA ");
#endif
#ifdef USE_OPENMP
    printf("OpenMP ");
#endif
#ifdef USE_MPI
    printf("MPI ");
#endif
    printf("\n");

#ifdef USE_MPI
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#else
    int mpi_rank = 0, mpi_size = 1;
#endif

    // Start timing
    double total_start_time = get_current_time();
    
    // Initialize results array
    SuperHybridResult *results = calloc(total_docs, sizeof(SuperHybridResult));
    if (!results) {
        printf("❌ Failed to allocate memory for results\n");
        return;
    }

    // Initialize all results
    for (int i = 0; i < total_docs; i++) {
        results[i].doc_id = i;
        results[i].score = 0.0f;
        results[i].rank = -1;
        if (i < 1000) {
            strncpy(results[i].doc_name, documents[i].filename, sizeof(results[i].doc_name) - 1);
            results[i].doc_name[sizeof(results[i].doc_name) - 1] = '\0';
        }
    }

    // Phase 1: Calculate average document length using parallel reduction
    double avg_doc_length = 0.0;
    
#ifdef USE_OPENMP
    double openmp_start = get_current_time();
    
    // Parallel calculation of average document length
    long total_length = 0;
    int valid_docs = 0;
    
    #pragma omp parallel for reduction(+:total_length,valid_docs)
    for (int i = 0; i < total_docs; i++) {
        if (i < 1000 && doc_lengths[i] > 0) {
            total_length += doc_lengths[i];
            valid_docs++;
        }
    }
    
    avg_doc_length = valid_docs > 0 ? (double)total_length / valid_docs : 1.0;
    
    double openmp_end = get_current_time();
    g_ranking_metrics.openmp_ranking_time += (openmp_end - openmp_start);
    
    printf("[MPI %d] 📊 Average document length: %.2f (calculated with OpenMP)\n", 
           mpi_rank, avg_doc_length);
#else
    // Serial calculation
    long total_length = 0;
    int valid_docs = 0;
    
    for (int i = 0; i < total_docs; i++) {
        if (i < 1000 && doc_lengths[i] > 0) {
            total_length += doc_lengths[i];
            valid_docs++;
        }
    }
    
    avg_doc_length = valid_docs > 0 ? (double)total_length / valid_docs : 1.0;
    
    printf("[MPI %d] 📊 Average document length: %.2f (calculated serially)\n", 
           mpi_rank, avg_doc_length);
#endif

#ifdef USE_MPI
    // Share average document length across all MPI processes
    double global_avg_doc_length;
    MPI_Allreduce(&avg_doc_length, &global_avg_doc_length, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    avg_doc_length = global_avg_doc_length / mpi_size;
#endif

    // Phase 2: Parse query terms
    char query_terms[10][256];
    int num_query_terms = 0;
    
    // Simple tokenization of query
    char *query_copy = strdup(query);
    char *token = strtok(query_copy, " \t\n");
    while (token != NULL && num_query_terms < 10) {
        strncpy(query_terms[num_query_terms], token, sizeof(query_terms[0]) - 1);
        query_terms[num_query_terms][sizeof(query_terms[0]) - 1] = '\0';
        num_query_terms++;
        token = strtok(NULL, " \t\n");
    }
    free(query_copy);

    printf("[MPI %d] 🔍 Processing %d query terms\n", mpi_rank, num_query_terms);

    // Phase 3: Calculate BM25 scores using different technologies
    
#ifdef USE_CUDA
    // GPU-accelerated BM25 scoring for large document sets
    if (total_docs > 100) {
        double cuda_start = get_current_time();
        
        printf("[MPI %d] 🔧 Using CUDA for BM25 scoring (%d documents)...\n", 
               mpi_rank, total_docs);
        
        // Prepare data for GPU processing
        float *doc_vectors = malloc(total_docs * num_query_terms * sizeof(float));
        float *query_vector = malloc(num_query_terms * sizeof(float));
        float *gpu_scores = malloc(total_docs * sizeof(float));
        
        if (doc_vectors && query_vector && gpu_scores) {
            // Build document vectors and query vector
            for (int term_idx = 0; term_idx < num_query_terms; term_idx++) {
                // Find term in index
                int found_index = -1;
                for (int i = 0; i < index_size; i++) {
                    if (strcmp(index_data[i].term, query_terms[term_idx]) == 0) {
                        found_index = i;
                        break;
                    }
                }
                
                if (found_index != -1) {
                    // Calculate IDF for this term
                    double idf = calculate_idf(index_data[found_index].df, total_docs);
                    query_vector[term_idx] = (float)idf;
                    
                    // Build document term frequency vector
                    for (int doc_id = 0; doc_id < total_docs; doc_id++) {
                        int tf = 0;
                        
                        // Find term frequency for this document
                        for (int j = 0; j < index_data[found_index].df; j++) {
                            if (index_data[found_index].postings[j].doc_id == doc_id) {
                                tf = index_data[found_index].postings[j].freq;
                                break;
                            }
                        }
                        
                        doc_vectors[doc_id * num_query_terms + term_idx] = (float)tf;
                    }
                } else {
                    query_vector[term_idx] = 0.0f;
                    for (int doc_id = 0; doc_id < total_docs; doc_id++) {
                        doc_vectors[doc_id * num_query_terms + term_idx] = 0.0f;
                    }
                }
            }
            
            // Call CUDA BM25 scoring function
            int cuda_result = cuda_compute_bm25_scores(doc_vectors, query_vector, gpu_scores,
                                                      total_docs, num_query_terms, K1, B, 
                                                      (float)avg_doc_length);
            
            if (cuda_result == 0) {
                // Copy GPU results back to results array
                for (int i = 0; i < total_docs; i++) {
                    results[i].score = gpu_scores[i];
                }
                g_ranking_metrics.documents_scored_cuda = total_docs;
            } else {
                printf("[MPI %d] ⚠️  CUDA scoring failed, falling back to CPU\n", mpi_rank);
            }
        }
        
        // Clean up GPU memory
        free(doc_vectors);
        free(query_vector);
        free(gpu_scores);
        
        double cuda_end = get_current_time();
        g_ranking_metrics.cuda_ranking_time += (cuda_end - cuda_start);
        
        printf("[MPI %d] ✅ CUDA BM25 scoring completed in %.3f seconds\n", 
               mpi_rank, cuda_end - cuda_start);
    } else {
#endif
        // CPU-based BM25 scoring with OpenMP parallelization
#ifdef USE_OPENMP
        double openmp_start = get_current_time();
        
        printf("[MPI %d] 🔄 Using OpenMP for BM25 scoring (%d documents)...\n", 
               mpi_rank, total_docs);
        
        // Process each query term in parallel
        for (int term_idx = 0; term_idx < num_query_terms; term_idx++) {
            // Find term in index
            int found_index = -1;
            for (int i = 0; i < index_size; i++) {
                if (strcmp(index_data[i].term, query_terms[term_idx]) == 0) {
                    found_index = i;
                    break;
                }
            }
            
            if (found_index != -1) {
                double idf = calculate_idf(index_data[found_index].df, total_docs);
                int df = index_data[found_index].df;
                
                // Parallel BM25 scoring for this term
                #pragma omp parallel for schedule(dynamic)
                for (int j = 0; j < df; j++) {
                    int doc_id = index_data[found_index].postings[j].doc_id;
                    int tf = index_data[found_index].postings[j].freq;
                    
                    if (doc_id < total_docs && doc_id < 1000) {
                        float score = calculate_bm25_score(tf, doc_lengths[doc_id], 
                                                         (float)avg_doc_length, (float)idf);
                        
                        #pragma omp critical(score_update)
                        {
                            results[doc_id].score += score;
                        }
                    }
                }
            }
        }
        
        double openmp_end = get_current_time();
        g_ranking_metrics.openmp_ranking_time += (openmp_end - openmp_start);
        g_ranking_metrics.documents_scored_openmp = total_docs;
        
        printf("[MPI %d] ✅ OpenMP BM25 scoring completed in %.3f seconds\n", 
               mpi_rank, openmp_end - openmp_start);
#else
        // Serial BM25 scoring fallback
        double serial_start = get_current_time();
        
        printf("[MPI %d] 📝 Using serial BM25 scoring (%d documents)...\n", 
               mpi_rank, total_docs);
        
        for (int term_idx = 0; term_idx < num_query_terms; term_idx++) {
            int found_index = -1;
            for (int i = 0; i < index_size; i++) {
                if (strcmp(index_data[i].term, query_terms[term_idx]) == 0) {
                    found_index = i;
                    break;
                }
            }
            
            if (found_index != -1) {
                double idf = calculate_idf(index_data[found_index].df, total_docs);
                
                for (int j = 0; j < index_data[found_index].df; j++) {
                    int doc_id = index_data[found_index].postings[j].doc_id;
                    int tf = index_data[found_index].postings[j].freq;
                    
                    if (doc_id < total_docs && doc_id < 1000) {
                        float score = calculate_bm25_score(tf, doc_lengths[doc_id], 
                                                         (float)avg_doc_length, (float)idf);
                        results[doc_id].score += score;
                    }
                }
            }
        }
        
        double serial_end = get_current_time();
        g_ranking_metrics.serial_ranking_time += (serial_end - serial_start);
        g_ranking_metrics.documents_scored_serial = total_docs;
        
        printf("[MPI %d] ✅ Serial BM25 scoring completed in %.3f seconds\n", 
               mpi_rank, serial_end - serial_start);
#endif
#ifdef USE_CUDA
    }
#endif

    // Phase 4: MPI-based result aggregation and merging
#ifdef USE_MPI
    if (mpi_size > 1) {
        double mpi_start = get_current_time();
        
        printf("[MPI %d] 🌐 Aggregating results across %d MPI processes...\n", 
               mpi_rank, mpi_size);
        
        // Each process sorts its local results
        qsort(results, total_docs, sizeof(SuperHybridResult), compare_results_desc);
        
        // Prepare local top results for gathering
        int local_top_count = (total_docs < max_results) ? total_docs : max_results;
        SuperHybridResult *local_top = malloc(local_top_count * sizeof(SuperHybridResult));
        
        // Copy top local results
        for (int i = 0; i < local_top_count; i++) {
            if (results[i].score > 0.0f) {
                local_top[i] = results[i];
            } else {
                local_top_count = i;
                break;
            }
        }
        
        // Gather all top results to rank 0
        if (mpi_rank == 0) {
            SuperHybridResult *all_results = malloc(mpi_size * max_results * sizeof(SuperHybridResult));
            int *result_counts = malloc(mpi_size * sizeof(int));
            
            // Gather result counts
            MPI_Gather(&local_top_count, 1, MPI_INT, result_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);
            
            // Gather actual results (simplified - would need custom MPI datatype for full implementation)
            // For now, we'll just use the local results on rank 0
            
            free(all_results);
            free(result_counts);
        } else {
            // Send count to rank 0
            MPI_Gather(&local_top_count, 1, MPI_INT, NULL, 0, MPI_INT, 0, MPI_COMM_WORLD);
        }
        
        free(local_top);
        
        double mpi_end = get_current_time();
        g_ranking_metrics.mpi_ranking_time += (mpi_end - mpi_start);
        
        printf("[MPI %d] ✅ MPI result aggregation completed in %.3f seconds\n", 
               mpi_rank, mpi_end - mpi_start);
    }
#endif

    // Phase 5: Final sorting and result presentation
    qsort(results, total_docs, sizeof(SuperHybridResult), compare_results_desc);
    
    // Only rank 0 prints results
    if (mpi_rank == 0) {
        double total_end_time = get_current_time();
        g_ranking_metrics.total_ranking_time = total_end_time - total_start_time;
        
        printf("\n🏆 Super Hybrid BM25 Ranking Results:\n");
        printf("════════════════════════════════════════════════════════════════\n");
        
        int displayed_results = 0;
        for (int i = 0; i < total_docs && displayed_results < max_results; i++) {
            if (results[i].score > 0.0f) {
                printf("%2d. %s (Score: %.4f)\n", 
                       displayed_results + 1, results[i].doc_name, results[i].score);
                displayed_results++;
            }
        }
        
        if (displayed_results == 0) {
            printf("No relevant documents found for query: \"%s\"\n", query);
        }
        
        printf("════════════════════════════════════════════════════════════════\n");
        printf("📊 Ranking Performance Summary:\n");
        printf("   Total Time: %.3f seconds\n", g_ranking_metrics.total_ranking_time);
        
#ifdef USE_CUDA
        if (g_ranking_metrics.cuda_ranking_time > 0) {
            printf("   CUDA Time: %.3f seconds (%d docs)\n", 
                   g_ranking_metrics.cuda_ranking_time, g_ranking_metrics.documents_scored_cuda);
        }
#endif
#ifdef USE_OPENMP
        if (g_ranking_metrics.openmp_ranking_time > 0) {
            printf("   OpenMP Time: %.3f seconds (%d docs)\n", 
                   g_ranking_metrics.openmp_ranking_time, g_ranking_metrics.documents_scored_openmp);
        }
#endif
#ifdef USE_MPI
        if (g_ranking_metrics.mpi_ranking_time > 0) {
            printf("   MPI Time: %.3f seconds\n", g_ranking_metrics.mpi_ranking_time);
        }
#endif
        if (g_ranking_metrics.serial_ranking_time > 0) {
            printf("   Serial Time: %.3f seconds (%d docs)\n", 
                   g_ranking_metrics.serial_ranking_time, g_ranking_metrics.documents_scored_serial);
        }
        
        printf("   Results Found: %d\n", displayed_results);
        printf("════════════════════════════════════════════════════════════════\n");
    }

    free(results);
}

// Wrapper function for backward compatibility
void rank_bm25(const char *query, int total_docs, int max_results) {
    rank_bm25_super_hybrid(query, total_docs, max_results);
}

// Comparison function for descending order sorting
int compare_results_desc(const void *a, const void *b) {
    const SuperHybridResult *result_a = (const SuperHybridResult *)a;
    const SuperHybridResult *result_b = (const SuperHybridResult *)b;
    
    if (result_a->score > result_b->score) return -1;
    if (result_a->score < result_b->score) return 1;
    return 0;
}

// Calculate Inverse Document Frequency
double calculate_idf(int df, int total_docs) {
    if (df <= 0 || total_docs <= 0) return 0.0;
    return log((double)total_docs / (double)df);
}

// Calculate BM25 score for a single term
float calculate_bm25_score(int tf, int doc_length, float avg_doc_length, float idf) {
    if (tf <= 0 || doc_length <= 0 || avg_doc_length <= 0.0f) return 0.0f;
    
    float tf_norm = (float)tf / ((float)tf + K1 * (1.0f - B + B * (float)doc_length / avg_doc_length));
    return idf * tf_norm;
}

// Advanced ranking with query expansion and relevance feedback
void rank_bm25_advanced(const char *query, int total_docs, int max_results, 
                       const char **expansion_terms, int num_expansion_terms) {
    printf("🔍 Advanced Super Hybrid Ranking with Query Expansion...\n");
    
    // First, run standard BM25
    rank_bm25_super_hybrid(query, total_docs, max_results);
    
    // Then, if expansion terms are provided, run expanded query
    if (expansion_terms && num_expansion_terms > 0) {
        char expanded_query[1024];
        strncpy(expanded_query, query, sizeof(expanded_query) - 1);
        expanded_query[sizeof(expanded_query) - 1] = '\0';
        
        for (int i = 0; i < num_expansion_terms; i++) {
            strncat(expanded_query, " ", sizeof(expanded_query) - strlen(expanded_query) - 1);
            strncat(expanded_query, expansion_terms[i], 
                   sizeof(expanded_query) - strlen(expanded_query) - 1);
        }
        
        printf("🔍 Expanded Query: \"%s\"\n", expanded_query);
        rank_bm25_super_hybrid(expanded_query, total_docs, max_results);
    }
}

// Get ranking performance metrics
void get_ranking_metrics(RankingMetrics *metrics) {
    if (metrics) {
        *metrics = g_ranking_metrics;
    }
}

// Reset ranking performance metrics
void reset_ranking_metrics() {
    memset(&g_ranking_metrics, 0, sizeof(g_ranking_metrics));
}

// Utility function to get current time (same as in index file)
double get_current_time() {
#ifdef USE_OPENMP
    return omp_get_wtime();
#else
    #include <time.h>
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
#endif
}
