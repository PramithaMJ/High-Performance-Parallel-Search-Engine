#include "../include/ranking.h"
#include "../include/parser.h"
#include "../include/utils.h"
#include "../include/index.h"
#include "../include/metrics.h"

// Conditionally include OpenMP header
#ifdef _OPENMP
  #include <omp.h>
#endif

// Forward declaration of get_doc_filename
extern const char* get_doc_filename(int doc_id);
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int doc_id;
    double score;
} Result;

// Comparison function for qsort
int cmp(const void *a, const void *b) {
    Result *r1 = (Result *)a;
    Result *r2 = (Result *)b;
    return (r2->score > r1->score) - (r2->score < r1->score);
}

/**
 * Enhanced BM25 ranking function with improved parallelism
 * - Uses parallel tokenization and term processing
 * - Uses thread-local storage for deterministic results
 * - Implements more efficient memory management
 * - Uses OpenMP tasks for better load balancing
 */
void rank_bm25(const char *query, int total_docs, int top_k) {
    // Start timing for query processing
    start_timer();
    
    // Create a copy of the query for tokenization
    char query_copy[256];
    strncpy(query_copy, query, sizeof(query_copy) - 1);
    query_copy[sizeof(query_copy) - 1] = '\0';

    // Tokenize the query and store tokens in an array
    char *query_tokens[64] = {NULL}; // Up to 64 tokens
    int query_token_count = 0;
    
    char *tokenize_copy = strdup(query_copy);
    char *token = strtok(tokenize_copy, " \t\n\r");
    while (token && query_token_count < 64) {
        query_tokens[query_token_count++] = strdup(token);
        token = strtok(NULL, " \t\n\r");
    }
    free(tokenize_copy);
    
    // Initialize result array
    Result results[1000] = {{0}};
    int result_count = 0;

    // Calculate average document length in parallel using reduction
    double avg_dl = 0;
    #pragma omp parallel reduction(+:avg_dl)
    {
        #pragma omp for schedule(static)
        for (int i = 0; i < total_docs; ++i)
            avg_dl += get_doc_length(i);
    }
    avg_dl /= total_docs;

    // Process all query terms in parallel
    // Each term's processing is independent and can be done in parallel
    // But we need thread-local storage for the scores to ensure deterministic results
    
    // Allocate memory for term scores
    double **term_scores_array = (double**)malloc(query_token_count * sizeof(double*));
    int *term_found_array = (int*)calloc(query_token_count, sizeof(int));
    
    // Process each query term in parallel
    #pragma omp parallel for schedule(dynamic)
    for (int term_idx = 0; term_idx < query_token_count; term_idx++) {
        // Allocate thread-local score array for this term
        double *term_scores = (double*)calloc(total_docs, sizeof(double));
        term_scores_array[term_idx] = term_scores;
        
        // Process this query term
        char *current_token = query_tokens[term_idx];
        to_lowercase(current_token);
        
        if (!is_stopword(current_token)) {
            // Get stemmed version of the term
            char *term = stem(current_token);
            
            // Look for the term in the index
            for (int i = 0; i < index_size; ++i) {
                if (strcmp(index_data[i].term, term) == 0) {
                    int df = index_data[i].posting_count;
                    double idf = log((total_docs - df + 0.5) / (df + 0.5) + 1.0);
                    term_found_array[term_idx] = 1;
                    
                    // Calculate scores for all documents containing this term
                    // This is sequential within the thread, for deterministic results
                    for (int j = 0; j < df; ++j) {
                        int d = index_data[i].postings[j].doc_id;
                        int tf = index_data[i].postings[j].freq;
                        double dl = get_doc_length(d);
                        double score = idf * ((tf * (1.5 + 1)) / (tf + 1.5 * (1 - 0.75 + 0.75 * dl / avg_dl)));
                        term_scores[d] = score;
                    }
                    break;
                }
            }
            
            // If the term wasn't found, try singular/plural variation
            if (!term_found_array[term_idx]) {
                // Thread-local buffer for alternative term
                char alternative_term[256] = {0};
                
                // Get length safely
                int len = strlen(term);
                if (len >= sizeof(alternative_term)) {
                    len = sizeof(alternative_term) - 2;
                }
                
                // Try plural if not ending with 's'
                if (len > 0 && term[len-1] != 's') {
                    strncpy(alternative_term, term, len);
                    alternative_term[len] = 's';
                    alternative_term[len+1] = '\0';
                } 
                // Try singular if ending with 's'
                else if (len > 1) {
                    strncpy(alternative_term, term, len-1);
                    alternative_term[len-1] = '\0';
                }
                
                // Search for the alternative form
                for (int i = 0; i < index_size; ++i) {
                    if (alternative_term[0] != '\0' && strcmp(index_data[i].term, alternative_term) == 0) {
                        int df = index_data[i].posting_count;
                        double idf = log((total_docs - df + 0.5) / (df + 0.5) + 1.0);
                        double alt_factor = 1.0; // No penalty for alt forms
                        term_found_array[term_idx] = 1;
                        
                        // Calculate scores
                        for (int j = 0; j < df; ++j) {
                            int d = index_data[i].postings[j].doc_id;
                            int tf = index_data[i].postings[j].freq;
                            double dl = get_doc_length(d);
                            double score = alt_factor * idf * 
                                ((tf * (1.5 + 1)) / (tf + 1.5 * (1 - 0.75 + 0.75 * dl / avg_dl)));
                            term_scores[d] = score;
                        }
                        break;
                    }
                }
            }
        }
    }
    
    // Merge the term scores into the final results array
    // This part is sequential to ensure deterministic results
    for (int term_idx = 0; term_idx < query_token_count; term_idx++) {
        double *term_scores = term_scores_array[term_idx];
        
        // Add this term's scores to the final results
        for (int d = 0; d < total_docs; ++d) {
            if (term_scores[d] > 0) {
                results[d].doc_id = d;
                results[d].score += term_scores[d];
                if (d + 1 > result_count)
                    result_count = d + 1;
            }
        }
    }
    
    // Sort the results by score (descending)
    qsort(results, result_count, sizeof(Result), cmp);
    
    // Record query processing time
    double query_time = stop_timer();
    metrics.query_processing_time = query_time;
    record_query_latency(query_time);
    
    printf("Query processed in %.2f ms\n", query_time);
    
    // Display the results
    int results_found = 0;
    
    // Parallel results display preparation
    char **result_strings = NULL;
    if (result_count > 0) {
        result_strings = (char**)malloc(top_k * sizeof(char*));
        for (int i = 0; i < top_k; i++) {
            result_strings[i] = (char*)malloc(256 * sizeof(char));
            result_strings[i][0] = '\0';
        }
        // Calculate the actual number of items to process
        int items_to_process = (top_k < result_count) ? top_k : result_count;
        
        #pragma omp parallel for schedule(dynamic) if(items_to_process > 10)
        for (int i = 0; i < items_to_process; ++i) {
            if (results[i].score > 0) {
                const char* filename = get_doc_filename(results[i].doc_id);
                snprintf(result_strings[i], 255, "#%d: %s (Score: %.4f)", 
                         i+1, filename, results[i].score);
                #pragma omp atomic
                results_found++;
            }
        }
    }
    
    // Display results sequentially for proper ordering
    int display_count = (top_k < result_count) ? top_k : result_count;
    for (int i = 0; i < display_count; ++i) {
        if (result_strings[i][0] != '\0') {
            printf("%s\n", result_strings[i]);
        }
    }
    
    // Clean up result strings
    if (result_strings) {
        for (int i = 0; i < top_k; i++) {
            free(result_strings[i]);
        }
        free(result_strings);
    }
    
    // Display summary
    if (results_found == 0) {
        printf("No matching documents found for query: \"%s\"\n", query);
    } else {
        printf("\nFound %d matching document(s) for query: \"%s\"\n", results_found, query);
    }
    
    // Clean up resources
    for (int i = 0; i < query_token_count; i++) {
        free(query_tokens[i]);
        free(term_scores_array[i]);
    }
    free(term_scores_array);
    free(term_found_array);
}
