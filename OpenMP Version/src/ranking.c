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

typedef struct
{
    int doc_id;
    double score;
} Result;

int cmp(const void *a, const void *b)
{
    Result *r1 = (Result *)a;
    Result *r2 = (Result *)b;
    return (r2->score > r1->score) - (r2->score < r1->score);
}

void rank_bm25(const char *query, int total_docs, int top_k)
{
    // Start timing for query processing
    start_timer();
    
    char query_copy[256];
    strncpy(query_copy, query, sizeof(query_copy) - 1);
    query_copy[sizeof(query_copy) - 1] = '\0';

    char *token = strtok(query_copy, " \t\n\r");
    Result results[1000] = {{0}};
    int result_count = 0;

    double avg_dl = 0;
    
    // Parallel calculation of average document length
    #pragma omp parallel
    {
        double local_avg_dl = 0;
        
        #pragma omp for nowait
        for (int i = 0; i < total_docs; ++i)
            local_avg_dl += get_doc_length(i);
        
        #pragma omp critical
        avg_dl += local_avg_dl;
    }
    
    avg_dl /= total_docs;

    // Process each query term
    while (token)
    {
        to_lowercase(token);
        if (!is_stopword(token))
        {
            // Standard term processing for all query terms
            char *term = stem(token);
            
            int term_found = 0;
            
            // Find the term in the index
            for (int i = 0; i < index_size; ++i)
            {
                if (strcmp(index_data[i].term, term) == 0)
                {
                    int df = index_data[i].posting_count;
                    double idf = log((total_docs - df + 0.5) / (df + 0.5) + 1.0);
                    term_found = 1;
                    
                    // Store results in thread-private arrays before final merge
                    double *thread_scores = malloc(total_docs * sizeof(double));
                    memset(thread_scores, 0, total_docs * sizeof(double));
                    
                    // Sequential processing of document scores for consistent results
                    for (int j = 0; j < df; ++j)
                    {
                        int d = index_data[i].postings[j].doc_id;
                        int tf = index_data[i].postings[j].freq;
                        double dl = get_doc_length(d);
                        double score = idf * ((tf * (1.5 + 1)) / (tf + 1.5 * (1 - 0.75 + 0.75 * dl / avg_dl)));
                        
                        thread_scores[d] = score;
                    }
                    
                    // Merge scores into final results (sequentially)
                    for (int d = 0; d < total_docs; ++d) {
                        if (thread_scores[d] > 0) {
                            results[d].doc_id = d;
                            results[d].score += thread_scores[d];
                            if (d + 1 > result_count)
                                result_count = d + 1;
                        }
                    }
                    
                    free(thread_scores);
                    break;
                }
            }
            
            // If the term wasn't found, try variations (singular/plural forms)
            if (!term_found) {
                // Use a thread-safe buffer for the alternative term
                char alternative_term[256] = {0};
                
                // Get length of the original term safely
                int len = strlen(term);
                if (len >= sizeof(alternative_term)) {
                    len = sizeof(alternative_term) - 2;  // Leave room for potential 's'
                }
                
                // Try plural form if term doesn't end with 's'
                if (len > 0 && term[len-1] != 's') {
                    // Create plural form - safely
                    strncpy(alternative_term, term, len);
                    alternative_term[len] = 's';
                    alternative_term[len+1] = '\0';
                } 
                // Try singular form if term ends with 's'
                else if (len > 1) {
                    // Create singular form - safely
                    strncpy(alternative_term, term, len-1);
                    alternative_term[len-1] = '\0';
                }
                
                // Search the index again with the alternative form
                for (int i = 0; i < index_size; ++i) {
                    if (alternative_term[0] != '\0' && strcmp(index_data[i].term, alternative_term) == 0) {
                        int df = index_data[i].posting_count;
                        double idf = log((total_docs - df + 0.5) / (df + 0.5) + 1.0);
                        
                        // Apply same scoring factor for alternative forms (no penalty)
                        double alt_factor = 1.0; // Treat singular/plural the same
                        
                        // Store results in thread-private arrays before final merge for consistency
                        double *thread_scores = malloc(total_docs * sizeof(double));
                        memset(thread_scores, 0, total_docs * sizeof(double));
                        
                        // Sequential processing for deterministic results
                        for (int j = 0; j < df; ++j) {
                            int d = index_data[i].postings[j].doc_id;
                            int tf = index_data[i].postings[j].freq;
                            double dl = get_doc_length(d);
                            double score = alt_factor * idf * ((tf * (1.5 + 1)) / 
                                            (tf + 1.5 * (1 - 0.75 + 0.75 * dl / avg_dl)));
                            
                            thread_scores[d] = score;
                        }
                        
                        // Merge scores into final results (sequentially)
                        for (int d = 0; d < total_docs; ++d) {
                            if (thread_scores[d] > 0) {
                                results[d].doc_id = d;
                                results[d].score += thread_scores[d];
                                if (d + 1 > result_count)
                                    result_count = d + 1;
                            }
                        }
                        
                        free(thread_scores);
                        term_found = 1; // Mark as found to avoid showing "no results" message
                        break;
                    }
                }
            }
        }
        token = strtok(NULL, " \t\n\r");
    }

    qsort(results, result_count, sizeof(Result), cmp);
    
    // Record query processing time
    double query_time = stop_timer();
    metrics.query_processing_time = query_time;
    
    // Record query latency for statistical purposes
    record_query_latency(query_time);
    
    printf("Query processed in %.2f ms\n", query_time);
    
    int results_found = 0;
    for (int i = 0; i < top_k && i < result_count; ++i)
    {
        if (results[i].score > 0) {
            const char* filename = get_doc_filename(results[i].doc_id);
            printf("#%d: %s (Score: %.4f)\n", i+1, filename, results[i].score);
            results_found++;
        }
    }
    
    if (results_found == 0) {
        printf("No matching documents found for query: \"%s\"\n", query);
        
        // Only suggest alternatives for more complex cases (not singular/plural which are already handled)
        // Potential future expansion: could implement more sophisticated suggestions here
        // For now, we don't show the "Did you mean" for singular/plural forms since we search both automatically
    } else {
        printf("\nFound %d matching document(s) for query: \"%s\"\n", results_found, query);
    }
}
