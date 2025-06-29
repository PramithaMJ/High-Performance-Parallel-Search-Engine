#include "../include/ranking.h"
#include "../include/parser.h"
#include "../include/utils.h"
#include "../include/index.h"
#include "../include/metrics.h"
#include <mpi.h>

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

// Custom MPI datatype for Result struct
MPI_Datatype MPI_RESULT_TYPE;

int cmp(const void *a, const void *b)
{
    Result *r1 = (Result *)a;
    Result *r2 = (Result *)b;
    return (r2->score > r1->score) - (r2->score < r1->score);
}

// Function to create MPI Result datatype
void create_result_datatype() {
    // Define the struct layout for MPI
    MPI_Datatype types[2] = {MPI_INT, MPI_DOUBLE};
    int blocklengths[2] = {1, 1};
    MPI_Aint offsets[2];
    
    // Calculate offsets
    offsets[0] = offsetof(Result, doc_id);
    offsets[1] = offsetof(Result, score);
    
    // Create the MPI datatype
    MPI_Type_create_struct(2, blocklengths, offsets, types, &MPI_RESULT_TYPE);
    MPI_Type_commit(&MPI_RESULT_TYPE);
}

void rank_bm25(const char *query, int total_docs, int top_k)
{
    int rank, size, provided;
    // Initialize MPI if not already initialized
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
        MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
        create_result_datatype();
        
        // Log thread support level if verbose is enabled
        if (provided != MPI_THREAD_MULTIPLE) {
            printf("Warning: MPI implementation doesn't fully support MPI_THREAD_MULTIPLE\n");
            printf("Provided threading level: %d\n", provided);
        }
    } else {
        // Create the datatype if MPI is already initialized
        static int type_created = 0;
        if (!type_created) {
            create_result_datatype();
            type_created = 1;
        }
    }
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Start timing for query processing (only on rank 0)
    if (rank == 0) {
        start_timer();
    }
    
    // Calculate document range for this process with better load balancing
    int docs_per_proc = (total_docs + size - 1) / size;  // Ceiling division for more even distribution
    int start_doc = rank * docs_per_proc;
    int end_doc = (rank == size - 1) ? total_docs : start_doc + docs_per_proc;
    
    // Debug info about document distribution
    if (rank == 0) {
        printf("Distributing %d documents across %d MPI processes\n", total_docs, size);
    }
    
    char query_copy[256];
    strcpy(query_copy, query);

    char *query_terms[32]; // Store query terms for parallel processing
    int num_terms = 0;
    
    // Tokenize query and store terms
    char *token = strtok(query_copy, " \t\n\r");
    while (token && num_terms < 32) {
        to_lowercase(token);
        if (!is_stopword(token)) {
            query_terms[num_terms] = strdup(stem(token));
            num_terms++;
        }
        token = strtok(NULL, " \t\n\r");
    }
    
    // Local results for this process
    Result *local_results = (Result *)calloc(total_docs, sizeof(Result));
    for (int i = 0; i < total_docs; i++) {
        local_results[i].doc_id = i;
        local_results[i].score = 0.0;
    }
    
    // Calculate average document length across all nodes
    double local_sum_dl = 0.0;
    double global_avg_dl = 0.0;
    
    // Use OpenMP to parallelize local document length calculation
    #pragma omp parallel reduction(+:local_sum_dl)
    {
        #pragma omp for
        for (int i = start_doc; i < end_doc; i++) {
            local_sum_dl += get_doc_length(i);
        }
    }
    
    // Reduce to get global average
    double global_sum_dl = 0.0;
    MPI_Allreduce(&local_sum_dl, &global_sum_dl, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    global_avg_dl = global_sum_dl / total_docs;
    
    // Process each query term in parallel with OpenMP
    #pragma omp parallel
    {
        // Thread-local results to avoid critical sections
        Result *thread_results = (Result *)calloc(total_docs, sizeof(Result));
        if (thread_results) {  // Check allocation success
            for (int i = 0; i < total_docs; i++) {
                thread_results[i].doc_id = i;
                thread_results[i].score = 0.0;
            }
            
            #pragma omp for
            for (int term_idx = 0; term_idx < num_terms; term_idx++) {
                char *term = query_terms[term_idx];
                
                // Find the term in the index
                for (int i = 0; i < index_size; ++i) {
                    if (strcmp(index_data[i].term, term) == 0) {
                        int df = index_data[i].posting_count;
                        double idf = log((total_docs - df + 0.5) / (df + 0.5) + 1.0);
                        
                        // Process postings for this term
                        for (int j = 0; j < df; ++j) {
                            int d = index_data[i].postings[j].doc_id;
                            // Only process if document is in this node's range or if we're processing all documents
                            if ((d >= start_doc && d < end_doc) || size == 1) {
                                int tf = index_data[i].postings[j].freq;
                                double dl = get_doc_length(d);
                                // BM25 formula: k = 1.5, b = 0.75
                                double score = idf * ((tf * (1.5 + 1)) / (tf + 1.5 * (1 - 0.75 + 0.75 * dl / global_avg_dl)));
                                thread_results[d].score += score;
                            }
                        }
                        break;
                    }
                }
            }
            
            // Merge thread results into local results
            #pragma omp critical
            {
                for (int i = 0; i < total_docs; i++) {
                    local_results[i].score += thread_results[i].score;
                }
            }
            
            free(thread_results);
        }
    }
    
    // Free query terms
    for (int i = 0; i < num_terms; i++) {
        free(query_terms[i]);
    }
    
    // Sort local results
    qsort(local_results, total_docs, sizeof(Result), cmp);
    
    // Send top-k local results to rank 0
    Result *all_results = NULL;
    if (rank == 0) {
        all_results = (Result *)malloc(size * top_k * sizeof(Result));
    }
    
    // Gather top-k results from all processes
    MPI_Gather(local_results, top_k, MPI_RESULT_TYPE, 
               all_results, top_k, MPI_RESULT_TYPE, 0, MPI_COMM_WORLD);
    
    // Root process combines and sorts all results
    if (rank == 0) {
        // Sort all gathered results
        qsort(all_results, size * top_k, sizeof(Result), cmp);
        
        // Record query processing time
        double query_time = stop_timer();
        metrics.query_processing_time = query_time;
        
        // Record query latency for statistical purposes
        record_query_latency(query_time);
        
        printf("Query processed in %.2f ms\n", query_time);
        
        // Display top-k results
        for (int i = 0; i < top_k && i < size * top_k; ++i) {
            if (all_results[i].score > 0) {
                printf("File: %s - Score: %.4f\n", 
                       get_doc_filename(all_results[i].doc_id), 
                       all_results[i].score);
            }
        }
        
        free(all_results);
    }
    
    free(local_results);
    
    // Don't finalize MPI here, as it may be needed for other operations
}
