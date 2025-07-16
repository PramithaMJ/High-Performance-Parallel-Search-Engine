#include "../include/ranking.h"
#include "../include/parser.h"
#include "../include/utils.h"
#include "../include/index.h"
#include "../include/metrics.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

// Forward declaration of get_doc_filename
extern const char* get_doc_filename(int doc_id);

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
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    // Start timing for query processing (only on rank 0)
    if (mpi_rank == 0) {
        start_timer();
    }
    
    // Calculate document range for this process
    int docs_per_proc = (total_docs + mpi_size - 1) / mpi_size;  // Ceiling division
    int start_doc = mpi_rank * docs_per_proc;
    int end_doc = (mpi_rank == mpi_size - 1) ? total_docs : start_doc + docs_per_proc;
    
    char query_copy[256];
    strcpy(query_copy, query);

    // Local results for this process
    Result local_results[1000] = {{0}};
    int local_result_count = 0;

    // Calculate average document length (only rank 0 does this and broadcasts)
    double avg_dl = 0;
    if (mpi_rank == 0) {
        for (int i = 0; i < total_docs; ++i)
            avg_dl += get_doc_length(i);
        avg_dl /= total_docs;
    }
    MPI_Bcast(&avg_dl, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    char *token = strtok(query_copy, " \t\n\r");
    while (token)
    {
        to_lowercase(token);
        if (!is_stopword(token))
        {
            char *term = stem(token);
            
            // Find the term in the index
            int found_index = -1;
            for (int i = 0; i < index_size; ++i)
            {
                if (strcmp(index_data[i].term, term) == 0)
                {
                    found_index = i;
                    break;
                }
            }
            
            if (found_index != -1)
            {
                int df = index_data[found_index].posting_count;
                double idf = log((total_docs - df + 0.5) / (df + 0.5) + 1.0);
                
                // Process postings for documents in this process's range
                for (int j = 0; j < df; ++j)
                {
                    int d = index_data[found_index].postings[j].doc_id;
                    
                    // Only process documents in this process's range
                    if (d >= start_doc && d < end_doc)
                    {
                        int tf = index_data[found_index].postings[j].freq;
                        double dl = get_doc_length(d);
                        double score = idf * ((tf * (1.5 + 1)) / (tf + 1.5 * (1 - 0.75 + 0.75 * dl / avg_dl)));

                        local_results[d].doc_id = d;
                        local_results[d].score += score;
                        if (d + 1 > local_result_count)
                            local_result_count = d + 1;
                    }
                }
            }
        }
        token = strtok(NULL, " \t\n\r");
    }

    // Sort local results
    qsort(local_results, local_result_count, sizeof(Result), cmp);
    
    // Gather top results from all processes
    Result gathered_results[1000 * 32];  // Assume max 32 processes
    int gathered_count = 0;
    
    // Get top local results (limit to prevent overflow)
    int local_top = (local_result_count < top_k * 2) ? local_result_count : top_k * 2;
    
    // Gather counts from all processes
    int all_counts[32];
    MPI_Gather(&local_top, 1, MPI_INT, all_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (mpi_rank == 0) {
        // Calculate displacements for gathering
        int displacements[32];
        displacements[0] = 0;
        gathered_count = all_counts[0];
        
        for (int i = 1; i < mpi_size; i++) {
            displacements[i] = displacements[i-1] + all_counts[i-1];
            gathered_count += all_counts[i];
        }
        
        // Gather all results
        MPI_Gatherv(local_results, local_top * sizeof(Result), MPI_BYTE,
                   gathered_results, all_counts, displacements, MPI_BYTE, 0, MPI_COMM_WORLD);
        
        // Convert byte counts to Result counts for sorting
        for (int i = 0; i < mpi_size; i++) {
            all_counts[i] /= sizeof(Result);
        }
        gathered_count /= sizeof(Result);
        
        // Sort all gathered results
        qsort(gathered_results, gathered_count, sizeof(Result), cmp);
        
        // Record query processing time
        double query_time = stop_timer();
        metrics.query_processing_time = query_time;
        
        // Record query latency for statistical purposes
        record_query_latency(query_time);
        
        printf("Query processed in %.2f ms\n", query_time);
        
        // Print top results
        int printed = 0;
        for (int i = 0; i < gathered_count && printed < top_k; ++i)
        {
            if (gathered_results[i].score > 0) {
                printf("File: %s - Score: %.4f\n", 
                       get_doc_filename(gathered_results[i].doc_id), 
                       gathered_results[i].score);
                printed++;
            }
        }
        
        if (printed == 0) {
            printf("No results found for the query.\n");
        }
    } else {
        // Non-root processes just send their results
        MPI_Gatherv(local_results, local_top * sizeof(Result), MPI_BYTE,
                   NULL, NULL, NULL, MPI_BYTE, 0, MPI_COMM_WORLD);
    }
}
