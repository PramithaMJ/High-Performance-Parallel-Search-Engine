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
#include <stddef.h>

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
    
    if (mpi_rank == 0) {
        printf("Debug: Verifying document filenames before search\n");
        for (int i = 0; i < total_docs && i < 5; i++) {
            printf("Debug: Doc %d filename: '%s'\n", i, get_doc_filename(i));
        }
    }
    
    if (mpi_rank == 0) {
        start_timer();
    }
    
    // Calculate document range for this process
    int docs_per_proc = (total_docs + mpi_size - 1) / mpi_size;
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
                
                // Process all postings since we have the full merged index
                for (int j = 0; j < df; ++j)
                {
                    int d = index_data[found_index].postings[j].doc_id;
                    
                    // Process all documents (not just in this process's range)
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
        token = strtok(NULL, " \t\n\r");
    }

    qsort(local_results, local_result_count, sizeof(Result), cmp);
    
    // Gather top results from all processes
    Result gathered_results[1000 * 32];
    int gathered_count = 0;
    
    int local_top = (local_result_count < top_k * 2) ? local_result_count : top_k * 2;
    
    // Create MPI datatype for Result struct if not already created
    static int type_created = 0;
    static MPI_Datatype MPI_RESULT_TYPE;
    if (!type_created) {
        MPI_Datatype types[2] = {MPI_INT, MPI_DOUBLE};
        int blocklengths[2] = {1, 1};
        MPI_Aint offsets[2];
        
        // Calculate offsets
        offsets[0] = offsetof(Result, doc_id);
        offsets[1] = offsetof(Result, score);
        
        // Create and commit the type
        MPI_Type_create_struct(2, blocklengths, offsets, types, &MPI_RESULT_TYPE);
        MPI_Type_commit(&MPI_RESULT_TYPE);
        type_created = 1;
    }
    
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
    
        // Each process sends local_top results, receives local_top from each process
        MPI_Gather(local_results, local_top, MPI_RESULT_TYPE,
                  gathered_results, local_top, MPI_RESULT_TYPE, 0, MPI_COMM_WORLD);
        
        qsort(gathered_results, gathered_count, sizeof(Result), cmp);
        
        double query_time = stop_timer();
        metrics.query_processing_time = query_time;
        
        record_query_latency(query_time);
        
        printf("Query processed in %.2f ms\n", query_time);
        
        int printed = 0;
        for (int i = 0; i < gathered_count && printed < top_k; ++i)
        {
            if (gathered_results[i].score > 0) {
                const char* filename = get_doc_filename(gathered_results[i].doc_id);
                if (filename && filename[0] != '\0') {
                    printf("File: %s - Score: %.4f\n", 
                           filename, 
                           gathered_results[i].score);
                } else {
                    printf("File: doc_id %d - Score: %.4f (Filename not available)\n", 
                           gathered_results[i].doc_id,
                           gathered_results[i].score);
                }
                printed++;
            }
        }
        
        if (printed == 0) {
            printf("No results found for the query.\n");
        }
    } else {
        MPI_Gather(local_results, local_top, MPI_RESULT_TYPE,
                  NULL, local_top, MPI_RESULT_TYPE, 0, MPI_COMM_WORLD);
    }
}
