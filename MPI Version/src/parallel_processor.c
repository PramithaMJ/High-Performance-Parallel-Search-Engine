#include "../include/parallel_processor.h"
#include "../include/mpi_comm.h"
#include "../include/metrics.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

// MPI tags
#define TASK_ASSIGN_TAG 200
#define TASK_RESULT_TAG 201
#define TASK_COMPLETE_TAG 202

// Create a parallel processor
ParallelProcessor* create_parallel_processor(int mpi_rank, int mpi_size) {
    ParallelProcessor* processor = (ParallelProcessor*)malloc(sizeof(ParallelProcessor));
    
    if (!processor) {
        fprintf(stderr, "Failed to allocate parallel processor\n");
        return NULL;
    }
    
    // Initialize processor
    processor->mpi_rank = mpi_rank;
    processor->mpi_size = mpi_size;
    processor->buffer_size = PP_BUFFER_SIZE;
    processor->buffer = (char*)malloc(processor->buffer_size);
    
    if (!processor->buffer) {
        fprintf(stderr, "Failed to allocate processor buffer\n");
        free(processor);
        return NULL;
    }
    
    return processor;
}

// Free a parallel processor
void free_parallel_processor(ParallelProcessor* processor) {
    if (!processor) return;
    
    if (processor->buffer) {
        free(processor->buffer);
    }
    
    free(processor);
}

// Process documents in parallel
void process_documents_parallel(ParallelProcessor* processor, 
                             char file_paths[][256], 
                             int file_count, 
                             ProcessorCallback callback) {
    int rank = processor->mpi_rank;
    int size = processor->mpi_size;
    
    // Start timing
    if (rank == 0) {
        start_timer();
    }
    
    if (size == 1) {
        // Single process, process all files sequentially
        for (int i = 0; i < file_count; i++) {
            callback(file_paths[i], i, NULL);
        }
    } else if (rank == 0) {
        // Master process - distribute work
        int files_assigned = 0;
        int files_completed = 0;
        int *worker_status = (int*)calloc(size, sizeof(int));
        MPI_Status status;
        
        // Initial distribution - assign one file to each worker
        for (int i = 1; i < size && files_assigned < file_count; i++) {
            MPI_Send(&files_assigned, 1, MPI_INT, i, TASK_ASSIGN_TAG, MPI_COMM_WORLD);
            worker_status[i] = 1;  // Mark worker as busy
            files_assigned++;
        }
        
        // Process remaining files as workers complete their tasks
        while (files_completed < file_count) {
            // Check for completed work
            MPI_Probe(MPI_ANY_SOURCE, TASK_RESULT_TAG, MPI_COMM_WORLD, &status);
            int worker = status.MPI_SOURCE;
            
            // Receive result notification
            int file_index;
            MPI_Recv(&file_index, 1, MPI_INT, worker, TASK_RESULT_TAG, MPI_COMM_WORLD, &status);
            
            // Mark file as complete
            files_completed++;
            
            // Assign new work if available
            if (files_assigned < file_count) {
                MPI_Send(&files_assigned, 1, MPI_INT, worker, TASK_ASSIGN_TAG, MPI_COMM_WORLD);
                files_assigned++;
            } else {
                // No more work, send completion signal
                int done = -1;
                MPI_Send(&done, 1, MPI_INT, worker, TASK_ASSIGN_TAG, MPI_COMM_WORLD);
                worker_status[worker] = 0; 
            }
        }
        
        // Send completion signal to any remaining workers
        for (int i = 1; i < size; i++) {
            if (worker_status[i]) {
                int done = -1;
                MPI_Send(&done, 1, MPI_INT, i, TASK_ASSIGN_TAG, MPI_COMM_WORLD);
            }
        }
        
        free(worker_status);
    } else {
        // Worker process - receive work and process it
        MPI_Status status;
        int file_index;
        
        // Receive work
        MPI_Recv(&file_index, 1, MPI_INT, 0, TASK_ASSIGN_TAG, MPI_COMM_WORLD, &status);
        
        // Process files until receiving completion signal
        while (file_index >= 0 && file_index < file_count) {
            callback(file_paths[file_index], file_index, processor);
            
            MPI_Send(&file_index, 1, MPI_INT, 0, TASK_RESULT_TAG, MPI_COMM_WORLD);
            MPI_Recv(&file_index, 1, MPI_INT, 0, TASK_ASSIGN_TAG, MPI_COMM_WORLD, &status);
        }
    }
    
    // Wait for all processes to complete
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Record processing time
    if (rank == 0) {
        double processing_time = stop_timer();
        printf("Parallel document processing completed in %.2f ms using %d processes\n",
               processing_time, size);
    }
}

// Process batch of tokens in parallel
void process_tokens_parallel(ParallelProcessor* processor, 
                           char** tokens, 
                           int token_count, 
                           TokenCallback callback) {
    int rank = processor->mpi_rank;
    int size = processor->mpi_size;
    
    if (size == 1) {
        // Single process, process all tokens sequentially
        for (int i = 0; i < token_count; i++) {
            callback(tokens[i], i, NULL);
        }
        return;
    }
    
    // Determine workload distribution
    int tokens_per_process = token_count / size;
    int remaining_tokens = token_count % size;
    int start_idx = rank * tokens_per_process + (rank < remaining_tokens ? rank : remaining_tokens);
    int end_idx = start_idx + tokens_per_process + (rank < remaining_tokens ? 1 : 0);
    
    for (int i = start_idx; i < end_idx; i++) {
        callback(tokens[i], i, processor);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
}
