#include "../include/load_balancer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <sys/stat.h>

// Work distribution strategies
#define STRATEGY_STATIC 0
#define STRATEGY_DYNAMIC 1
#define STRATEGY_ADAPTIVE 2

// Global variables for work distribution
static int current_strategy = STRATEGY_STATIC;
static double *process_load_metrics = NULL;

void init_load_balancer(int mpi_size) {
    process_load_metrics = (double*)calloc(mpi_size, sizeof(double));
    
    current_strategy = STRATEGY_STATIC;
    
    char *strategy_env = getenv("SEARCH_ENGINE_LOAD_STRATEGY");
    if (strategy_env) {
        if (strcmp(strategy_env, "dynamic") == 0) {
            current_strategy = STRATEGY_DYNAMIC;
        } else if (strcmp(strategy_env, "adaptive") == 0) {
            current_strategy = STRATEGY_ADAPTIVE;
        }
    }
}

void free_load_balancer() {
    if (process_load_metrics) {
        free(process_load_metrics);
        process_load_metrics = NULL;
    }
}

// similar to current implementation but with file size
void static_distribute_workload(int mpi_rank, int mpi_size, int file_count, 
                                char file_paths[][256], int *start_idx, int *end_idx) {
    // Get file sizes for better balancing
    off_t *file_sizes = (off_t*)calloc(file_count, sizeof(off_t));
    
    for (int i = 0; i < file_count; i++) {
        struct stat st;
        if (stat(file_paths[i], &st) == 0) {
            file_sizes[i] = st.st_size;
        }
    }
    
    // Calculate total bytes and target bytes per process
    off_t total_bytes = 0;
    for (int i = 0; i < file_count; i++) {
        total_bytes += file_sizes[i];
    }
    
    off_t bytes_per_process = total_bytes / mpi_size;
    
    // Distribute files to achieve balanced byte count
    off_t current_bytes = 0;
    int current_process = 0;
    int *process_start_idx = (int*)calloc(mpi_size, sizeof(int));
    int *process_end_idx = (int*)calloc(mpi_size, sizeof(int));
    
    process_start_idx[0] = 0;
    
    for (int i = 0; i < file_count; i++) {
        current_bytes += file_sizes[i];
        
        if (current_bytes >= (current_process + 1) * bytes_per_process && current_process < mpi_size - 1) {
            process_end_idx[current_process] = i;
            current_process++;
            process_start_idx[current_process] = i + 1;
        }
    }
    
    // Ensure the last process handles all remaining files
    process_end_idx[mpi_size - 1] = file_count - 1;
    
    // Set the start and end indices for the current process
    *start_idx = process_start_idx[mpi_rank];
    *end_idx = process_end_idx[mpi_rank];
    
    // Clean up
    free(file_sizes);
    free(process_start_idx);
    free(process_end_idx);
}

// Dynamic work distribution with master-worker pattern
void dynamic_distribute_workload(int mpi_rank, int mpi_size, int file_count, 
                                char file_paths[][256], int *files_to_process, int *file_indices) {
    int work_unit_size = 5;
    int current_file = 0;
    MPI_Status status;
    
    if (mpi_rank == 0) {
        // Master process distributes work
        int working_processes = 0;
        
        // Initial distribution
        for (int i = 1; i < mpi_size && current_file < file_count; i++) {
            int work_size = (current_file + work_unit_size <= file_count) ? 
                            work_unit_size : (file_count - current_file);
            
            MPI_Send(&work_size, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            
            if (work_size > 0) {
                int indices[work_size];
                for (int j = 0; j < work_size; j++) {
                    indices[j] = current_file++;
                }
                MPI_Send(indices, work_size, MPI_INT, i, 0, MPI_COMM_WORLD);
                working_processes++;
            }
        }
        
        // Handle requests for more work
        while (working_processes > 0) {
            int worker_rank;
            MPI_Recv(&worker_rank, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
            working_processes--;
            
            if (current_file < file_count) {
                int work_size = (current_file + work_unit_size <= file_count) ? 
                                work_unit_size : (file_count - current_file);
                
                MPI_Send(&work_size, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
                
                if (work_size > 0) {
                    int indices[work_size];
                    for (int j = 0; j < work_size; j++) {
                        indices[j] = current_file++;
                    }
                    MPI_Send(indices, work_size, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
                    working_processes++;
                }
            } else {
                // No more work
                int work_size = 0;
                MPI_Send(&work_size, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
            }
        }
        
        // Master process doesn't process files
        *files_to_process = 0;
    } else {
        // Worker processes
        int work_size;
        MPI_Recv(&work_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        
        *files_to_process = work_size;
        
        if (work_size > 0) {
            MPI_Recv(file_indices, work_size, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            
            // Request more work when done
            MPI_Send(&mpi_rank, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        }
    }
}

void distribute_workload(int strategy, int mpi_rank, int mpi_size, int file_count, 
                        char file_paths[][256], int *start_idx, int *end_idx) {
    
    if (strategy == STRATEGY_STATIC || strategy == STRATEGY_ADAPTIVE) {
        // For static or initial adaptive distribution
        static_distribute_workload(mpi_rank, mpi_size, file_count, file_paths, start_idx, end_idx);
    } else if (strategy == STRATEGY_DYNAMIC) {
        // For dynamic distribution, handle it differently - not using start/end indices
        // This requires restructuring the calling code to use dynamic work allocation
        int files_to_process[1];
        int file_indices[file_count];
        
        dynamic_distribute_workload(mpi_rank, mpi_size, file_count, file_paths, files_to_process, file_indices);
        
        // Convert dynamic allocation to start/end format for compatibility
        if (*files_to_process > 0) {
            *start_idx = file_indices[0];
            *end_idx = file_indices[*files_to_process - 1] + 1;
        } else {
            *start_idx = 0;
            *end_idx = 0;
        }
    }
}
