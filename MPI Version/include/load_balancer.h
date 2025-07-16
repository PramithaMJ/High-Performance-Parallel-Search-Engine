#ifndef LOAD_BALANCER_H
#define LOAD_BALANCER_H

// Work distribution strategies
#define STRATEGY_STATIC 0
#define STRATEGY_DYNAMIC 1
#define STRATEGY_ADAPTIVE 2

// Initialize the load balancer
void init_load_balancer(int mpi_size);

// Free resources used by the load balancer
void free_load_balancer();

// Distribute workload across MPI processes
void distribute_workload(int strategy, int mpi_rank, int mpi_size, int file_count, 
                        char file_paths[][256], int *start_idx, int *end_idx);

#endif /* LOAD_BALANCER_H */
