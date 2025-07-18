#ifndef MPI_CRAWLER_UTILS_H
#define MPI_CRAWLER_UTILS_H

int mpi_share_urls(char** queue, int* depth, int* front, int* rear, int max_size, 
                   int mpi_rank, int mpi_size, int (*has_visited_fn)(const char*),
                   void (*mark_visited_fn)(const char*));


void mpi_gather_stats(int local_count, int* global_count, int mpi_size);


void visualize_hybrid_structure(int mpi_rank, int mpi_size, int omp_threads);

#endif 
