#ifndef MPI_CRAWLER_UTILS_H
#define MPI_CRAWLER_UTILS_H

/**
 * Share URLs between MPI processes for better load balancing
 * 
 * @param queue URL queue array
 * @param depth URL depth array corresponding to queue
 * @param front Current front index of the queue
 * @param rear Current rear index of the queue
 * @param max_size Maximum size of the queue
 * @param mpi_rank Current MPI process rank
 * @param mpi_size Total number of MPI processes
 * @param has_visited_fn Function pointer to check if URL has been visited
 * @param mark_visited_fn Function pointer to mark URL as visited
 * @return Number of URLs added to the queue
 */
int mpi_share_urls(char** queue, int* depth, int* front, int* rear, int max_size, 
                   int mpi_rank, int mpi_size, int (*has_visited_fn)(const char*),
                   void (*mark_visited_fn)(const char*));

/**
 * Gather global crawl statistics across all MPI processes
 * 
 * @param local_count Local page count for this process
 * @param global_count Pointer to store the global count
 * @param mpi_size Total number of MPI processes
 */
void mpi_gather_stats(int local_count, int* global_count, int mpi_size);

/**
 * Visualize the hybrid MPI + OpenMP structure
 * 
 * @param mpi_rank Current MPI process rank
 * @param mpi_size Total number of MPI processes
 * @param omp_threads Number of OpenMP threads per process
 */
void visualize_hybrid_structure(int mpi_rank, int mpi_size, int omp_threads);

#endif // MPI_CRAWLER_UTILS_H
