#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/crawler.h"

#define MAX_URL_LENGTH 512
#define MPI_TAG_URL_ASSIGNMENT 100
#define MPI_TAG_URL_RESULT 101
#define MPI_TAG_WORKLOAD_REQUEST 102
#define MPI_TAG_WORKLOAD_RESPONSE 103
#define MPI_TAG_TERMINATION 104

// Structure to hold URL data for sharing between MPI processes
typedef struct {
    char url[MAX_URL_LENGTH];
    int depth;
    int valid;  // Flag to indicate if entry contains a valid URL
} SharedURL;

// Function to coordinate URL sharing between MPI processes
int mpi_share_urls(char** queue, int* depth, int* front, int* rear, int max_size, 
                   int mpi_rank, int mpi_size, int (*has_visited_fn)(const char*),
                   void (*mark_visited_fn)(const char*)) {
    if (mpi_size <= 1) return 0;
    
    // How many URLs to share in each direction
    const int MAX_SHARE_URLS = 3;
    SharedURL urls_to_send[MAX_SHARE_URLS];
    memset(urls_to_send, 0, sizeof(urls_to_send));
    
    // Extract URLs from our queue to share
    int queue_size = (*rear - *front + max_size) % max_size;
    int urls_extracted = 0;
    
    if (queue_size > MAX_SHARE_URLS * 2) {
        // Extract URLs from different parts of the queue for diversity
        int step = queue_size / (MAX_SHARE_URLS + 1);
        if (step < 1) step = 1;
        
        for (int i = 0; i < MAX_SHARE_URLS && urls_extracted < MAX_SHARE_URLS; i++) {
            int pos = (*front + (i * step)) % max_size;
            if (queue[pos] && strlen(queue[pos]) < MAX_URL_LENGTH) {
                strncpy(urls_to_send[urls_extracted].url, queue[pos], MAX_URL_LENGTH - 1);
                urls_to_send[urls_extracted].url[MAX_URL_LENGTH - 1] = '\0';
                urls_to_send[urls_extracted].depth = depth[pos];
                urls_to_send[urls_extracted].valid = 1;
                urls_extracted++;
            }
        }
        
        if (urls_extracted > 0) {
            printf("[MPI %d] Prepared %d URLs to share with other processes\n", mpi_rank, urls_extracted);
        }
    }
    
    // Determine destination process (next in ring topology)
    int dest = (mpi_rank + 1) % mpi_size;
    int source = (mpi_rank + mpi_size - 1) % mpi_size;
    
    // Send URLs to the next process in the ring
    MPI_Send(urls_to_send, sizeof(SharedURL) * MAX_SHARE_URLS, MPI_BYTE, 
             dest, MPI_TAG_URL_ASSIGNMENT, MPI_COMM_WORLD);
    
    // Receive URLs from the previous process in the ring
    SharedURL urls_received[MAX_SHARE_URLS];
    MPI_Status status;
    MPI_Recv(urls_received, sizeof(SharedURL) * MAX_SHARE_URLS, MPI_BYTE, 
             source, MPI_TAG_URL_ASSIGNMENT, MPI_COMM_WORLD, &status);
    
    // Process received URLs
    int urls_added = 0;
    for (int i = 0; i < MAX_SHARE_URLS; i++) {
        if (urls_received[i].valid && urls_received[i].url[0] != '\0') {
            // Check if we have space in the queue and if we haven't already visited this URL
            if ((*rear + 1) % max_size != *front && !(*has_visited_fn)(urls_received[i].url)) {
                // Add to queue
                char* url_copy = strdup(urls_received[i].url);
                if (url_copy) {
                    queue[*rear] = url_copy;
                    depth[*rear] = urls_received[i].depth;
                    *rear = (*rear + 1) % max_size;
                    (*mark_visited_fn)(url_copy);
                    urls_added++;
                }
            }
        }
    }
    
    if (urls_added > 0) {
        printf("[MPI %d] Received and added %d URLs from process %d\n", 
               mpi_rank, urls_added, source);
    }
    
    return urls_added;
}

// Function to gather global crawl statistics across all MPI processes
void mpi_gather_stats(int local_count, int* global_count, int mpi_size) {
    int all_counts[mpi_size];
    
    // Gather all counts to all processes
    MPI_Allgather(&local_count, 1, MPI_INT, all_counts, 1, MPI_INT, MPI_COMM_WORLD);
    
    // Sum up the total
    *global_count = 0;
    for (int i = 0; i < mpi_size; i++) {
        *global_count += all_counts[i];
    }
}

// Function to visualize MPI and OpenMP hierarchy for better understanding
void visualize_hybrid_structure(int mpi_rank, int mpi_size, int omp_threads) {
    // Only rank 0 prints the visualization
    if (mpi_rank == 0) {
        printf("\n╔══════════════════════════════════════════════╗\n");
        printf("║     Hybrid Parallel Crawling Architecture     ║\n");
        printf("╠══════════════════════════════════════════════╣\n");
        printf("║ Total MPI Processes: %-3d                     ║\n", mpi_size);
        printf("║ OpenMP Threads per Process: %-3d              ║\n", omp_threads);
        printf("║ Total Parallel Units: %-3d                    ║\n", mpi_size * omp_threads);
        printf("╚══════════════════════════════════════════════╝\n");
        
        printf("\nParallel Structure:\n");
        for (int i = 0; i < mpi_size; i++) {
            printf("MPI Process %d: [", i);
            for (int j = 0; j < omp_threads; j++) {
                printf("T%d%s", j, (j < omp_threads-1) ? "|" : "");
            }
            printf("]\n");
        }
        printf("\n");
    }
    
    // Give time for visualization to be seen
    MPI_Barrier(MPI_COMM_WORLD);
}
