#ifndef MPI_COMM_H
#define MPI_COMM_H

#include <mpi.h>

// Initialize MPI communication utilities
void init_mpi_comm();

// Free MPI communication resources
void free_mpi_comm();

// Wait for all pending non-blocking requests to complete
void wait_all_requests();

// Add a request to the pool
int add_request(MPI_Request request);

// Optimized scatter implementation for large datasets
void optimized_scatter(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                     void *recvbuf, int recvcount, MPI_Datatype recvtype,
                     int root, MPI_Comm comm);

// Optimized gather implementation for large datasets
void optimized_gather(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                    void *recvbuf, int recvcount, MPI_Datatype recvtype,
                    int root, MPI_Comm comm);

#endif /* MPI_COMM_H */
