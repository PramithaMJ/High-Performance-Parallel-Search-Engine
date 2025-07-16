#include "../include/mpi_comm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

// Buffer for non-blocking communication
static void *comm_buffer = NULL;
static size_t comm_buffer_size = 0;
static MPI_Request *request_pool = NULL;
static int request_count = 0;
static int max_requests = 32;

void init_mpi_comm() {
    // Initialize the communication buffer with a default size
    comm_buffer_size = 1024 * 1024; // 1MB default
    comm_buffer = malloc(comm_buffer_size);
    
    // Create request pool for non-blocking operations
    request_pool = (MPI_Request*)malloc(max_requests * sizeof(MPI_Request));
    request_count = 0;
}

void free_mpi_comm() {
    if (comm_buffer) {
        free(comm_buffer);
        comm_buffer = NULL;
    }
    
    if (request_pool) {
        free(request_pool);
        request_pool = NULL;
    }
}

void wait_all_requests() {
    if (request_count > 0) {
        MPI_Status *statuses = (MPI_Status*)malloc(request_count * sizeof(MPI_Status));
        MPI_Waitall(request_count, request_pool, statuses);
        free(statuses);
        request_count = 0;
    }
}

int add_request(MPI_Request request) {
    if (request_count >= max_requests) {
        // Resize the request pool
        max_requests *= 2;
        request_pool = (MPI_Request*)realloc(request_pool, max_requests * sizeof(MPI_Request));
        if (!request_pool) {
            fprintf(stderr, "Failed to resize request pool\n");
            return -1;
        }
    }
    
    request_pool[request_count] = request;
    return request_count++;
}

// Optimized scatter for distributing work
void optimized_scatter(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                     void *recvbuf, int recvcount, MPI_Datatype recvtype,
                     int root, MPI_Comm comm) {
    
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    // For small messages, use standard scatter
    int type_size;
    MPI_Type_size(sendtype, &type_size);
    size_t total_size = sendcount * size * type_size;
    
    if (total_size < 1024 * 1024) { // Less than 1MB
        MPI_Scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);
        return;
    }
    
    // For large messages, use non-blocking point-to-point communications
    if (rank == root) {
        MPI_Request requests[size - 1];
        int req_idx = 0;
        
        // Copy root's portion directly
        memcpy(recvbuf, (char*)sendbuf + rank * sendcount * type_size, recvcount * type_size);
        
        // Send to other processes
        for (int i = 0; i < size; i++) {
            if (i != root) {
                MPI_Isend((char*)sendbuf + i * sendcount * type_size, 
                         sendcount, sendtype, i, 0, comm, &requests[req_idx++]);
            }
        }
        
        // Wait for all sends to complete
        MPI_Waitall(size - 1, requests, MPI_STATUSES_IGNORE);
    } else {
        // Receive from root
        MPI_Recv(recvbuf, recvcount, recvtype, root, 0, comm, MPI_STATUS_IGNORE);
    }
}

// Optimized gather for collecting results
void optimized_gather(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                    void *recvbuf, int recvcount, MPI_Datatype recvtype,
                    int root, MPI_Comm comm) {
    
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    
    // For small messages, use standard gather
    int type_size;
    MPI_Type_size(sendtype, &type_size);
    size_t total_size = recvcount * size * type_size;
    
    if (total_size < 1024 * 1024) { // Less than 1MB
        MPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm);
        return;
    }
    
    // For large messages, use non-blocking point-to-point communications
    if (rank == root) {
        MPI_Request requests[size - 1];
        int req_idx = 0;
        
        // Copy root's portion directly
        memcpy((char*)recvbuf + rank * recvcount * type_size, sendbuf, sendcount * type_size);
        
        // Receive from other processes
        for (int i = 0; i < size; i++) {
            if (i != root) {
                MPI_Irecv((char*)recvbuf + i * recvcount * type_size, 
                         recvcount, recvtype, i, 0, comm, &requests[req_idx++]);
            }
        }
        
        // Wait for all receives to complete
        MPI_Waitall(size - 1, requests, MPI_STATUSES_IGNORE);
    } else {
        // Send to root
        MPI_Send(sendbuf, sendcount, sendtype, root, 0, comm);
    }
}
