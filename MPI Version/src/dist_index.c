#include "../include/dist_index.h"
#include "../include/mpi_comm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

// Hash function for term distribution
static unsigned int hash_string(const char *str) {
    unsigned int hash = 5381;
    int c;
    
    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + c;
    }
    
    return hash;
}

// Initialize a distributed index
DistributedIndex* create_distributed_index(int mpi_rank, int mpi_size) {
    DistributedIndex* index = (DistributedIndex*)malloc(sizeof(DistributedIndex));
    
    if (!index) {
        fprintf(stderr, "Failed to allocate distributed index\n");
        return NULL;
    }
    
    // Initialize index
    index->mpi_rank = mpi_rank;
    index->mpi_size = mpi_size;
    index->local_capacity = DIST_INDEX_INITIAL_SIZE;
    index->local_size = 0;
    
    // Allocate local entries
    index->local_entries = (DistIndexEntry*)malloc(
        index->local_capacity * sizeof(DistIndexEntry));
    
    if (!index->local_entries) {
        fprintf(stderr, "Failed to allocate local entries\n");
        free(index);
        return NULL;
    }
    
    // Initialize local entries
    for (int i = 0; i < index->local_capacity; i++) {
        index->local_entries[i].term[0] = '\0';
        index->local_entries[i].posting_count = 0;
    }
    
    return index;
}

void free_distributed_index(DistributedIndex* index) {
    if (!index) return;
    
    if (index->local_entries) {
        free(index->local_entries);
    }
    
    free(index);
}

// Determine which process should own a term
int get_term_owner(DistributedIndex* index, const char* term) {
    unsigned int hash = hash_string(term);
    return hash % index->mpi_size;
}

// Add a term to the distributed index
void dist_add_term(DistributedIndex* index, const char* term, int doc_id, int freq) {
    int owner = get_term_owner(index, term);
    
    if (owner == index->mpi_rank) {
        int found = -1;
        
        for (int i = 0; i < index->local_size; i++) {
            if (strcmp(index->local_entries[i].term, term) == 0) {
                found = i;
                break;
            }
        }
        
        if (found != -1) {
            DistIndexEntry* entry = &index->local_entries[found];
            
            for (int i = 0; i < entry->posting_count; i++) {
                if (entry->postings[i].doc_id == doc_id) {
                    entry->postings[i].freq += freq;
                    return;
                }
            }
            
            if (entry->posting_count < DIST_MAX_POSTINGS) {
                entry->postings[entry->posting_count].doc_id = doc_id;
                entry->postings[entry->posting_count].freq = freq;
                entry->posting_count++;
            }
        } else {
            if (index->local_size >= index->local_capacity) {
                int new_capacity = index->local_capacity * 2;
                DistIndexEntry* new_entries = (DistIndexEntry*)realloc(
                    index->local_entries, new_capacity * sizeof(DistIndexEntry));
                
                if (!new_entries) {
                    fprintf(stderr, "Failed to resize local entries\n");
                    return;
                }
                
                index->local_entries = new_entries;
                index->local_capacity = new_capacity;
                
                for (int i = index->local_size; i < index->local_capacity; i++) {
                    index->local_entries[i].term[0] = '\0';
                    index->local_entries[i].posting_count = 0;
                }
            }
            
            // Add new term
            DistIndexEntry* entry = &index->local_entries[index->local_size];
            strncpy(entry->term, term, DIST_MAX_TERM_LENGTH - 1);
            entry->term[DIST_MAX_TERM_LENGTH - 1] = '\0';
            entry->posting_count = 1;
            entry->postings[0].doc_id = doc_id;
            entry->postings[0].freq = freq;
            index->local_size++;
        }
    } else {

        char buffer[DIST_MAX_TERM_LENGTH + 2 * sizeof(int)];
        
        strncpy(buffer, term, DIST_MAX_TERM_LENGTH - 1);
        buffer[DIST_MAX_TERM_LENGTH - 1] = '\0';
        
        memcpy(buffer + DIST_MAX_TERM_LENGTH, &doc_id, sizeof(int));
        memcpy(buffer + DIST_MAX_TERM_LENGTH + sizeof(int), &freq, sizeof(int));
        
        MPI_Request request;
        MPI_Isend(buffer, DIST_MAX_TERM_LENGTH + 2 * sizeof(int), MPI_CHAR, 
                 owner, DIST_INDEX_TAG, MPI_COMM_WORLD, &request);
        add_request(request);
    }
}

// Process incoming terms
void dist_process_incoming_terms(DistributedIndex* index) {
    // Process all pending messages
    int flag;
    MPI_Status status;
    char buffer[DIST_MAX_TERM_LENGTH + 2 * sizeof(int)];
    
    // Check for incoming messages
    MPI_Iprobe(MPI_ANY_SOURCE, DIST_INDEX_TAG, MPI_COMM_WORLD, &flag, &status);
    
    while (flag) {
        // Receive message
        MPI_Recv(buffer, DIST_MAX_TERM_LENGTH + 2 * sizeof(int), MPI_CHAR,
                status.MPI_SOURCE, DIST_INDEX_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Extract term, doc_id, and freq
        char term[DIST_MAX_TERM_LENGTH];
        int doc_id, freq;
        
        strncpy(term, buffer, DIST_MAX_TERM_LENGTH - 1);
        term[DIST_MAX_TERM_LENGTH - 1] = '\0';
        
        memcpy(&doc_id, buffer + DIST_MAX_TERM_LENGTH, sizeof(int));
        memcpy(&freq, buffer + DIST_MAX_TERM_LENGTH + sizeof(int), sizeof(int));
        
        // Add term to local index
        dist_add_term(index, term, doc_id, freq);
        
        // Check for more messages
        MPI_Iprobe(MPI_ANY_SOURCE, DIST_INDEX_TAG, MPI_COMM_WORLD, &flag, &status);
    }
}

// Synchronize the distributed index
void dist_sync_index(DistributedIndex* index) {
    dist_process_incoming_terms(index);
    
    wait_all_requests();
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Gather index statistics
    int local_size = index->local_size;
    int *all_sizes = NULL;
    
    if (index->mpi_rank == 0) {
        all_sizes = (int*)malloc(index->mpi_size * sizeof(int));
    }
    
    MPI_Gather(&local_size, 1, MPI_INT, all_sizes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (index->mpi_rank == 0) {
        int total_entries = 0;
        for (int i = 0; i < index->mpi_size; i++) {
            total_entries += all_sizes[i];
        }
        
        printf("Distributed index synchronized: %d total entries across %d processes\n",
               total_entries, index->mpi_size);
        
        free(all_sizes);
    }
}

// Search for a term in the distributed index
DistSearchResult dist_search_term(DistributedIndex* index, const char* term) {
    DistSearchResult result = {0, NULL};
    int owner = get_term_owner(index, term);
    
    if (owner == index->mpi_rank) {
        // Search local index
        for (int i = 0; i < index->local_size; i++) {
            if (strcmp(index->local_entries[i].term, term) == 0) {
                result.posting_count = index->local_entries[i].posting_count;
                result.postings = (DistPosting*)malloc(
                    result.posting_count * sizeof(DistPosting));
                
                for (int j = 0; j < result.posting_count; j++) {
                    result.postings[j] = index->local_entries[i].postings[j];
                }
                
                break;
            }
        }
        
        MPI_Bcast(&result.posting_count, 1, MPI_INT, owner, MPI_COMM_WORLD);
        
        if (result.posting_count > 0) {
            MPI_Bcast(result.postings, result.posting_count * sizeof(DistPosting),
                     MPI_CHAR, owner, MPI_COMM_WORLD);
        }
    } else {
        // First receive posting count
        MPI_Bcast(&result.posting_count, 1, MPI_INT, owner, MPI_COMM_WORLD);
        
        if (result.posting_count > 0) {
            // Allocate memory for postings
            result.postings = (DistPosting*)malloc(
                result.posting_count * sizeof(DistPosting));
            
            // Receive postings
            MPI_Bcast(result.postings, result.posting_count * sizeof(DistPosting),
                     MPI_CHAR, owner, MPI_COMM_WORLD);
        }
    }
    
    return result;
}

void free_dist_search_result(DistSearchResult* result) {
    if (result->postings) {
        free(result->postings);
        result->postings = NULL;
    }
    
    result->posting_count = 0;
}
