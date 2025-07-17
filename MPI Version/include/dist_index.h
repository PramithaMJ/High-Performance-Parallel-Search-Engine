#ifndef DIST_INDEX_H
#define DIST_INDEX_H

#define DIST_INDEX_INITIAL_SIZE 1000
#define DIST_MAX_TERM_LENGTH 128
#define DIST_MAX_POSTINGS 1000
#define DIST_INDEX_TAG 100

// Posting entry
typedef struct {
    int doc_id;
    int freq;
} DistPosting;

// Index entry
typedef struct {
    char term[DIST_MAX_TERM_LENGTH];
    int posting_count;
    DistPosting postings[DIST_MAX_POSTINGS];
} DistIndexEntry;

// Distributed index
typedef struct {
    int mpi_rank;
    int mpi_size;
    int local_capacity;
    int local_size;
    DistIndexEntry* local_entries;
} DistributedIndex;

// Search result
typedef struct {
    int posting_count;
    DistPosting* postings;
} DistSearchResult;

// Create a distributed index
DistributedIndex* create_distributed_index(int mpi_rank, int mpi_size);

// Free a distributed index
void free_distributed_index(DistributedIndex* index);

// Add a term to the distributed index
void dist_add_term(DistributedIndex* index, const char* term, int doc_id, int freq);

// Process incoming terms
void dist_process_incoming_terms(DistributedIndex* index);

// Synchronize the distributed index
void dist_sync_index(DistributedIndex* index);

// Search for a term in the distributed index
DistSearchResult dist_search_term(DistributedIndex* index, const char* term);

// Free a search result
void free_dist_search_result(DistSearchResult* result);

#endif /* DIST_INDEX_H */
