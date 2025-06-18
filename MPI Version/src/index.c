#include "../include/index.h"
#include "../include/parser.h"
#include "../include/metrics.h"
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h> // For free function
#include <string.h>
#include <libgen.h> // For basename function
#include <mpi.h>    // MPI header

InvertedIndex index_data[10000];
int index_size = 0;
int doc_lengths[1000] = {0};
Document documents[1000]; // Array to store document filenames

// MPI process info
int mpi_rank = 0;
int mpi_size = 1;

// Initialize MPI info
void init_mpi_info() 
{
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
}

// No locks needed in MPI version as processes have separate memory spaces
void init_locks() { /* No operation needed */ }
void destroy_locks() { /* No operation needed */ }

int build_index(const char *folder_path)
{
    // Initialize MPI info
    init_mpi_info();

    // Start measuring indexing time (only on rank 0)
    if (mpi_rank == 0) {
        start_timer();
        printf("Opening directory: %s\n", folder_path);
    }
    
    // Only the root process (rank 0) reads the directory
    char file_paths[1000][256]; // Assuming max 1000 files
    int file_count = 0;
    
    if (mpi_rank == 0) {
        DIR *dir = opendir(folder_path);
        if (!dir)
        {
            printf("Error: Could not open directory: %s\n", folder_path);
            return 0;
        }

        // First pass: collect all file names
        struct dirent *entry;

        while ((entry = readdir(dir)) != NULL && file_count < 1000)
        {
            // Process all files except hidden files (those starting with .)
            if (entry->d_name[0] != '.')
            {
                snprintf(file_paths[file_count], sizeof(file_paths[file_count]),
                         "%s/%s", folder_path, entry->d_name);
                file_count++;
            }
        }
        closedir(dir);
        printf("Found %d files to process\n", file_count);
    }
    
    // Broadcast the file count to all processes
    MPI_Bcast(&file_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Broadcast all file paths to all processes
    MPI_Bcast(file_paths, file_count * 256, MPI_CHAR, 0, MPI_COMM_WORLD);
    
    // Calculate work distribution for each process
    int files_per_process = file_count / mpi_size;
    int remaining_files = file_count % mpi_size;
    int start_idx = mpi_rank * files_per_process + (mpi_rank < remaining_files ? mpi_rank : remaining_files);
    int end_idx = start_idx + files_per_process + (mpi_rank < remaining_files ? 1 : 0);
    
    // Parallel file processing with MPI
    int successful_docs = 0;
    int local_successful = 0;

    for (int i = start_idx; i < end_idx; i++)
    {
        printf("Process %d processing file: %s\n", mpi_rank, file_paths[i]);

        if (parse_file_parallel(file_paths[i], i))
        {
            // Store the filename (basename) for this document
            char *path_copy = strdup(file_paths[i]);
            char *filename = basename(path_copy);

            // No need for critical section in MPI - processes have separate memory
            strncpy(documents[i].filename, filename, MAX_FILENAME_LEN - 1);
            documents[i].filename[MAX_FILENAME_LEN - 1] = '\0';

            free(path_copy);
            printf("Process %d successfully parsed file: %s (doc_id: %d)\n",
                   mpi_rank, file_paths[i], i);
            local_successful++;
        }
        else
        {
            printf("Process %d failed to parse file: %s\n", mpi_rank, file_paths[i]);
        }
    }
    
    // Sum up successful documents across all processes
    MPI_Reduce(&local_successful, &successful_docs, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Record indexing time
    metrics.indexing_time = stop_timer();
    printf("Indexing completed for %d documents in %.2f ms using %d MPI processes\n",
           successful_docs, metrics.indexing_time, mpi_size);

    // Update index statistics
    update_index_stats(successful_docs, metrics.total_tokens, index_size);

    // Cleanup locks
    destroy_locks();

    return successful_docs;
}

// MPI-compatible version of add_token
void add_token(const char *token, int doc_id)
{
    // Skip empty tokens or tokens that are too long
    if (!token || strlen(token) == 0 || strlen(token) > 100)
    {
        return;
    }

    // Count the token for metrics (no need for atomic operation in MPI)
    metrics.total_tokens++;

    // In MPI, each process has its own memory space, so no locks needed
    // Search for existing term
    int found = -1;
    for (int i = 0; i < index_size; ++i)
    {
        if (strcmp(index_data[i].term, token) == 0)
        {
            found = i;
            break;
        }
    }

    if (found != -1)
    {
        // Check if we already have this document in the postings list
        int doc_found = -1;
        for (int j = 0; j < index_data[found].posting_count; ++j)
        {
            if (index_data[found].postings[j].doc_id == doc_id)
            {
                doc_found = j;
                break;
            }
        }

        if (doc_found != -1)
        {
            index_data[found].postings[doc_found].freq++;
        }
        else if (index_data[found].posting_count < 1000)
        {
            index_data[found].postings[index_data[found].posting_count++] = (Posting){doc_id, 1};
        }
    }
    else if (index_size < 10000 && strlen(token) < sizeof(index_data[0].term) - 1)
    {
        // Add new term
        strcpy(index_data[index_size].term, token);
        index_data[index_size].postings[0] = (Posting){doc_id, 1};
        index_data[index_size].posting_count = 1;
        index_size++;
    }

    // Update document length (no locks needed in MPI)
    doc_lengths[doc_id]++;
}

// MPI-optimized version for batch token processing
void add_tokens_batch(const char **tokens, int *doc_ids, int count)
{
    // Calculate work distribution for each process
    int tokens_per_process = count / mpi_size;
    int remaining_tokens = count % mpi_size;
    int start_idx = mpi_rank * tokens_per_process + (mpi_rank < remaining_tokens ? mpi_rank : remaining_tokens);
    int end_idx = start_idx + tokens_per_process + (mpi_rank < remaining_tokens ? 1 : 0);
    
    // Each process handles its portion of tokens
    for (int i = start_idx; i < end_idx; i++)
    {
        add_token(tokens[i], doc_ids[i]);
    }
    
    // Synchronize all processes before continuing
    MPI_Barrier(MPI_COMM_WORLD);
}

// MPI-compatible getter functions (no locks needed)
int get_doc_length(int doc_id)
{
    // In MPI, each process has its own memory space
    return doc_lengths[doc_id];
}

int get_doc_count()
{
    // In MPI, each process has its own copy of index_size
    return index_size;
}

// Function to get the filename for a document ID (thread-safe)
const char *get_doc_filename(int doc_id)
{
    if (doc_id >= 0 && doc_id < 1000)
    {
        return documents[doc_id].filename;
    }
    return "Unknown Document";
}

// Function to clear the index for rebuilding with MPI
void clear_index()
{
    // Each process resets its own copy of the data
    
    // Reset index data
    index_size = 0;
    
    // Reset document lengths
    memset(doc_lengths, 0, sizeof(doc_lengths));
    
    // Reset documents array
    memset(documents, 0, sizeof(documents));
    
    // Make sure all processes complete the reset before continuing
    MPI_Barrier(MPI_COMM_WORLD);
}

// MPI version of print_index
void print_index()
{
    // Only rank 0 prints the index to avoid duplicate output
    if (mpi_rank == 0) {
        printf("Inverted Index Contents:\n");
        printf("Total Terms: %d\n", index_size);

        for (int i = 0; i < (index_size < 50 ? index_size : 30); i++)
        {
            printf("Term: '%s' (%d docs)\n", index_data[i].term, index_data[i].posting_count);
            printf("  Postings: ");

            for (int j = 0; j < index_data[i].posting_count; j++)
            {
                printf("(doc:%d, freq:%d) ",
                       index_data[i].postings[j].doc_id,
                       index_data[i].postings[j].freq);

                if (j > 5 && index_data[i].posting_count > 10)
                {
                    printf("... and %d more", index_data[i].posting_count - j - 1);
                    break;
                }
            }
            printf("\n");
        }

        if (index_size > 50)
        {
            printf("... and %d more terms\n", index_size - 30);
        }
    }
    
    // Make sure all processes wait until printing is done
    MPI_Barrier(MPI_COMM_WORLD);
}

// Parallel search function for better query performance
int parallel_search_term(const char *term, Posting **results, int *result_count)
{
    *results = NULL;
    *result_count = 0;

    int local_found = -1;
    int global_found = -1;
    
    // Divide the index range among processes
    int terms_per_process = index_size / mpi_size;
    int remaining_terms = index_size % mpi_size;
    int start_idx = mpi_rank * terms_per_process + (mpi_rank < remaining_terms ? mpi_rank : remaining_terms);
    int end_idx = start_idx + terms_per_process + (mpi_rank < remaining_terms ? 1 : 0);
    
    // Each process searches its portion of the index
    for (int i = start_idx; i < end_idx && i < index_size; i++)
    {
        if (strcmp(index_data[i].term, term) == 0)
        {
            local_found = i;
            break;
        }
    }
    
    // Gather all found indices to rank 0
    int *all_found = NULL;
    if (mpi_rank == 0) {
        all_found = (int*)malloc(mpi_size * sizeof(int));
    }
    
    MPI_Gather(&local_found, 1, MPI_INT, all_found, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Process 0 determines the final result
    if (mpi_rank == 0) {
        for (int i = 0; i < mpi_size; i++) {
            if (all_found[i] != -1) {
                global_found = all_found[i];
                break;
            }
        }
        free(all_found);
    }
    
    // Broadcast the result to all processes
    MPI_Bcast(&global_found, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (global_found != -1)
    {
        *results = index_data[global_found].postings;
        *result_count = index_data[global_found].posting_count;
        return 1;
    }

    return 0;
}

// Function to set the thread count (no-op in MPI version)
void set_thread_count(int num_threads)
{
    if (mpi_rank == 0) {
        printf("Note: Thread count setting is ignored in MPI version. Use mpirun -np <NUM_PROCESSES> instead.\n");
    }
}

// Function to print MPI process information
void print_thread_info()
{
    // Only rank 0 prints the overall info
    if (mpi_rank == 0) {
        printf("MPI Information:\n");
        printf("  Total number of MPI processes: %d\n", mpi_size);
    }
    
    // Each process prints its own ID
    printf("  Process %d of %d running on host: ", mpi_rank, mpi_size);
    
    // Get hostname
    char hostname[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(hostname, &name_len);
    printf("%s\n", hostname);
    
    // Synchronize output
    MPI_Barrier(MPI_COMM_WORLD);
}

// Function to distribute files efficiently across nodes
void distribute_files_across_nodes(const char* folder_path, 
                                 char file_paths[1000][256], 
                                 int* file_count, 
                                 int* start_idx, 
                                 int* end_idx) {
    // Initialize counters
    *file_count = 0;
    
    // Only root process reads the directory
    if (mpi_rank == 0) {
        DIR *dir = opendir(folder_path);
        if (!dir) {
            printf("Error: Could not open directory: %s\n", folder_path);
            *file_count = 0;
            return;
        }

        struct dirent *entry;
        while ((entry = readdir(dir)) != NULL && *file_count < 1000) {
            // Process all files except hidden files (those starting with .)
            if (entry->d_name[0] != '.') {
                snprintf(file_paths[*file_count], 256,
                         "%s/%s", folder_path, entry->d_name);
                (*file_count)++;
            }
        }
        closedir(dir);
        printf("Found %d files to process\n", *file_count);
    }
    
    // Broadcast the file count to all processes
    MPI_Bcast(file_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Broadcast all file paths to all processes
    MPI_Bcast(file_paths, (*file_count) * 256, MPI_CHAR, 0, MPI_COMM_WORLD);
    
    // Calculate optimal work distribution based on node ID
    int files_per_process = *file_count / mpi_size;
    int remaining_files = *file_count % mpi_size;
    
    // Each process gets its portion of files with load balancing
    *start_idx = mpi_rank * files_per_process + (mpi_rank < remaining_files ? mpi_rank : remaining_files);
    *end_idx = *start_idx + files_per_process + (mpi_rank < remaining_files ? 1 : 0);
    
    printf("Process %d on node [%s] will handle files %d to %d\n", 
           mpi_rank, get_node_name(), *start_idx, *end_idx - 1);
}

// Helper function to get node name
char* get_node_name() {
    static char hostname[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(hostname, &name_len);
    return hostname;
}