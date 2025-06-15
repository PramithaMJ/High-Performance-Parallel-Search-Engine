#include "../include/index.h"
#include "../include/parser.h"
#include "../include/metrics.h"
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h> // For free function
#include <string.h>
#include <libgen.h> // For basename function
#include <omp.h>    // OpenMP header

InvertedIndex index_data[10000];
int index_size = 0;
int doc_lengths[1000] = {0};
Document documents[1000]; // Array to store document filenames

// Thread-safe locks for critical sections
omp_lock_t index_lock;
omp_lock_t doc_length_lock;

// Initialize OpenMP locks
void init_locks()
{
    omp_init_lock(&index_lock);
    omp_init_lock(&doc_length_lock);
}

// Destroy OpenMP locks
void destroy_locks()
{
    omp_destroy_lock(&index_lock);
    omp_destroy_lock(&doc_length_lock);
}

int build_index(const char *folder_path)
{
    // Initialize locks
    init_locks();

    // Start measuring indexing time
    start_timer();

    printf("Opening directory: %s\n", folder_path);
    DIR *dir = opendir(folder_path);
    if (!dir)
    {
        printf("Error: Could not open directory: %s\n", folder_path);
        destroy_locks();
        return 0;
    }

    // First pass: collect all file names
    struct dirent *entry;
    char file_paths[1000][256]; // Assuming max 1000 files
    int file_count = 0;

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

    // Parallel file processing
    int successful_docs = 0;

    #pragma omp parallel for schedule(dynamic) reduction(+ : successful_docs)
    for (int i = 0; i < file_count; i++)
    {
        int thread_id = omp_get_thread_num();
        printf("Thread %d processing file: %s\n", thread_id, file_paths[i]);

        if (parse_file_parallel(file_paths[i], i))
        {
            // Store the filename (basename) for this document
            char *path_copy = strdup(file_paths[i]);
            char *filename = basename(path_copy);

            // Critical section for updating document metadata
            #pragma omp critical(doc_metadata)
            {
                strncpy(documents[i].filename, filename, MAX_FILENAME_LEN - 1);
                documents[i].filename[MAX_FILENAME_LEN - 1] = '\0';
            }

            free(path_copy);
            printf("Thread %d successfully parsed file: %s (doc_id: %d)\n",
                   thread_id, file_paths[i], i);
            successful_docs++;
        }
        else
        {
            printf("Thread %d failed to parse file: %s\n", thread_id, file_paths[i]);
        }
    }

    // Record indexing time
    metrics.indexing_time = stop_timer();
    printf("Indexing completed for %d documents in %.2f ms using %d threads\n",
           successful_docs, metrics.indexing_time, omp_get_max_threads());

    // Update index statistics
    update_index_stats(successful_docs, metrics.total_tokens, index_size);

    // Cleanup locks
    destroy_locks();

    return successful_docs;
}

// Thread-safe version of add_token
void add_token(const char *token, int doc_id)
{
    // Skip empty tokens or tokens that are too long
    if (!token || strlen(token) == 0 || strlen(token) > 100)
    {
        return;
    }

// Count the token for metrics (atomic operation)
    #pragma omp atomic
    metrics.total_tokens++;

    // Use lock for index operations to ensure thread safety
    omp_set_lock(&index_lock);

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

    omp_unset_lock(&index_lock);

    // Update document length (thread-safe)
    omp_set_lock(&doc_length_lock);
    doc_lengths[doc_id]++;
    omp_unset_lock(&doc_length_lock);
}

// Optimized parallel version for batch token processing
void add_tokens_batch(const char **tokens, int *doc_ids, int count)
{
    #pragma omp parallel for schedule(dynamic, 10)
    for (int i = 0; i < count; i++)
    {
        add_token(tokens[i], doc_ids[i]);
    }
}

// Thread-safe getter functions
int get_doc_length(int doc_id)
{
    int length;
    omp_set_lock(&doc_length_lock);
    length = doc_lengths[doc_id];
    omp_unset_lock(&doc_length_lock);
    return length;
}

int get_doc_count()
{
    int count;
    omp_set_lock(&index_lock);
    count = index_size;
    omp_unset_lock(&index_lock);
    return count;
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

// Function to clear the index for rebuilding
void clear_index()
{
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            // Reset index data
            index_size = 0;
        }

        #pragma omp section
        {
            // Reset document lengths
            memset(doc_lengths, 0, sizeof(doc_lengths));
        }

        #pragma omp section
        {
            // Reset documents array
            memset(documents, 0, sizeof(documents));
        }
    }
}

// Parallel version of print_index with better performance
void print_index()
{
    printf("Inverted Index Contents:\n");
    printf("Total Terms: %d\n", index_size);

// Use parallel sections for different parts of printing if needed
    #pragma omp parallel for schedule(static, 10) if (index_size > 100)
    for (int i = 0; i < (index_size < 50 ? index_size : 30); i++)
    {
        #pragma omp critical(print_output)
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
    }

    if (index_size > 50)
    {
        printf("... and %d more terms\n", index_size - 30);
    }
}

// Parallel search function for better query performance
int parallel_search_term(const char *term, Posting **results, int *result_count)
{
    *results = NULL;
    *result_count = 0;

    int found = -1;

    #pragma omp parallel for
    for (int i = 0; i < index_size; i++)
    {
        if (strcmp(index_data[i].term, term) == 0)
        {
            #pragma omp critical(search_result)
            {
                if (found == -1)
                { // Only set if not already found
                    found = i;
                }
            }
        }
    }

    if (found != -1)
    {
        *results = index_data[found].postings;
        *result_count = index_data[found].posting_count;
        return 1;
    }

    return 0;
}

// Function to set the number of threads
void set_thread_count(int num_threads)
{
    omp_set_num_threads(num_threads);
    printf("Set OpenMP thread count to: %d\n", num_threads);
}

// Function to get current thread information
void print_thread_info()
{
    printf("OpenMP Information:\n");
    printf("  Max threads available: %d\n", omp_get_max_threads());
    printf("  Number of processors: %d\n", omp_get_num_procs());

    #pragma omp parallel
    {
        #pragma omp single
        {
            printf("  Current number of threads in parallel region: %d\n", omp_get_num_threads());
        }
    }
}