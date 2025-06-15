#include "../include/index.h"
#include "../include/parser.h"
#include "../include/metrics.h"
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h> // For free function
#include <string.h>
#include <libgen.h> // For basename function

InvertedIndex index_data[10000];
int index_size = 0;
int doc_lengths[1000] = {0};
Document documents[1000]; // Array to store document filenames

int build_index(const char *folder_path)
{
    // Start measuring indexing time
    start_timer();
    
    printf("Opening directory: %s\n", folder_path);
    DIR *dir = opendir(folder_path);
    if (!dir) {
        printf("Error: Could not open directory: %s\n", folder_path);
        return 0;
    }
    struct dirent *entry;
    int doc_id = 0;

    while ((entry = readdir(dir)) != NULL)
    {
        // Process all files except hidden files (those starting with .)
        if (entry->d_name[0] != '.')
        {
            char path[256];
            snprintf(path, sizeof(path), "%s/%s", folder_path, entry->d_name);
            printf("Processing file: %s\n", path);
            if (parse_file(path, doc_id))
            {
                // Store the filename (basename) for this document
                // Use a safer approach to get the filename
                char *path_copy = strdup(path);
                char *filename = basename(path_copy);
                strncpy(documents[doc_id].filename, filename, MAX_FILENAME_LEN - 1);
                documents[doc_id].filename[MAX_FILENAME_LEN - 1] = '\0';
                free(path_copy); // Free the duplicated path
                
                printf("Successfully parsed file: %s (doc_id: %d)\n", path, doc_id);
                doc_id++;
            }
            else
            {
                printf("Failed to parse file: %s\n", path);
            }
        }
    }

    closedir(dir);
    
    // Record indexing time
    metrics.indexing_time = stop_timer();
    printf("Indexing completed for %d documents in %.2f ms\n", doc_id, metrics.indexing_time);
    
    // Update index statistics
    update_index_stats(doc_id, metrics.total_tokens, index_size);
    
    return doc_id;
}

void add_token(const char *token, int doc_id)
{
    // Skip empty tokens or tokens that are too long
    if (!token || strlen(token) == 0 || strlen(token) > 100) {
        return;
    }
    
    // Count the token for metrics
    metrics.total_tokens++;

    for (int i = 0; i < index_size; ++i)
    {
        if (strcmp(index_data[i].term, token) == 0)
        {
            // Check if we already have this document in the postings list
            for (int j = 0; j < index_data[i].posting_count; ++j)
            {
                if (index_data[i].postings[j].doc_id == doc_id)
                {
                    index_data[i].postings[j].freq++;
                    doc_lengths[doc_id]++;
                    return;
                }
            }
            
            // Make sure we don't exceed the array size
            if (index_data[i].posting_count < 1000) { // assuming max 1000 docs per term
                index_data[i].postings[index_data[i].posting_count++] = (Posting){doc_id, 1};
                doc_lengths[doc_id]++;
            }
            return;
        }
    }
    
    // Make sure we don't exceed our index capacity
    if (index_size < 10000) {
        // Make sure term isn't too long for our buffer
        if (strlen(token) < sizeof(index_data[0].term) - 1) {
            strcpy(index_data[index_size].term, token);
            index_data[index_size].postings[0] = (Posting){doc_id, 1};
            index_data[index_size].posting_count = 1;
            doc_lengths[doc_id]++;
            index_size++;
        }
    }
}

int get_doc_length(int doc_id)
{
    return doc_lengths[doc_id];
}

int get_doc_count()
{
    return index_size;
}

// Function to get the filename for a document ID
const char* get_doc_filename(int doc_id)
{
    if (doc_id >= 0 && doc_id < 1000) {
        return documents[doc_id].filename;
    }
    return "Unknown Document";
}

// Function to clear the index for rebuilding
void clear_index()
{
    // Free any allocated memory for terms
    for (int i = 0; i < index_size; i++) {
        // In this implementation, we're using fixed-size arrays, so no need to free anything
    }
    
    // Reset all counters and data structures
    index_size = 0;
    memset(doc_lengths, 0, sizeof(doc_lengths));
    memset(documents, 0, sizeof(documents));
}

// Function to print the contents of the inverted index
void print_index()
{
    printf("Inverted Index Contents:\n");
    printf("Total Terms: %d\n", index_size);
    
    for (int i = 0; i < index_size; i++) {
        printf("Term: '%s' (%d docs)\n", index_data[i].term, index_data[i].posting_count);
        printf("  Postings: ");
        
        for (int j = 0; j < index_data[i].posting_count; j++) {
            printf("(doc:%d, freq:%d) ", 
                   index_data[i].postings[j].doc_id,
                   index_data[i].postings[j].freq);
            
            if (j > 5 && index_data[i].posting_count > 10) {
                printf("... and %d more", index_data[i].posting_count - j - 1);
                break;
            }
        }
        printf("\n");
        
        if (i > 30 && index_size > 50) {
            printf("... and %d more terms\n", index_size - i - 1);
            break;
        }
    }
}
