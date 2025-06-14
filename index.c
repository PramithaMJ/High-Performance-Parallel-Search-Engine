#include "index.h"
#include "parser.h"
#include <dirent.h>
#include <stdio.h>
#include <string.h>

InvertedIndex index_data[10000];
int index_size = 0;
int doc_lengths[1000] = {0};

int build_index(const char *folder_path)
{
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
        if (entry->d_name[0] != '.' && strstr(entry->d_name, ".txt"))
        {
            char path[256];
            snprintf(path, sizeof(path), "%s/%s", folder_path, entry->d_name);
            printf("Processing file: %s\n", path);
            if (parse_file(path, doc_id))
            {
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
    return doc_id;
}

void add_token(const char *token, int doc_id)
{
    // Skip empty tokens or tokens that are too long
    if (!token || strlen(token) == 0 || strlen(token) > 100) {
        return;
    }

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
