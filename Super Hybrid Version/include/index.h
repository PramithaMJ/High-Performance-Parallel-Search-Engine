#ifndef INDEX_H
#define INDEX_H

#define MAX_FILENAME_LEN 256

// Document structure to store file names
typedef struct {
    char filename[MAX_FILENAME_LEN];
} Document;

int build_index(const char *folder_path);
void add_token(const char *token, int doc_id);
int get_doc_length(int doc_id);
int get_doc_count();
const char* get_doc_filename(int doc_id);
void print_index();
void clear_index();
void synchronize_document_filenames();
void print_all_index_terms();

typedef struct
{
    int doc_id;
    int freq;
} Posting;

typedef struct
{
    char term[64];
    Posting postings[1000];
    int posting_count;
} InvertedIndex;

extern InvertedIndex index_data[10000];
extern int index_size;

#endif
