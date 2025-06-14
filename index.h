#ifndef INDEX_H
#define INDEX_H

int build_index(const char *folder_path);
void add_token(const char *token, int doc_id);
int get_doc_length(int doc_id);
int get_doc_count();
void print_index();

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
