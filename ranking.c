#include "ranking.h"
#include "parser.h"
#include "utils.h"
#include "index.h"

// Forward declaration of get_doc_filename
extern const char* get_doc_filename(int doc_id);
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct
{
    int doc_id;
    double score;
} Result;

int cmp(const void *a, const void *b)
{
    Result *r1 = (Result *)a;
    Result *r2 = (Result *)b;
    return (r2->score > r1->score) - (r2->score < r1->score);
}

void rank_bm25(const char *query, int total_docs, int top_k)
{
    char query_copy[256];
    strcpy(query_copy, query);

    char *token = strtok(query_copy, " \t\n\r");
    Result results[1000] = {{0}};
    int result_count = 0;

    double avg_dl = 0;
    for (int i = 0; i < total_docs; ++i)
        avg_dl += get_doc_length(i);
    avg_dl /= total_docs;

    while (token)
    {
        to_lowercase(token);
        if (!is_stopword(token))
        {
            char *term = stem(token);
            for (int i = 0; i < index_size; ++i)
            {
                if (strcmp(index_data[i].term, term) == 0)
                {
                    int df = index_data[i].posting_count;
                    double idf = log((total_docs - df + 0.5) / (df + 0.5) + 1.0);
                    for (int j = 0; j < df; ++j)
                    {
                        int d = index_data[i].postings[j].doc_id;
                        int tf = index_data[i].postings[j].freq;
                        double dl = get_doc_length(d);
                        double score = idf * ((tf * (1.5 + 1)) / (tf + 1.5 * (1 - 0.75 + 0.75 * dl / avg_dl)));

                        results[d].doc_id = d;
                        results[d].score += score;
                        if (d + 1 > result_count)
                            result_count = d + 1;
                    }
                    break;
                }
            }
        }
        token = strtok(NULL, " \t\n\r");
    }

    qsort(results, result_count, sizeof(Result), cmp);
    for (int i = 0; i < top_k && i < result_count; ++i)
    {
        if (results[i].score > 0)
            printf("File: %s - Score: %.4f\n", get_doc_filename(results[i].doc_id), results[i].score);
    }
}
