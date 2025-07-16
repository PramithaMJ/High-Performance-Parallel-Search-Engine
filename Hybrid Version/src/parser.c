#include "../include/parser.h"
#include "../include/utils.h"
#include "../include/index.h"
#include "../include/metrics.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

int parse_file(const char *filepath, int doc_id)
{
    // Start timing for parsing
    start_timer();
    
    FILE *fp = fopen(filepath, "r");
    if (!fp) {
        printf("Could not open file: %s\n", filepath);
        return 0;
    }

    fseek(fp, 0, SEEK_END);
    long len = ftell(fp);
    rewind(fp);
    
    // Guard against very large files
    if (len > 10000000) { // 10MB limit
        printf("File too large: %s (%ld bytes)\n", filepath, len);
        fclose(fp);
        return 0;
    }

    char *content = malloc(len + 1);
    if (!content) {
        printf("Memory allocation failed for file: %s\n", filepath);
        fclose(fp);
        return 0;
    }
    
    size_t read_bytes = fread(content, 1, len, fp);
    content[read_bytes] = '\0';
    fclose(fp);

    tokenize(content, doc_id);
    free(content);
    
    // Record parsing time
    metrics.parsing_time += stop_timer();
    return 1;
}

void tokenize(char *text, int doc_id)
{
    // Start timing for tokenization
    start_timer();
    
    // Debug counters
    int token_count = 0;
    int added_count = 0;
    
    char *token = strtok(text, " \t\n\r.,;:!?\"()[]{}<>");
    while (token)
    {
        token_count++;
        to_lowercase(token);
        
        // Special debug for microservice term
        if (strstr(token, "microservice") != NULL) {
            printf("DEBUG: Found 'microservice' in token: '%s'\n", token);
        }
        
        if (!is_stopword(token))
        {
            // Start timing stemming
            start_timer();
            char *stemmed = stem(token);
            // Record stemming time
            metrics.stemming_time += stop_timer();
            
            // Debug output for important terms
            if (strlen(token) > 5) {
                printf("DEBUG: Token '%s' stemmed to '%s'\n", token, stemmed);
            }
            
            add_token(stemmed, doc_id);
            added_count++;
        }
        token = strtok(NULL, " \t\n\r.,;:!?\"()[]{}<>");
    }
    
    printf("DEBUG: Tokenized %d tokens, added %d terms for doc_id %d\n", 
           token_count, added_count, doc_id);
}
    
    // Record tokenizing time
    metrics.tokenizing_time += stop_timer();
}

void to_lowercase(char *str)
{
    for (; *str; ++str)
        *str = tolower(*str);
}

// Parallel version of parse_file function
int parse_file_parallel(const char *filepath, int doc_id)
{
    // Start timing for parsing
    start_timer();
    
    FILE *fp = fopen(filepath, "r");
    if (!fp) {
        printf("Could not open file: %s\n", filepath);
        return 0;
    }

    fseek(fp, 0, SEEK_END);
    long len = ftell(fp);
    rewind(fp);
    
    // Guard against very large files
    if (len > 10000000) { // 10MB limit
        printf("File too large: %s (%ld bytes)\n", filepath, len);
        fclose(fp);
        return 0;
    }

    char *content = malloc(len + 1);
    if (!content) {
        printf("Memory allocation failed for file: %s\n", filepath);
        fclose(fp);
        return 0;
    }
    
    size_t read_bytes = fread(content, 1, len, fp);
    content[read_bytes] = '\0';
    fclose(fp);

    // Use tokenize function (which already has OpenMP parallelism)
    tokenize(content, doc_id);
    free(content);
    
    // Record parsing time
    metrics.parsing_time += stop_timer();
    return 1;
}
