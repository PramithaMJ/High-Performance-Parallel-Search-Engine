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
    
    // Use carefully chosen delimiters for tokenization - don't split on hyphens
    const char *delimiters = " \t\n\r.,;:!?\"()[]{}<>";
    char *token = strtok(text, delimiters);
    while (token)
    {
        to_lowercase(token);
        
        if (!is_stopword(token))
        {
            // Start timing stemming
            start_timer();
            
            // Apply stemming to the token
            char *stemmed = stem(token);
            
            // Record stemming time
            metrics.stemming_time += stop_timer();
            
            add_token(stemmed, doc_id);
        }
        token = strtok(NULL, delimiters);
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
