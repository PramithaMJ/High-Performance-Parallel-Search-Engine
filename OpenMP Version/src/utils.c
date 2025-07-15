#include "../include/utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *stopword_list[1000];
int stopword_count = 0;

int is_stopword(const char *word)
{
    if (stopword_count == 0)
    {
        FILE *fp = fopen("stopwords.txt", "r");
        if (!fp)
            return 0;
        char buf[64];
        while (fgets(buf, sizeof(buf), fp))
        {
            buf[strcspn(buf, "\n")] = 0;
            stopword_list[stopword_count++] = strdup(buf);
        }
        fclose(fp);
    }

    for (int i = 0; i < stopword_count; ++i)
    {
        if (strcmp(word, stopword_list[i]) == 0)
            return 1;
    }
    return 0;
}

// Clean up allocated memory for stopwords
void cleanup_stopwords() {
    for (int i = 0; i < stopword_count; ++i) {
        if (stopword_list[i]) {
            free(stopword_list[i]);
            stopword_list[i] = NULL;
        }
    }
    stopword_count = 0;
}

// Enhanced stemmer implementation for simplified English stemming with compound word handling
char *stem(char *word)
{
    // Safety check for NULL
    if (!word) {
        return word;
    }
    
    int len = strlen(word);
    
    // Skip very short words (minimum 3 characters)
    if (len <= 2) {
        return word;
    }
    
    // Using thread-local storage to ensure thread safety in parallel processing
    static __thread char stemmed_word[256]; // Thread-local buffer to hold stemmed result
    memset(stemmed_word, 0, sizeof(stemmed_word));
    
    // Make a safe copy of the word
    if (len >= sizeof(stemmed_word)) {
        len = sizeof(stemmed_word) - 1;  // Truncate if too long
    }
    // Use memcpy instead of strncpy to avoid the warning
    memcpy(stemmed_word, word, len);
    stemmed_word[len] = '\0';
    
    // Special case handling for technical terms
    // We use a direct comparison approach for common technical terms that
    // may not follow standard English singular/plural rules
    
    // Microservices special case
    if (strcmp(stemmed_word, "microservice") == 0 || strcmp(stemmed_word, "microservices") == 0) {
        strcpy(stemmed_word, "microservice");
        return stemmed_word;
    }
    
    // API special case
    if (strcmp(stemmed_word, "api") == 0 || strcmp(stemmed_word, "apis") == 0) {
        strcpy(stemmed_word, "api");
        return stemmed_word;
    }
    
    // Database special case
    if (strcmp(stemmed_word, "database") == 0 || strcmp(stemmed_word, "databases") == 0) {
        strcpy(stemmed_word, "database");
        return stemmed_word;
    }
    
    // Index special case
    if (strcmp(stemmed_word, "index") == 0 || strcmp(stemmed_word, "indices") == 0 || 
        strcmp(stemmed_word, "indexes") == 0) {
        strcpy(stemmed_word, "index");
        return stemmed_word;
    }
    
    // Cloud special case
    if (strcmp(stemmed_word, "cloud") == 0 || strcmp(stemmed_word, "clouds") == 0) {
        strcpy(stemmed_word, "cloud");
        return stemmed_word;
    }
    
    // Container special case
    if (strcmp(stemmed_word, "container") == 0 || strcmp(stemmed_word, "containers") == 0) {
        strcpy(stemmed_word, "container");
        return stemmed_word;
    }
    
    // Additional special cases can be added here as needed
    
    // Handle basic plural forms - "s" and "es" endings
    if (len > 2 && stemmed_word[len-1] == 's') {
        // Handle -ies to -y conversion (e.g. "stories" -> "story")
        if (len > 3 && stemmed_word[len-3] == 'i' && stemmed_word[len-2] == 'e') {
            stemmed_word[len-3] = 'y';
            stemmed_word[len-2] = '\0';
        }
        // Handle -es endings
        else if (len > 2 && stemmed_word[len-2] == 'e') {
            // Special cases for words ending with -es where we keep the 'e'
            if (len > 3 && (stemmed_word[len-3] == 's' || stemmed_word[len-3] == 'x' || 
                          stemmed_word[len-3] == 'z' || 
                          (len > 4 && stemmed_word[len-4] == 'c' && stemmed_word[len-3] == 'h') ||
                          (len > 4 && stemmed_word[len-4] == 's' && stemmed_word[len-3] == 'h'))) {
                stemmed_word[len-1] = '\0'; // Remove just the 's'
            } else {
                // Regular -es endings (remove both 'e' and 's')
                stemmed_word[len-2] = '\0';
            }
        }
        // Regular plural (just remove 's')
        else {
            stemmed_word[len-1] = '\0';
        }
    }
    
    return stemmed_word;
}
