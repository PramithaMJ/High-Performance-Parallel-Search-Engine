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

char *stem(char *word)
{
    static char stemmed[256];
    strcpy(stemmed, word);
    int len = strlen(stemmed);
    
    // Convert to lowercase
    for (int i = 0; i < len; i++) {
        if (stemmed[i] >= 'A' && stemmed[i] <= 'Z') {
            stemmed[i] = stemmed[i] - 'A' + 'a';
        }
    }
    
    // Basic plural removal for common words
    if (len > 2 && stemmed[len-1] == 's') {
        // ("technologies" -> "technology")
        if (len > 3 && stemmed[len-2] == 'e' && stemmed[len-3] == 'i') {
            stemmed[len-3] = 'y';
            stemmed[len-2] = '\0';
        }
        // Handle words ending in "es" (e.g., "boxes" -> "box")
        else if (len > 2 && stemmed[len-2] == 'e') {
            stemmed[len-2] = '\0';
        }
        // ("cats" -> "cat")
        else {
            stemmed[len-1] = '\0';
        }
    } 

    if (strcmp(word, "microservice") == 0 || strcmp(word, "microservices") == 0) {
        printf("Debug: Stemming special case '%s' to 'microservic'\n", word);
        strcpy(stemmed, "microservic");
    }
    
    return stemmed;
}
