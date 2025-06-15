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

// Stub: you can implement full Porter stemmer later
char *stem(char *word)
{
    return word; // no-op for now
}
