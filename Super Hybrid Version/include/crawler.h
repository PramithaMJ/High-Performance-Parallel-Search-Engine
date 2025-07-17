#ifndef CRAWLER_H
#define CRAWLER_H

#include <string.h>

/**
 * Function to download content from a URL and save it to the dataset directory
 * 
 * @param url The URL to download
 * @return The filename where content was saved, or NULL on failure
 */
char* download_url(const char* url);

/**
 * Function to perform recursive crawling from a starting URL
 * 
 * @param start_url The URL to start crawling from
 * @param maxDepth Controls how deep the crawler will go (1 = just the initial URL, 
 *                 2 = also crawl links from the initial URL, etc.)
 * @param maxPages Limits the total number of pages to crawl
 * @return The number of pages successfully crawled
 */
int crawl_website(const char* start_url, int maxDepth, int maxPages);

// Utility function to check if a string contains a certain substring (case-insensitive)
static inline int strcasestr_exists(const char* haystack, const char* needle) {
    return haystack && needle && strcasestr(haystack, needle) != NULL;
}

#endif // CRAWLER_H
