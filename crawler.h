#ifndef CRAWLER_H
#define CRAWLER_H

// Function to download content from a URL and save it to the dataset directory
// Returns the filename where content was saved, or NULL on failure
char* download_url(const char* url);

#endif // CRAWLER_H
