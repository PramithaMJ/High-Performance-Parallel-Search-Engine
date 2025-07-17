#ifndef PARSER_H
#define PARSER_H

int parse_file(const char *filepath, int doc_id);
int parse_file_parallel(const char *filepath, int doc_id); // Added parallelized version
void tokenize(char *text, int doc_id);
void to_lowercase(char *str);

#endif
