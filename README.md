# Search Engine

A simple search engine implementation that uses BM25 ranking algorithm to search through text documents.

## Features

- Document parsing and indexing
- Stopword removal
- BM25 ranking algorithm
- Command line interface

## Required Files

- `dataset/`: Directory containing text documents to be indexed
- `stopwords.txt`: File containing stopwords, one per line

## Compilation

```bash
make
```

## Running

```bash
./search_engine
```

By default, the search engine will:
1. Load stopwords from `stopwords.txt`
2. Build an index from text documents in the `dataset/` directory
3. Prompt for a search query
4. Return top 10 documents matching the query

## Adding new documents

Simply add new text files to the `dataset/` directory. The search engine will automatically index them on next run.

## Adding stopwords

Edit the `stopwords.txt` file and add one stopword per line.

Example format:
```
the
a
an
in
of
```
