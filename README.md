# Search Engine

A comprehensive search engine implementation that uses the BM25 ranking algorithm to search through text documents. It features both local document indexing and web crawling capabilities to build a versatile search corpus.

## Features

- Document parsing and indexing
- Web crawling and content extraction
- Medium article crawling optimization
- Stopword removal for better search quality
- BM25 ranking algorithm for relevance-based results
- Command line interface with multiple operation modes
- URL normalization and handling

## Required Files

- `dataset/`: Directory containing text documents to be indexed
- `stopwords.txt`: File containing stopwords, one per line

## Dependencies

- libcurl: Required for web crawling functionality

## Compilation

### Full Build (Including Tests)
```bash
make
```

### Production Build (Without Tests)
```bash
make production
```

### Cleaning Build Files
```bash
make clean
```

## Running

### Basic Operation (Local Dataset Only)
```bash
./search_engine
```

By default, the search engine will:
1. Load stopwords from `stopwords.txt`
2. Build an index from text documents in the `dataset/` directory
3. Prompt for a search query
4. Return top 10 documents matching the query

### Web URL Operations

#### Download and Index a Single URL
```bash
./search_engine -u https://example.com
```

#### Crawl and Index a Website (Following Links)
```bash
./search_engine -c https://example.com -d 2 -p 10
```

#### Crawl and Index a Medium Profile
```bash
./search_engine -m @username
```

### Command Line Options

- `-u URL`: Download and index content from a specific URL
- `-c URL`: Crawl a website starting from the specified URL (follows links)
- `-m USER`: Crawl a Medium profile for a specific user (e.g., `-m @username`)
- `-d NUM`: Set maximum crawl depth (default: 2, max: 5)
- `-p NUM`: Set maximum pages to crawl (default: 10, max: 100)
- `-h`: Display help information

## Adding New Content

### Local Documents
Add new text files to the `dataset/` directory. The search engine will automatically index them on the next run.

### Web Content
Use the web crawling features to automatically download and index content from the web:

```bash
# Download a single page
./search_engine -u https://example.com/article.html

# Crawl a website (depth=3, max pages=20)
./search_engine -c https://example.com -d 3 -p 20

# Crawl a Medium profile
./search_engine -m @username
```

All crawled content is saved to the `dataset/` directory and becomes part of your search corpus.

## Optimizing Search Quality

### Adding Stopwords

Edit the `stopwords.txt` file and add one stopword per line.

Example format:
```
the
a
an
in
of
```

### Crawl Parameters

- Adjust the depth parameter (`-d`) to control how deeply the crawler follows links
- Set the page limit (`-p`) to control the maximum number of pages to download
- For Medium profiles, the crawler automatically optimizes settings for better article extraction

## Examples

```bash
# Basic search using local documents
./search_engine

# Download and search Wikipedia content
./search_engine -u https://en.wikipedia.org/wiki/Information_retrieval

# Crawl a technical blog limited to 15 pages
./search_engine -c https://techblog.example.com -p 15

# Crawl a Medium profile with default settings
./search_engine -m @popular_writer
```
