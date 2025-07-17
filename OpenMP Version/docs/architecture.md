# Search Engine Documentation

## Architecture Overview

The search engine is built with several key components:

### Core Components

1. **Parser** (`parser.c`/`parser.h`)
   - Handles document parsing, tokenization and preprocessing
   - Extracts text content from HTML and plain text files

2. **Index** (`index.c`/`index.h`)
   - Inverted index implementation
   - Term frequency and document frequency calculations
   - Document storage and retrieval

3. **Ranking** (`ranking.c`/`ranking.h`)
   - BM25 algorithm implementation
   - Score calculation and result sorting
   - Relevance determination

4. **Crawler** (`crawler.c`/`crawler.h`)
   - Web page crawling and content extraction
   - URL normalization and queue management
   - Robots.txt compliance

5. **Utilities** (`utils.c`/`utils.h`)
   - Common helper functions
   - String manipulation, data structures
   - File I/O operations

6. **Metrics** (`metrics.c`/`metrics.h`)
   - Performance monitoring
   - Search quality evaluation

### Application Flow

1. Documents are parsed and indexed either from local files or web crawling
2. User provides search query through command line
3. Query is tokenized and processed (stopwords removed)
4. Inverted index is searched for matching documents
5. BM25 algorithm ranks matching documents
6. Sorted results are returned to the user

## Configuration

The `config.ini` file contains all configurable parameters including:

- Indexing limits
- Crawler settings
- Ranking parameters
- File paths
- Performance settings

## Usage Examples

### Basic Search
```
./bin/search_engine -q "search query terms"
```

### Index Local Files
```
./bin/search_engine -i ./dataset
```

### Crawl and Index Web Content
```
./bin/search_engine -c "https://example.com" -d 2
```

### Evaluate Search Performance
```
./bin/evaluate -q "test query" -r reference_results.txt
```
