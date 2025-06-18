## Core Components in Serial Version

### 1. Main (main.c)

- The entry point that handles command-line arguments and coordinates program flow
- Supports multiple operation modes:
  - `-u URL`: Download and index content from a single URL
  - `-c URL`: Crawl website starting from URL (follows links)
  - `-m USER`: Crawl Medium profile for a specific user
  - `-d NUM`: Maximum crawl depth (default: 2)
  - `-p NUM`: Maximum pages to crawl (default: 10)
- Once content is downloaded/crawled, it builds an index and accepts search queries

### 2. Crawler (crawler.c)

- Handles website crawling and content extraction using libcurl
- Uses breadth-first search to navigate web pages up to a specified depth
- Key functionalities:
  - `crawl_website()`: Main crawling function that manages the BFS queue
  - `download_url()`: Downloads content from a URL
  - `normalize_url()`: Cleans URLs to avoid duplicates
  - `extract_links()`: Parses HTML to find linked pages
  - HTML content extraction with special handling for Medium.com articles

### 3. Parser (parser.c)

- Processes downloaded documents and prepares them for indexing
- Functions include:
  - `parse_file()`: Reads file content and initiates tokenization
  - `tokenize()`: Splits text into individual words/tokens
  - `to_lowercase()`: Converts tokens to lowercase for case-insensitive search
  - Removes stopwords using the `is_stopword()` function
  - Performs stemming with the `stem()` function to normalize word variations

### 4. Index (index.c)

- Manages the inverted index data structure
- Key components:
  - `InvertedIndex` struct: Stores terms and their document postings
  - `Document` struct: Stores document information
  - `build_index()`: Processes all files in the dataset directory
  - `add_token()`: Adds a token to the inverted index
  - `get_doc_length()`, `get_doc_count()`, `get_doc_filename()`: Utility functions

### 5. Ranking (ranking.c)

- Implements the BM25 algorithm for relevance-based ranking
- Key components:
  - `rank_bm25()`: Performs query processing using BM25 formula
  - BM25 parameters (k1=1.5, b=0.75) for term frequency normalization
  - Uses an inverted index to efficiently find matching documents

### 6. Metrics (metrics.c)

- Collects and reports performance metrics:
  - Execution time (crawling, parsing, tokenizing, indexing, etc.)
  - Memory usage
  - Document statistics
  - Query latency
- Provides functions to save metrics to CSV files for benchmarking

### 7. Benchmark & Evaluation (benchmark.c, evaluate.c)

- evaluate.c: Tests search quality with sample queries
- benchmark.c: Tools for performance measurement and comparison

## Data Flow & Processing Logic

1. **Web Crawling Process**:

   - Starts with a seed URL and performs breadth-first traversal
   - For each URL:
     - Checks if already visited using `has_visited()`
     - Downloads content with libcurl
     - Extracts clean text from HTML
     - Extracts links for further crawling
     - Saves content to the dataset directory with meaningful filenames
2. **Indexing Process**:

   - Reads all files in the dataset directory
   - For each file:
     - Parses content into tokens
     - Removes stopwords
     - Applies stemming
     - Adds tokens to inverted index with document frequencies
3. **Search Process**:

   - Tokenizes the search query
   - Removes stopwords and applies stemming
   - For each token:
     - Finds matching documents in the inverted index
     - Calculates BM25 scores based on term frequency and document length
   - Sorts documents by their accumulated scores
   - Returns the top-k results
4. **BM25 Ranking**:

   - Score = IDF × ((tf × (k1 + 1)) / (tf + k1 × (1 - b + b × dl / avgdl)))
   - Where:
     - IDF: Inverse document frequency
     - tf: Term frequency in document
     - dl: Document length
     - avgdl: Average document length
     - k1=1.5, b=0.75: Tuning parameters

## Special Features

1. **Medium.com Optimization**:

   - Custom URL handling for Medium profiles
   - Better article extraction with special handling for Medium markup
   - Rate-limiting between requests to prevent being blocked
2. **URL Normalization**:

   - Removes tracking parameters (utm_, fbclid, etc.)
   - Handles relative URLs properly
   - Prevents duplicate crawling of the same content
3. **HTML Content Extraction**:

   - Removes HTML tags, scripts, styles
   - Preserves meaningful headings and paragraphs
   - Converts HTML entities to proper characters
   - Outputs in a clean text format suitable for searching

## Performance Considerations

The serial implementation has several performance bottlenecks:

1. Sequential web crawling - can't download pages in parallel
2. Sequential document parsing and indexing
3. No parallel query processing
4. In-memory index with size limitations

These limitations make the serial version suitable for small to medium-sized document collections but less efficient for large-scale crawling and indexing tasks.

The metrics collection helps identify these bottlenecks, providing a baseline for comparing with a parallel implementation.
