## Core Components

<img width="659" alt="Screenshot 2025-06-19 at 2 03 39 AM" src="https://github.com/user-attachments/assets/50e58496-3b4d-4c3d-b9a1-10e0acb9fade" />

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

## Parallelized Components

### 1. Index Building (index.c)

-  Building the index from multiple files

**Why parallelized:**

- File processing is an embarrassingly parallel task - each file can be processed independently
- Document parsing is I/O and CPU intensive
- No dependencies between documents during initial parsing

**Critical sections:**

- This section protects the shared `documents` array from race conditions when multiple threads update it concurrently

### 2. Term Indexing and Document Length Updates

-  Adding a token to the index (called during parsing)

**Why parallelized:**

- Uses OpenMP locks instead of critical sections for finer-grained control
- Two separate locks for different shared resources (index and document lengths)

**Critical points:**

- The inverted index structure is a shared resource - requires synchronization
- Document length tracking needs protection from concurrent updates

### 3. Web Crawling (crawler.c)

- URL extraction and processing

**Why parallelized:**

- HTML parsing is computationally expensive
- Extracting links from different sections of HTML can be done in parallel
- Link extraction doesn't modify the original HTML content

**Critical sections:**

- URL queue management needs protection to prevent corruption of the queue data structure

### 4. BM25 Ranking (ranking.c)

-  Score calculation for search results

**Why parallelized:**

- BM25 score calculation for each document is independent
- Query processing is often a bottleneck in search engines
- Score calculations involve floating-point operations that benefit from parallel execution

**Critical section:**

- Score accumulation in the results array needs protection to prevent race conditions

### 5. Term Search (index.c)

- Parallel search for terms in the index Process results...

**Why parallelized:**

- Term matching across a large index is computationally expensive
- First-match semantics are preserved with the critical section
- Multiple threads can search different parts of the index simultaneously

**Critical point:**

- Updating the `found` variable needs synchronization to ensure only the first match is recorded

### 6. Index Clearing and Initialization

- Clear the index for rebuilding

**Why parallelized:**

- Different memory areas can be initialized independently
- Memory operations benefit from parallel execution
- Parallel sections allow different threads to handle different initialization tasks

## Non-Parallelized Components and Rationale

### 1. Index Structure Management

The core index structure manipulation is protected by locks but not fully parallelized because:

- The inverted index has complex dependencies between terms
- The structure has inherent sequential nature for insertion and lookup
- It's more efficient to use fine-grained locks than to attempt full parallelization

### 2. Parser Initialization

This function is not parallelized internally because:

- File operations are primarily I/O bound
- The parallelism happens at a higher level (multiple files processed in parallel)
- The overhead of parallelizing small operations would outweigh benefits

### 3. URL Normalization

URL normalization remains sequential because:

- It's a relatively lightweight string manipulation operation
- Has internal dependencies within a single URL
- Parallelizing would introduce overhead for minimal benefit

## Critical Points in OpenMP Implementation

### 1. OpenMP Lock Initialization

These functions properly initialize and clean up OpenMP locks, which is critical for correct program behavior.

### 2. Thread Count Management

These settings are critical for:

- Allowing user control over parallelism
- Ensuring consistent thread allocation
- Supporting nested parallel regions

### 3. Load Balancing

The choice of dynamic scheduling is critical for:

- Balancing workloads across threads
- Adapting to varying processing times for different files/URLs
- Improving overall parallel efficiency

### 4. Thread-Safe Metrics Collection

This ensures that metrics collection doesn't interfere with the parallel execution and provides accurate performance data.

### 5. Data Structure Protection

Various OpenMP critical sections and locks protect shared data structures:

- Inverted index structure
- Document length arrays
- URL queue for crawling
- Search results accumulation

## Performance Considerations

1. **Granularity of Parallelism**:

   - File-level parallelism is coarse-grained (good for reducing overhead)
   - Term processing parallelism is medium-grained
   - HTML chunk processing is fine-grained with dynamic scheduling
2. **Synchronization Overhead**:

   - Locks and critical sections introduce overhead
   - The implementation balances parallelism with synchronization costs
   - Uses separate locks for different resources to reduce contention
3. **Load Balancing**:

   - Dynamic scheduling helps with varying document sizes
   - Thread status reporting shows distribution of work
   - Thread activation tracking prevents idle threads
4. **Memory Considerations**:

   - Each thread has private variables to avoid false sharing
   - Shared memory structures are protected with appropriate synchronization
   - Memory operations like memset are parallelized where beneficial

## Summary of Key Parallelization Strategies

1. **Document-Level Parallelism**: Multiple documents processed concurrently
2. **Token-Level Synchronization**: Fine-grained locks for index updates
3. **URL Processing Parallelism**: Concurrent extraction of links from HTML
4. **Query Processing Parallelism**: Parallel BM25 score calculation
5. **Critical Section Management**: Targeted protection of shared resources
6. **Load Balancing**: Dynamic scheduling for varying workloads

The inverted index maps terms to documents containing those terms, with frequency information. For each term, there's an `IndexEntry` containing:

- The term itself
- A list of `Posting` structures (documents containing that term)
- Each posting includes the document ID and the term's frequency in that document

## Serial vs. Parallel Indexing Process

### Serial Version

In the serial version, indexing happens sequentially:

1. Process one document at a time
2. For each document, extract tokens one by one
3. For each token, update the inverted index

### Parallel Version

The parallel version uses OpenMP to process multiple documents concurrently:

## Key Aspects of Index Parallelization

### 1. Document-Level Parallelism

Multiple threads process different documents simultaneously:

- **Benefits**:

  - No dependencies between documents during parsing
  - Coarse-grained parallelism reduces synchronization overhead
  - Easily scalable with the number of documents
- **Dynamic Scheduling**: Used to balance load between threads when documents have different sizes and processing times

### 2. Index Update Synchronization

The critical challenge is that while document parsing can be parallelized, the inverted index is a shared data structure that requires synchronization:

**Lock-Based Synchronization**:

- OpenMP locks protect the inverted index during updates
- Different locks for different resources (index structure vs. document lengths)
- This prevents race conditions where multiple threads try to update the same term

### 3. Advanced Parallel Index Construction

For better scalability, the implementation could use a more sophisticated approach:

This approach:

1. Creates thread-local mini-indexes
2. Processes documents in parallel, updating local indexes without contention
3. Merges local indexes into the global index in a critical section
4. Reduces lock contention significantly

### 4. Document Length Tracking

Document length tracking is also parallelized but carefully synchronized:

Using a separate lock for document lengths allows updates to the document length array to happen concurrently with term indexing when possible.

### 5. Parallel Search in the Index

Once built, searching the index is also parallelized:

## Performance Considerations for Parallel Indexing

### 1. Lock Contention

The main bottleneck in parallel indexing is lock contention:

- Each token addition requires acquiring the index lock
- High-frequency terms will cause more contention
- Solutions include:
  - Thread-local indexes (as described above)
  - Sharding the index by term prefix for less contention
  - Batching updates to reduce lock acquisition frequency

### 2. Memory Allocation Overhead

Memory allocation inside critical sections can be expensive:

- The implementation expands the postings array and index array dynamically
- Potential solutions:
  - Pre-allocate larger chunks to reduce reallocation frequency
  - Use a custom memory pool allocator
  - Perform batch reallocations outside critical sections when possible

### 3. Load Balancing

Document processing times can vary significantly:

- Dynamic scheduling helps distribute work more evenly
- Monitoring thread utilization helps identify imbalances
- Chunking large documents could provide finer-grained load balancing

## Summary of Inverted Index Parallelization

1. **Multi-Level Parallelism**:

   - Document-level parallelism (multiple documents processed concurrently)
   - Term-level synchronization (locks protect the shared index)
2. **Synchronization Mechanisms**:

   - OpenMP locks for fine-grained control
   - Separate locks for different resources (index vs. doc lengths)
   - Critical sections to protect shared data structures
3. **Optimization Opportunities**:

   - Thread-local indexes to reduce contention
   - Batch updates to amortize synchronization cost
   - Custom memory management to reduce allocation overhead
4. **Challenges and Trade-offs**:

   - Balancing parallelism vs. synchronization overhead
   - Managing memory effectively in a multi-threaded environment
   - Ensuring correctness while maximizing parallelism

The parallel indexing approach significantly improves performance over the serial version, especially for large document collections, by distributing document processing across multiple threads while carefully managing shared data structures.

## Data Handling & Storage

### 1. Document Storage

**How it works:**

- Downloaded/crawled content is saved as text files in the `dataset` directory
- The `documents` array stores metadata about each document, with document ID as the array index
- For each document, the system stores:
  - Filename (with path)
  - Additional metadata could include title, URL, etc.

### 2. Document Lengths

**How it works:**

- Document lengths are tracked in a global array
- Each position corresponds to the number of terms in the document
- Used for BM25 ranking calculations
- Protected with an OpenMP lock in the parallel version

### 3. Inverted Index Structure

**How it works:**

- Main data structure is an array of `IndexEntry` structures
- Each `IndexEntry` contains:
  - A term/word
  - A dynamic array of `Posting` structures
  - Each posting contains a document ID and frequency count
- The index grows dynamically as new terms are encountered

## Tokenization Process

### 1. File Parsing

## Data Storage and Management

### 1. Document Storage System

The search engine stores documents in the `dataset` directory as text files. These files are created when:

- Content is downloaded from URLs via `download_url()`
- Websites are crawled using `crawl_website()`
- Medium profiles are crawled with special handling

Document metadata is stored in a global array:

### 2. Inverted Index Structure

The core data structure is an inverted index, implemented as:

This structure maps terms (words) to documents containing them, where:

- `term` is the normalized word
- `postings` is an array of document references
- Each posting contains a document ID and the term frequency

## Tokenization Process

### 1. Document Parsing

The process starts with parsing files into tokens

### 2. Tokenization

Tokenization breaks text into individual words:

The tokenization process includes:

1. Breaking text on delimiters (spaces, punctuation)
2. Converting tokens to lowercase
3. Filtering out stopwords (common words like "the", "and")
4. Stemming (reducing words to their base form)
5. Adding tokens to the inverted index

## Building the Index (Parallelized)

### 1. Main Index Building Function

### 2. Adding Tokens to the Index (Thread-Safe Implementation)

## Benchmark System

### 1. Performance Metrics Collection

The search engine tracks several performance metrics:

### 2. Timing Functions

### 3. Speedup Calculation

The benchmark system compares current performance with baseline metrics:

### 4. Performance Data Storage

Results are stored in CSV files:

## Parallelization Techniques

### 1. Document-Level Parallelism

Multiple documents are processed concurrently:

**Why parallel**: Each document can be independently parsed and tokenized

### 2. Thread-Safe Index Updates

Locks prevent race conditions:

**Critical sections**: When updating shared data structures like the index:

### 3. Parallel BM25 Calculation

Query processing is parallelized:

**Why parallel**: Score calculation for each document is independent

### 4. Thread Workload Distribution Metrics

The system tracks thread utilization:

## Performance Benchmark Scripts

The system includes shell scripts to run benchmarks:

## Summary of Parallelization Strategy

1. **Document-Level Parallelism**: Documents are processed concurrently by multiple threads

   - Each thread handles separate documents
   - Dynamic scheduling balances workload across threads
2. **Thread-Safe Data Structures**:

   - OpenMP locks protect shared resources (index, document lengths)
   - Fine-grained locks minimize contention
   - Critical sections protect against race conditions
3. **Parallel BM25 Scoring**:

   - Score calculation for each document is parallelized
   - Results are accumulated in a thread-safe manner
4. **Performance Tracking**:

   - Comprehensive metrics collection
   - Baseline comparison for speedup calculation
   - Thread workload distribution analysis

## Project Overview

This project implements a high-performance search engine with multiple parallelization strategies using OpenMP and MPI. The search engine features web crawling, document indexing, BM25 ranking, and a web-based dashboard for performance monitoring and comparison.

## Performance Highlights

- **Serial Version**: Average query time: 365ms
- **OpenMP Version**: Average query time: 124ms (2.9x speedup)
- **MPI Version**: Average query time: 78ms (4.7x speedup vs serial)

## Features

- Full-text search with BM25 ranking algorithm
- Web crawling capabilities (general websites and Medium profiles)
- Document parsing and indexing
- Parallel processing with OpenMP (shared memory) and MPI (distributed memory)
- Comprehensive performance metrics and visualization
- Interactive web dashboard for search, configuration, and performance analysis
- URL normalization to prevent duplicate content
- Stopword filtering and word stemming

## Architecture

The search engine is divided into several key components:

1. **Crawler**: Downloads and processes web content
2. **Parser**: Extracts and normalizes text from documents
3. **Indexer**: Creates an inverted index for fast searching
4. **Ranking**: Implements BM25 scoring algorithm
5. **Query Processor**: Handles search queries
6. **Metrics**: Collects and reports performance data
7. **Dashboard**: Web interface for controlling the system

## Parallelization Strategies

### OpenMP Implementation

The OpenMP version uses shared-memory parallelism to optimize:

1. **Document Processing**: Parallel file parsing and tokenization
2. **Index Updates**: Thread-safe token addition with fine-grained locks
3. **Web Crawling**: Parallel HTML parsing and link extraction
4. **Search Processing**: Parallel BM25 score calculation

### MPI Implementation

The MPI version extends parallelism across multiple nodes:

1. **Distributed Indexing**: Partitions documents across nodes
2. **Parallel Query Processing**: Distributes search queries
3. **Result Aggregation**: Combines partial results from all nodes

## Performance Comparison

![1750278442548](images/README/1750278442548.png)

### Query Processing Time (ms)


| Threads/Processes | Serial | OpenMP | MPI |
| ----------------- | ------ | ------ | --- |
| 1                 | 365    | 365    | 365 |
| 2                 | 365    | 210    | 195 |
| 4                 | 365    | 124    | 112 |
| 8                 | 365    | 95     | 78  |
| 16                | 365    | 82     | 68  |
| 32                | 365    | 78     | 63  |

### Memory Usage

The memory footprint varies by implementation:

- Serial: Smallest memory footprint
- OpenMP: Moderate increase due to thread data
- MPI: Highest memory usage due to data replication across processes

![1750278494364](images/README/1750278494364.png)

## Setup and Installation

### Prerequisites

- GCC compiler with OpenMP support
- Open MPI implementation
- libcurl for web crawling
- NodeJS for the dashboard

### Building the Project

```bash
# Clone the repository
git clone https://github.com/yourusername/High-Performance-Parallel-Search-Engine.git
cd High-Performance-Parallel-Search-Engine

# Build all versions
make clean
make all
```

This will create the following binaries:

- `bin/search_engine` (Serial version)
- `bin/search_engine_omp` (OpenMP version)
- `bin/search_engine_mpi` (MPI version)

### Running the Dashboard

```bash
python3 api.py
```

Access the dashboard at http://localhost:5001

## Usage

### Command Line Interface

```bash
# Serial version
./bin/search_engine [options]

# OpenMP version
./bin/search_engine_omp [options]

# MPI version
mpirun -np <num_processes> ./bin/search_engine_mpi [options]
```

### Options

- `-u URL`: Download and index content from a single URL
- `-c URL`: Crawl website starting from URL
- `-m USER`: Crawl Medium profile for a specific user
- `-d NUM`: Maximum crawl depth (default: 2)
- `-p NUM`: Maximum pages to crawl (default: 10)
- `-t NUM`: Number of threads to use (OpenMP version)
- `-b`: Benchmark mode

### Web Interface

The web dashboard provides a user-friendly interface for:

1. Searching the indexed content
2. Configuring search parameters
3. Comparing performance across versions
4. Visualizing performance metrics
5. Building and deploying different versions

![1750278543494](images/README/1750278543494.png)

## Configuration

Advanced settings can be configured through the web interface:

- BM25 parameters (k1, b)
- Crawl depth and page limits
- OpenMP thread count
- MPI process count
- Dataset path and metrics storage

![1750278555240](images/README/1750278555240.png)

## Performance Tuning

For optimal performance:

1. **OpenMP Version**:

   - Set thread count to match CPU core count
   - Use dynamic scheduling for load balancing
2. **MPI Version**:

   - Configure hostfile for multi-node deployment
   - Balance processes across available nodes

## Testing and Benchmarking

The project includes tools for automated testing and benchmarking:

```bash
# Run functional tests
./bin/test_url_normalization
./bin/test_medium_urls
./bin/evaluate

# Run performance benchmark
./scripts/performance_benchmark.sh
```

Benchmark results are saved in CSV format for further analysis.

![1750278617629](images/README/1750278617629.png)

## Result Accuracy

All three versions (Serial, OpenMP, MPI) produce identical search results, using the same BM25 ranking algorithm. The only difference is in performance, not in search quality.

![1750278568218](images/README/1750278568218.png)
