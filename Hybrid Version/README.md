# Hybrid MPI+OpenMP Parallel Search Engine

A high-performance search engine implementation that combines MPI (Message Passing Interface) for distributed memory parallelism with OpenMP for shared memory parallelism. This hybrid approach leverages the benefits of both paradigms to achieve maximum performance across compute clusters and multi-core systems.

## Features

- **Hybrid Parallelism**: Combines MPI for distributed processing across nodes with OpenMP for multi-threading within each node
- **Configurable Parallelism**: Use `-np` to set MPI process count and `-t` to set OpenMP thread count
- Document parsing and indexing with parallel processing
- Web crawling and content extraction
- Medium article crawling optimization
- Stopword removal for better search quality
- BM25 ranking algorithm with distributed computation for faster relevance-based results
- Load balancing strategies to optimize workload distribution
- Performance benchmarking tools to find optimal MPI/OpenMP configurations for your hardware

## Hybrid Implementation Architecture

The search engine uses a two-level parallelism approach:

1. **Level 1: MPI (Distributed Memory)**

   - Distributes document collections across multiple nodes/processes
   - Each MPI process works on its own subset of documents
   - Uses custom MPI datatypes for efficient communication of search results
   - Performs collective operations (e.g., `MPI_Allreduce`) for global statistics like average document length
2. **Level 2: OpenMP (Shared Memory)**

   - Each MPI process creates multiple OpenMP threads
   - Threads work on different query terms in parallel within each process's document subset
   - Uses thread-local storage to minimize critical sections and lock contention
   - Synchronizes at key points with reduction operations for score calculation

### Implementation Details

- **Document Distribution**: Documents are evenly divided among MPI processes with a ceiling division algorithm for load balancing
- **Query Processing**: Each query term is processed in parallel across threads within each process
- **Result Aggregation**: Local top-k results from each MPI process are gathered and merged for final ranking
- **Performance Monitoring**: Built-in timing metrics track query latency and throughput

## Performance Benefits

The hybrid MPI+OpenMP implementation provides several performance advantages:

1. **Scalability across nodes**: Using MPI allows the search engine to distribute work across multiple compute nodes in a cluster
2. **Efficient memory utilization**: OpenMP enables thread-level parallelism with shared memory, reducing memory overhead
3. **Load balancing**: Dynamic workload distribution among threads and processes
4. **Reduced communication overhead**: Minimizing inter-node communication by using shared memory within nodes
5. **Resource optimization**: Ability to fine-tune the balance between processes and threads based on available hardware

## Usage

### Compilation

```bash
# Clean and build the search engine
make clean
make
```

### Running the Search Engine

You can run the search engine with different configurations for MPI processes and OpenMP threads:

#### Using the Wrapper Script (Recommended)

The wrapper script offers both positional parameters and flag-style options:

```bash
# Positional parameters: <MPI_PROCESSES> <OMP_THREADS> [options]
./run_search 4 8 -q "artificial intelligence"

# Flag-style options
./run_search -np 2 -t 4 -q "machine learning"
```

#### Direct MPI Execution

You can also use `mpirun` directly:

```bash
mpirun -np 4 ./bin/search_engine -t 8 -q "deep learning"
```

#### Using Make Targets

```bash
# Run with default configuration (4 MPI processes, 4 OpenMP threads)
make run

# Run with custom configuration
make run_custom MPI_PROCS=2 OMP_THREADS=8
```

### Command Line Options

- `-np <NUM>` - Set number of MPI processes
- `-t <NUM>` - Set number of OpenMP threads per process
- `-q <QUERY>` - Run search with the specified query
- `-c <URL>` - Crawl website starting from URL
- `-m <USER>` - Crawl Medium profile (e.g., @username)
- `-d <NUM>` - Set maximum crawl depth (default: 2)
- `-p <NUM>` - Set maximum pages to crawl (default: 10)
- `-i` - Print OpenMP information
- `-h` - Show help message

### Performance Benchmarking

To find the optimal configuration for your hardware:

```bash
./scripts/performance_benchmark.sh
```

This script tests various combinations of MPI processes and OpenMP threads and generates a detailed performance report with timing metrics.

To visualize the performance results:

```bash
./scripts/visualize_performance.py
```

This will generate graphs showing the performance characteristics of different MPI and OpenMP configurations, including:

- 3D bar chart of processing time by configuration
- Line chart of performance scaling with total core count
- Heatmap visualization of optimal configurations

## Project Structure

The project is organized using the following directory structure:

- `bin/`: Contains all executable files
  - `search_engine`: Main search engine executable
  - `evaluate`: Evaluation tool for search results
  - `test_url_normalization`: Test for URL normalization
  - `test_medium_urls`: Test for Medium URL handling

## Implementation Details

The hybrid parallelization approach distributes work as follows:

### MPI (Inter-node parallelism)

- Document corpus is partitioned across MPI processes
- Each MPI process handles a subset of documents for indexing and searching
- Query processing is performed in parallel across all MPI processes
- Results are gathered and merged by the root process

### OpenMP (Intra-node parallelism)

- Document parsing and tokenization uses thread-level parallelism
- Average document length calculation is parallelized with OpenMP
- Term scoring is performed in parallel by multiple threads within each MPI process
- Thread-local result arrays minimize locking overhead

### Hybrid Optimizations

- Custom MPI datatype for efficient result gathering
- Reduction operations for aggregating partial results
- Thread-local processing to minimize critical sections
- Efficient memory management to reduce data duplication

### Performance Considerations

- Thread affinity settings can be tuned for optimal performance
- The ratio of MPI processes to OpenMP threads should be adjusted based on network latency and node configuration
- For systems with high core counts but slow inter-node communication, fewer MPI processes with more OpenMP threads may be optimal
- For systems with fast interconnects, more MPI processes may provide better performance

  - `test_url_normalization`: Test for URL normalization
  - `test_medium_urls`: Test for Medium URL handling
- `src/`: Contains all source code (.c files)

  - Core components: main, parser, index, ranking, crawler, etc.
- `include/`: Contains all header files (.h files)

  - Declarations and interfaces for all components
- `obj/`: Contains object files (.o files) generated during compilation
- `data/`: Contains text data files used by the application

  - `stopwords.txt`: List of words to ignore during indexing and searching
  - `medium_url_fixes.txt`: Special handling rules for Medium URLs
  - `serial_metrics.csv`: Performance metrics data
- `dataset/`: Contains the document corpus to be indexed
- `tests/`: Contains test source files

  - Test implementations for various components
- `scripts/`: Contains utility shell scripts

  - `benchmark.sh`: Script for performance benchmarking
  - `run_benchmark.sh`: Script to execute benchmarks

## Required Files

- `dataset/`: Directory containing text documents to be indexed
- `data/stopwords.txt`: File containing stopwords, one per line

## Building the Project

To build the entire project including tests:

```
make all
```

For a production build (without tests):

```
make production
```

To clean the build:

```
make clean
```

## Dependencies

- libcurl (for web crawling)

## How Run

```bash
mpirun --oversubscribe -np 10 ./bin/search_engine -np 10 -t 8 -c https://medium.com/@lpramithamj -d 2 -p 10

./run_search -np 8 -t 8 -c https://medium.com/@lpramithamj -d 2 -p 10
```
