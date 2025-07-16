# MPI-Based Parallel Search Engine

A comprehensive search engine implementation that uses the BM25 ranking algorithm to search through text documents, optimized for distributed computing environments using MPI (Message Passing Interface). It features both local document indexing and web crawling capabilities to build a versatile search corpus, with workload distributed across multiple processes and nodes.

## Features

- Parallel document parsing and indexing using MPI
- Distributed web crawling and content extraction
- Medium article crawling optimization
- Stopword removal for better search quality
- BM25 ranking algorithm for relevance-based results
- Command line interface with multiple operation modes
- URL normalization and handling
- Multi-node execution support for cluster environments
- Dynamic workload distribution across processes

## Project Structure

The project is organized using the following directory structure:

- `bin/`: Contains all executable files

  - `search_engine`: Main search engine executable
  - `evaluate`: Evaluation tool for search results
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

## Running the MPI Version

### Prerequisites

1. Install an MPI implementation (OpenMPI or MPICH recommended)

   ```bash
   # For Ubuntu/Debian
   sudo apt-get install openmpi-bin libopenmpi-dev

   # For CentOS/RHEL
   sudo yum install openmpi openmpi-devel

   # For macOS (using Homebrew)
   brew install open-mpi
   ```
2. Ensure proper network configuration for multi-node execution

   - All nodes must be able to communicate via SSH
   - Hostnames must be properly defined in `/etc/hosts` or DNS
   - A shared filesystem (NFS or similar) is recommended for dataset access

### How to run

```
mpirun -np 4 ./bin/search_engine -m @lpramithamj -d 2 -p 10
mpirun -np 4 ./bin/search_engine -c https://medium.com/@lpramithamj -d 2 -p 10
```
