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

### Running on a Single Node

The `run_mpi.sh` script provides an easy way to run the search engine with multiple MPI processes on a single node:

```bash
# Run with default number of processes (4)
./scripts/run_mpi.sh

# Run with a specific number of processes (e.g., 8)
./scripts/run_mpi.sh 8

# Run with specific arguments (e.g., crawling mode)
./scripts/run_mpi.sh 4 -m @lpramithamj
```

### Running on Multiple Nodes

For multi-node execution, we provide the `run_multi_node.sh` script and a `hostfile` for node configuration:

1. Edit the `hostfile` to specify your node configuration:
   ```
   node1 slots=4
   node2 slots=4
   node3 slots=4
   ```

2. Run the search engine across multiple nodes:
   ```bash
   # Run with default configuration (3 nodes, 4 processes per node)
   ./scripts/run_multi_node.sh
   
   # Run with specific configuration (2 nodes, 6 processes per node)
   ./scripts/run_multi_node.sh 2 6
   
   # Run with specific arguments
   ./scripts/run_multi_node.sh 3 4 -m @lpramithamj
   ```

### Running on a Cluster with PBS

For PBS-based clusters, we provide a PBS job script:

1. Edit `scripts/pbs_job.sh` to match your cluster configuration
2. Submit the job:
   ```bash
   qsub scripts/pbs_job.sh
   ```

## Performance Considerations

The MPI implementation offers several performance advantages over the OpenMP version:

1. **Distributed Memory**: Can utilize resources across multiple physical nodes
2. **Scalability**: Performance scales with additional nodes, not just cores
3. **Dynamic Load Balancing**: Work is distributed to minimize idle time
4. **Reduced Memory Contention**: Each process has its own memory space

For best performance:
1. Match the number of MPI processes to available CPU cores
2. Consider network speed when distributing work across nodes
3. Use a balanced dataset to prevent bottlenecks
4. Adjust the number of processes based on the task (indexing vs. crawling)

## MPI Implementation Architecture

This MPI-based search engine distributes work across processes using the following approach:

1. **Master-Worker Model**:
   - Rank 0 (Master) coordinates work distribution and result collection
   - Other ranks (Workers) process assigned data chunks

2. **Data Distribution**:
   - Documents are distributed evenly across processes
   - Each process handles a subset of the document corpus
   - Dynamic load balancing adjusts workload based on process capabilities

3. **Communication Pattern**:
   - Point-to-point messaging for data distribution and collection
   - Collective operations for global coordination and aggregation
   - Barriers for synchronization at critical phases

4. **Index Merging**:
   - Local indexes created on each process
   - Global index assembled through collective operations
   - Optimized to minimize communication overhead

5. **Fault Tolerance**:
   - Basic error detection and reporting
   - Graceful handling of process failures

## Usage Examples

### Example 1: Indexing a Document Collection

```bash
# Using 8 MPI processes to index documents in the dataset directory
mpirun -np 8 ./bin/search_engine -i dataset

# Alternative using the convenience script
./scripts/run_mpi.sh 8 -i dataset
```

### Example 2: Running a Search Query

```bash
# Search for "artificial intelligence" using 4 MPI processes
mpirun -np 4 ./bin/search_engine -q "artificial intelligence"
```

### Example 3: Web Crawling with Multiple Nodes

```bash
# Crawl from a Medium profile using 3 nodes with 4 processes each
./scripts/run_multi_node.sh 3 4 -m @lpramithamj
```

### Example 4: Advanced Usage with PBS Cluster

1. Edit the PBS script with your parameters:
```bash
#!/bin/bash
#PBS -N search_engine_job
#PBS -l nodes=4:ppn=8
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o search_engine_job.log

cd $PBS_O_WORKDIR
module load openmpi

# Index a large corpus
mpirun -np $PBS_NP ./bin/search_engine -i /shared/large_corpus
```

2. Submit the job:
```bash
qsub scripts/pbs_job.sh
```

## Dependencies

- MPI implementation (OpenMPI or MPICH recommended)
- libcurl (for web crawling)
