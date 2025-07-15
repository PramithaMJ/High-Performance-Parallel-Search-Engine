# Running High Performance Parallel Search Engine Versions

This guide provides instructions on how to build and run each version of the search engine, along with an overview of the key parallelized functions.

## Table of Contents
- [Build Instructions](#build-instructions)
- [Running Each Version](#running-each-version)
  - [Serial Version](#serial-version)
  - [OpenMP Version](#openmp-version)
  - [MPI Version](#mpi-version)
  - [Hybrid Version](#hybrid-version)
- [Web Interface](#web-interface)
- [Key Parallelized Functions](#key-parallelized-functions)
  - [Document-Level Parallelism](#document-level-parallelism)
  - [Index Building Parallelization](#index-building-parallelization)
  - [Web Crawling Parallelization](#web-crawling-parallelization)
  - [Search Query Parallelization](#search-query-parallelization)
- [Command-line Options for All Versions](#command-line-options-for-all-versions)
- [Benchmark & Performance Comparison](#benchmark--performance-comparison)

## Build Instructions

All versions can be built from their respective directories using the `make` command:

```bash
# From the project root directory
cd Serial\ Version/
make clean && make

cd ../OpenMP\ Version/
make clean && make

cd ../MPI\ Version/
make clean && make

cd ../Hybrid\ Version/
make clean && make
```

## Running Each Version

### Serial Version

The serial version operates on a single thread with no parallelization:

```bash
cd Serial\ Version/
./bin/search_engine -u https://example.com
```

### OpenMP Version

The OpenMP version utilizes shared memory parallelism across multiple threads:

```bash
cd OpenMP\ Version/
./bin/search_engine_omp -u https://example.com -t 8  # Use 8 threads
```

### MPI Version

The MPI version distributes work across multiple processes (potentially on different machines):

```bash
cd MPI\ Version/
mpirun -np 8 ./bin/search_engine_mpi -u https://example.com  # Use 8 processes
```

For running across multiple machines:

```bash
cd MPI\ Version/
mpirun -np 16 -hostfile hostfile ./bin/search_engine_mpi -u https://example.com
```

### Hybrid Version

The Hybrid version combines both OpenMP and MPI for multi-level parallelism:

```bash
cd Hybrid\ Version/
mpirun -np 4 ./bin/search_engine -u https://example.com -t 4  # 4 processes, each with 4 threads
```

## Web Interface

You can also use the web dashboard to run and compare all versions:

```bash
cd WebSite/
./start_dashboard.sh
```

Then open your browser and navigate to: http://localhost:5001

## Key Parallelized Functions

### Document-Level Parallelism

Both OpenMP and MPI versions parallelize document processing at different levels:

- **OpenMP Version**: Uses thread-level parallelism to process multiple documents concurrently
- **MPI Version**: Distributes document collections across multiple processes
- **Hybrid Version**: Combines both approaches

### Index Building Parallelization

The index building process is parallelized in different ways:

- **OpenMP**: Multiple threads process documents and update a shared index with synchronized access
- **MPI**: Each process builds a local index, then merges results
- **Hybrid**: Uses MPI for document distribution and OpenMP for local document processing

### Web Crawling Parallelization

Web crawling operations are parallelized to improve performance:

- **OpenMP**: Parallel HTML parsing and concurrent link extraction
- **MPI**: Distributed crawling with work sharing between processes
- **Hybrid**: Process-level parallelism for independent crawl paths with thread-level parallelism for HTML processing

### Search Query Parallelization

The search query process is parallelized to deliver faster results:

- **OpenMP**: Parallel BM25 score calculation across multiple documents
- **MPI**: Distributed search across partitioned indexes
- **Hybrid**: Multi-level parallel query processing

### Command-line Options for All Versions

```
Options:
  -u URL       Download and index content from a single URL
  -c URL       Crawl website starting from URL (follows links)
  -m USER      Crawl Medium profile for a specific user
  -d NUM       Maximum crawl depth (default: 2)
  -p NUM       Maximum pages to crawl (default: 10)
  -t NUM       Number of threads to use (OpenMP and Hybrid versions)
  -b           Benchmark mode
  -h           Display help message
```

## Benchmark & Performance Comparison

You can run benchmarks across all versions with:

```bash
./scripts/performance_benchmark.sh
```

This will generate performance metrics in the `data/` directory, which you can view in the web dashboard.
