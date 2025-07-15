# High Performance Parallel Search Engine

A high-performance search engine with multiple parallelization strategies using OpenMP and MPI. This project features web crawling, document indexing, BM25 ranking, and a web-based dashboard for performance monitoring.

![Search Dashboard](/Users/pramithajayasooriya/Desktop/High-Performance-Parallel-Search-Engine/Screenshots/Screenshot%202025-06-19%20at%201.47.35%20AM.png)

## Technologies Used

- **Languages**: C, Python, JavaScript, HTML/CSS
- **Parallelization**: OpenMP (shared memory), MPI (distributed memory)
- **Web Crawling**: libcurl
- **Web Interface**: NodeJS
- **Performance Metrics**: Custom benchmarking tools
- **Search Algorithm**: BM25 ranking

## Quick Start

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

### Running the Search Engine

#### Serial Version
```bash
cd Serial\ Version/
./bin/search_engine -u https://example.com
```

#### OpenMP Version
```bash
cd OpenMP\ Version/
./bin/search_engine_omp -u https://example.com -t 4  # Use 4 threads
```

#### MPI Version
```bash
cd MPI\ Version/
mpirun -np 4 ./bin/search_engine_mpi -u https://example.com  # Use 4 processes
```

### Command-line Options

- `-u URL`: Download and index content from a single URL
- `-c URL`: Crawl website starting from URL
- `-m USER`: Crawl Medium profile for a specific user
- `-d NUM`: Maximum crawl depth (default: 2)
- `-p NUM`: Maximum pages to crawl (default: 10)
- `-t NUM`: Number of threads to use (OpenMP version)
- `-b`: Benchmark mode

### Running the Web Dashboard

```bash
cd WebSite/
./start_dashboard.sh
```

Access the dashboard at http://localhost:5001

## Screenshots

### Search Interface
![Search Interface](/Users/pramithajayasooriya/Desktop/High-Performance-Parallel-Search-Engine/Screenshots/Screenshot%202025-06-19%20at%201.48.00%20AM.png)

### Performance Comparison
![Performance Comparison](/Users/pramithajayasooriya/Desktop/High-Performance-Parallel-Search-Engine/Screenshots/Screenshot%202025-06-19%20at%201.48.19%20AM.png)

### Configuration Settings
![Configuration Settings](/Users/pramithajayasooriya/Desktop/High-Performance-Parallel-Search-Engine/Screenshots/Screenshot%202025-06-19%20at%201.50.23%20AM.png)

## Performance

The search engine demonstrates significant performance improvements with parallelization:

- **Serial Version**: Average query time: 365ms
- **OpenMP Version**: Average query time: 124ms (2.9x speedup)
- **MPI Version**: Average query time: 78ms (4.7x speedup vs serial)

## Benchmarking

Run performance benchmarks with:

```bash
./scripts/performance_benchmark.sh
```

Results will be saved in the `data/` directory for analysis.
