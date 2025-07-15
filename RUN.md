# Common Structure

The project has four main versions:

- Serial Version
- OpenMP Version
- MPI Version
- Hybrid Version (MPI+OpenMP)

Each version follows a similar directory structure with:

- `bin/` - Executable files
- `src/` - Source code
- `include/` - Header files
- data - Configuration and data files
- `scripts/` - Helper scripts for running and benchmarking

## Running the Serial Version

```bash
# Navigate to the Serial Version directory
cd "Serial Version"

# Build the project
make clean
make all

# Run the search engine with a basic query
./bin/search_engine -q "your search query"

# OR crawl a website and build index
./bin/search_engine -c https://medium.com/@lpramithamj -d 2 -p 10

# Run benchmarks
./scripts/benchmark.sh
./scripts/performance_benchmark.sh
```

## Running the OpenMP Version

```bash
# Navigate to the OpenMP Version directory
cd "OpenMP Version"

# Build the project
make clean
make all

# Run with default thread configuration
./bin/search_engine -q "your search query"

# Run with specific number of threads (e.g., 8)
OMP_NUM_THREADS=8 ./bin/search_engine -c https://medium.com/@lpramithamj -d 2 -p 10
./bin/search_engine -t 6 -c https://medium.com/@lpramithamj -d 3 -p 20

# Run benchmarks
./scripts/benchmark.sh
./scripts/run_benchmark.sh
```

## Running the MPI Version

```bash
# Navigate to the MPI Version directory
cd "MPI Version"

# Build the project
make clean
make all

# Run using the provided script with default 4 processes
./scripts/run_mpi.sh

# Run with a specific number of MPI processes (e.g., 8)
./scripts/run_mpi.sh 8

# Run with specific arguments
./scripts/run_mpi.sh 4 -c https://medium.com/@lpramithamj -d 2 -p 10

# Or run directly with mpirun
mpirun -np 4 ./bin/search_engine -c https://medium.com/@lpramithamj -d 2 -p 10

# Run benchmarks
./scripts/benchmark.sh
./scripts/performance_benchmark.sh
```

## Running the Hybrid Version (MPI+OpenMP)

```bash
# Navigate to the Hybrid Version directory
cd "Hybrid Version"

# Build the project
make clean
make all

# Use the dedicated run_hybrid.sh script
# Format: ./scripts/run_hybrid.sh <MPI_PROCS> <OMP_THREADS> [other arguments]
./scripts/run_hybrid.sh 4 8 -c https://medium.com/@lpramithamj -d 2 -p 10

# Or use the convenience wrapper
./run_search -np 8 -t 8 -c https://medium.com/@lpramithamj -d 2 -p 10

# Or run directly with mpirun and environment variables
OMP_NUM_THREADS=8 mpirun -np 4 ./bin/search_engine -np 4 -t 8 -c https://medium.com/@lpramithamj -d 2 -p 10

# Run benchmarks
./scripts/benchmark.sh
./scripts/performance_benchmark.sh
```

## Using the Web Dashboard

The project also includes a web-based dashboard for visualizing and comparing performance:

```bash
# Navigate to the WebSite directory
cd WebSite

# Run the API server
python3 api.py

# Access the dashboard at http://localhost:5001
```

The dashboard allows you to:

- Compare performance between versions
- Execute searches with different engines
- View memory usage and execution metrics
- Configure build parameters

## Performance Benchmarking

To run a comprehensive benchmark across all versions:

```bash
# From the project root
bash ./Serial\ Version/scripts/performance_benchmark.sh
bash ./OpenMP\ Version/scripts/performance_benchmark.sh
bash ./MPI\ Version/scripts/performance_benchmark.sh
bash ./Hybrid\ Version/scripts/performance_benchmark.sh

# Results will be saved in the respective data/ directories as CSV files
```

The performance metrics collected include crawling time, parsing time, tokenizing time, indexing time, query processing time, memory usage, document count, unique terms, and average query latency.
