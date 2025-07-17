# CUDA+OpenMP Hybrid Parallel Search Engine

A high-performance search engine implementation that combines CUDA for GPU acceleration with OpenMP for CPU parallelization. This hybrid approach maximizes both GPU compute power and multi-core CPU capabilities to achieve unprecedented performance in document indexing and search operations.

## Features

- **Hybrid GPU+CPU Parallelism**: Combines CUDA for massive parallel GPU computation with OpenMP for efficient CPU multi-threading
- **Configurable Architecture**: Use `-g` to enable GPU acceleration and `-t` to set OpenMP thread count
- **Optimized Memory Management**: Efficient GPU memory allocation and transfer strategies
- **Parallel Document Processing**: GPU-accelerated tokenization and CPU-optimized parsing
- **Accelerated BM25 Ranking**: GPU-parallel score calculation with CPU result aggregation
- **Web Crawling Optimization**: OpenMP-accelerated URL processing and content extraction
- **Dynamic Load Balancing**: Intelligent work distribution between GPU and CPU resources
- **Performance Monitoring**: Real-time metrics for GPU utilization and CPU efficiency

## Hybrid Implementation Architecture

The search engine uses a three-level parallelism approach:

### Level 1: CUDA (GPU Parallelism)
- **Massive Parallel Processing**: Utilizes thousands of CUDA cores for parallel computation
- **Vector Operations**: GPU-accelerated string matching and similarity calculations
- **Memory Coalescing**: Optimized memory access patterns for maximum bandwidth
- **Parallel BM25 Scoring**: GPU kernels for parallel relevance score computation
- **Batch Processing**: Efficient batch operations for large document collections

### Level 2: OpenMP (CPU Parallelism)
- **Multi-threaded Processing**: CPU cores handle complex decision-making and coordination
- **I/O Operations**: Parallel file reading and web crawling operations
- **Memory Management**: Efficient CPU-GPU data transfer coordination
- **Result Aggregation**: CPU-optimized sorting and ranking of search results
- **Load Balancing**: Dynamic work distribution based on system capabilities

### Level 3: Hybrid Coordination
- **Asynchronous Processing**: Overlap GPU computation with CPU I/O operations
- **Memory Hierarchy Optimization**: Smart use of GPU global, shared, and texture memory
- **Pipeline Processing**: Streaming data processing between CPU and GPU
- **Adaptive Scheduling**: Dynamic workload allocation based on data characteristics

## Performance Benefits

The CUDA+OpenMP hybrid implementation provides exceptional performance advantages:

1. **Massive Parallelism**: GPU provides 1000+ parallel processing units vs 4-16 CPU cores
2. **Specialized Processing**: GPU excels at parallel arithmetic while CPU handles complex logic
3. **Memory Bandwidth**: GPU memory bandwidth (500+ GB/s) far exceeds CPU (50-100 GB/s)
4. **Overlap Computation**: Simultaneous GPU computation and CPU I/O operations
5. **Scalable Architecture**: Performance scales with both GPU capability and CPU cores
6. **Energy Efficiency**: GPU computation often more energy-efficient for parallel workloads

## Architecture Components

### CUDA Kernels
- `gpu_tokenize_documents()`: Parallel document tokenization
- `gpu_calculate_bm25_scores()`: Parallel BM25 score computation
- `gpu_string_search()`: Fast parallel string matching
- `gpu_vector_operations()`: Parallel vector arithmetic for ranking
- `gpu_memory_coalescing()`: Optimized memory access patterns

### OpenMP Threads
- **I/O Thread Pool**: Parallel file reading and web crawling
- **Coordination Threads**: CPU-GPU data transfer management
- **Aggregation Threads**: Result collection and final ranking
- **Monitor Threads**: Performance tracking and load balancing

### Memory Management
- **Pinned Memory**: Fast CPU-GPU transfers with pinned host memory
- **Unified Memory**: Simplified programming with automatic data migration
- **Memory Pools**: Pre-allocated GPU memory for reduced allocation overhead
- **Streaming**: Overlapped data transfer and computation

## Usage

### Prerequisites
- NVIDIA GPU with CUDA Capability 3.5+
- CUDA Toolkit 11.0+
- OpenMP-capable compiler (GCC 4.9+, Clang 3.7+)
- Minimum 4GB GPU memory recommended

### Compilation

```bash
# Build with CUDA and OpenMP support
make clean
make cuda
```

### Running the Search Engine

```bash
# Enable GPU acceleration with 8 CPU threads
./bin/search_engine -g -t 8 -q "artificial intelligence"

# GPU crawling with CPU thread optimization
./bin/search_engine -g -t 4 -c https://example.com -d 2 -p 50

# Performance benchmarking
./scripts/cuda_benchmark.sh
```

### Command Line Options

- `-g`: Enable GPU acceleration (requires CUDA-capable GPU)
- `-t NUM`: Set number of OpenMP threads (default: number of CPU cores)
- `-b NUM`: Set GPU thread block size (default: 256)
- `-m NUM`: Set GPU memory allocation strategy (0=basic, 1=unified, 2=pinned)
- `-p NUM`: Set CPU-GPU processing ratio (0-100, default: auto-detect)
- `-q QUERY`: Execute search query
- `-c URL`: Crawl website starting from URL
- `-u URL`: Download and index single URL
- `-benchmark`: Run comprehensive performance tests

## Performance Optimization Guide

### GPU Optimization
1. **Thread Block Size**: Optimize for your GPU architecture (128-512 threads)
2. **Memory Pattern**: Ensure coalesced memory access in kernels
3. **Occupancy**: Maximize GPU occupancy with proper resource usage
4. **Streaming**: Use CUDA streams for overlapped execution

### CPU Optimization
1. **Thread Count**: Set to number of physical CPU cores
2. **NUMA Awareness**: Pin threads to specific CPU cores if needed
3. **Cache Optimization**: Structure data for CPU cache efficiency
4. **I/O Parallelism**: Use OpenMP for parallel file operations

### Hybrid Optimization
1. **Work Distribution**: Balance load between GPU and CPU based on task type
2. **Memory Transfers**: Minimize CPU-GPU data transfers
3. **Pipeline Processing**: Overlap computation with data movement
4. **Adaptive Scheduling**: Dynamically adjust workload distribution

## Performance Comparison

Expected performance improvements over CPU-only implementations:

| Operation | CPU-Only | OpenMP | CUDA+OpenMP | Speedup |
|-----------|----------|---------|-------------|---------|
| Document Parsing | 100ms | 25ms | 15ms | 6.7x |
| BM25 Scoring | 200ms | 50ms | 8ms | 25x |
| String Matching | 150ms | 38ms | 5ms | 30x |
| Large Queries | 500ms | 125ms | 20ms | 25x |

*Performance varies based on hardware configuration and data characteristics*

## Hardware Requirements

### Minimum Requirements
- NVIDIA GPU: GTX 1050 / RTX 2060 or equivalent
- GPU Memory: 4GB GDDR5/6
- CPU: 4+ cores with OpenMP support
- System RAM: 8GB
- Storage: SSD recommended for large datasets

### Recommended Configuration
- NVIDIA GPU: RTX 3070 / RTX 4060 or better
- GPU Memory: 8GB+ GDDR6/6X
- CPU: 8+ cores (Intel i7/i9, AMD Ryzen 7/9)
- System RAM: 16GB+
- Storage: NVMe SSD

### Optimal Performance
- NVIDIA GPU: RTX 4080/4090, Tesla V100, A100
- GPU Memory: 16GB+ HBM2/GDDR6X
- CPU: 16+ cores with high memory bandwidth
- System RAM: 32GB+ with high-speed memory
- Storage: High-performance NVMe RAID

## Building the Project

### Dependencies
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install build-essential libcurl4-openssl-dev libomp-dev

# Install CUDA Toolkit (version 11.0+)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install cuda-toolkit
```

### Compilation Options
```bash
# Full build with optimizations
make cuda OPTIMIZE=1

# Debug build with profiling
make cuda DEBUG=1

# Build for specific GPU architecture
make cuda CUDA_ARCH=sm_75  # RTX 20 series
make cuda CUDA_ARCH=sm_86  # RTX 30 series
```

## Performance Monitoring

The hybrid implementation includes comprehensive performance monitoring:

- **GPU Metrics**: Utilization, memory usage, kernel execution times
- **CPU Metrics**: Thread efficiency, cache hit rates, I/O throughput
- **Hybrid Metrics**: Data transfer overhead, load balance efficiency
- **Energy Metrics**: Power consumption analysis (requires nvidia-ml-py)

## Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce batch size or enable unified memory
2. **Low GPU Utilization**: Increase thread block size or data parallelism
3. **CPU Bottleneck**: Increase OpenMP thread count or optimize I/O
4. **Memory Transfer Overhead**: Use pinned memory or streaming

### Debug Tools
```bash
# Check GPU capability
./bin/search_engine -gpu-info

# Profile GPU kernels
nvprof ./bin/search_engine -g -q "test query"

# Monitor memory usage
nvidia-smi -l 1
```

## Future Enhancements

- **Multi-GPU Support**: Scale across multiple GPUs using NCCL
- **Tensor Core Utilization**: Leverage mixed-precision for compatible GPUs
- **Dynamic Parallelism**: GPU kernel launches from GPU kernels
- **NVLink Optimization**: High-bandwidth GPU-GPU communication
- **CUDA Graphs**: Reduce kernel launch overhead for repetitive operations
