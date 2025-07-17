# MPI Search Engine Troubleshooting Guide

## Common Issues and Solutions

### 1. "Only seeing Process 0"
**Problem**: Only one process is shown, even when running with `-np 8`
**Solution**: 
- Use `mpirun -np 8` instead of `./bin/search_engine -np 8`
- Correct: `mpirun -np 8 ./bin/search_engine -m @lpramithamj`
- Wrong: `./bin/search_engine -np 8 -m @lpramithamj`

### 2. "Multiple input prompts appearing"
**Problem**: Chaotic output with multiple "Enter your search query:" prompts
**Solution**: 
- Fixed in the latest code - only rank 0 should handle input
- If still occurring, try recompiling: `./run_complete.sh --compile-only`

### 3. "mpirun command not found"
**Problem**: MPI is not installed
**Solution**: 
```bash
# On macOS with Homebrew
brew install open-mpi

# Verify installation
mpirun --version
mpicc --version
```

### 4. "No files to process"
**Problem**: Dataset directory is empty
**Solution**: 
- Add text files to the `dataset/` directory
- Or use crawling: `mpirun -np 8 ./bin/search_engine -m @lpramithamj`

### 5. "Compilation errors"
**Problem**: Missing dependencies or compilation failures
**Solution**: 
```bash
# Install dependencies
brew install open-mpi curl pkg-config

# Try the comprehensive runner
./run_complete.sh --compile-only
```

## Quick Testing Commands

### Test 1: Basic MPI functionality
```bash
./run_complete.sh --test-only
```

### Test 2: Compile only
```bash
./run_complete.sh --compile-only
```

### Test 3: Run with 8 processes
```bash
./run_complete.sh -np 8 -m @lpramithamj -d 3 -p 10
```

### Test 4: Simple run with existing dataset
```bash
./run_complete.sh -np 4
```

## Expected Output for Successful Run

```
 MPI Search Engine Comprehensive Runner
========================================
 Checking dependencies...
 mpicc found: ...
 mpirun found: ...
 libcurl found
🔨 Compiling MPI Search Engine...
 Compilation successful
🧪 Testing MPI setup...
 MPI test successful
 Checking dataset...
 Found X files in dataset
🏃 Running MPI Search Engine with 8 processes...
----------------------------------------
MPI Search Engine started with 8 processes
Running with 8 processes
Process 0: handling files 0 to 1 (total: 2 files)
Process 1: handling files 2 to 3 (total: 2 files)
Process 2: handling files 4 to 5 (total: 2 files)
...
Search engine ready for queries.
Enter your search query: 
```

## Understanding the "25" in Crawling Output

When you see "Thread 0 crawling [23/25]", the "25" refers to:
- Maximum pages to crawl from Medium profile (set in code)
- For `-m @username`: default is 25 pages
- For `-c URL`: default is 10 pages
- Can be changed with `-p NUM` flag

## Performance Tips

1. **Optimal Process Count**: Use 4-8 processes for best performance
2. **File Distribution**: Each process should handle 2-5 files ideally
3. **Memory Usage**: Monitor with `top` or `htop` during execution
4. **Network Crawling**: Use lower process counts (2-4) for web crawling to avoid overwhelming servers

## Debug Mode

Set `debug_mode = true` in `mpi_config.ini` for verbose output.
