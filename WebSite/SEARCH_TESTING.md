# Search Engine Testing Guide

This document provides guidance on testing the search functionality in the parallel search engine dashboard.

## Quick Start

1. Start the dashboard server:
```bash
cd WebSite
python api.py
```

2. Open the dashboard in your browser:
```
http://localhost:5001
```

3. Navigate to the "Search Engine" tab and enter a query.

## Testing with the Test Suite

We've provided a test script to help diagnose search issues:

```bash
cd WebSite
./test_search.py --suite
```

This will run a series of tests against different search engine versions.

## Testing a Specific Search

To test a specific search query:

```bash
./test_search.py -q "your search query" -v openmp
```

Available options:
- `-q, --query`: The search query
- `-v, --version`: Engine version (serial, openmp, mpi)
- `-t, --threads`: Number of threads for OpenMP (default: 4)
- `-p, --processes`: Number of processes for MPI (default: 4)
- `-c, --crawl`: URL to crawl
- `-d, --depth`: Crawl depth when using crawling (default: 1)

## Diagnosing Timeout Issues

If you're experiencing timeouts, use the search_monitor.py script:

```bash
./search_monitor.py -v openmp -q "your search query" -t 180
```

This will provide detailed diagnostics and output during the search process.

## Recommendations for Web Crawling

Web crawling can be resource-intensive and may cause timeouts. For best results:

1. Limit crawl depth to 1 or 2
2. Use MPI or OpenMP version for better performance
3. Use the extended timeout option for complex crawling

## Test on Different Datasets

You can test search on different datasets:

1. Default dataset
2. Custom dataset (use the advanced options panel)
3. Web crawling (use with caution)

For complex searches or web crawling, the OpenMP and MPI versions will perform better than the Serial version.
