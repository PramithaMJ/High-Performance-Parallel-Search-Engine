# Search Engine

A comprehensive search engine implementation that uses the BM25 ranking algorithm to search through text documents. It features both local document indexing and web crawling capabilities to build a versatile search corpus.

## Features

- Document parsing and indexing
- Web crawling and content extraction
- Medium article crawling optimization
- Stopword removal for better search quality
- BM25 ranking algorithm for relevance-based results
- Command line interface with multiple operation modes
- URL normalization and handling

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

## Dependencies

- libcurl (for web crawling)
