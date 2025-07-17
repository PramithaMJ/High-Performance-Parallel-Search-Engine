# Installation Guide

## Prerequisites

Before installing the search engine, ensure you have the following prerequisites:

- GCC compiler (version 8.0 or higher)
- libcurl development package
- pkg-config
- make

### Installing Prerequisites on Different Systems

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install gcc libcurl4-openssl-dev pkg-config make
```

#### macOS (using Homebrew)
```bash
brew install gcc pkg-config curl
```

#### Windows (using MSYS2)
```bash
pacman -S gcc pkg-config libcurl-devel make
```

## Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/search-engine.git
   cd search-engine
   ```

2. Build the search engine:
   ```bash
   make production
   ```

3. Run tests to ensure everything is working:
   ```bash
   make all
   ./bin/test_url_normalization
   ./bin/test_medium_urls
   ```

4. Update configuration if needed:
   - Open `config.ini` and modify settings as required

## Running the Search Engine

After installation, you can run the search engine with various options:

```bash
# Index documents in the dataset directory
./bin/search_engine -i ./dataset

# Search for a query
./bin/search_engine -q "your search query"

# Crawl a website and index its content
./bin/search_engine -c "https://example.com" -d 2
```

## Troubleshooting

### Common Issues

1. **Compilation errors related to libcurl**
   - Ensure libcurl development packages are installed
   - Check that pkg-config is correctly configured

2. **Slow performance during indexing**
   - Adjust `INDEX_BLOCK_SIZE` in the configuration file
   - Reduce `MAX_TOKENS_PER_DOC` if memory usage is high

3. **Crawler not working**
   - Check internet connectivity
   - Verify that the website allows crawling in its robots.txt

## Support

For additional support, please open an issue on the GitHub repository or contact the developers.
