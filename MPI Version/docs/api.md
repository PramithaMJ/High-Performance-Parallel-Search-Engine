# API Documentation

## Index Module

### `void index_init(config_t* config)`
Initializes the index with configuration settings.

### `int index_add_document(char* document_text, char* document_id)`
Adds a document to the index.
- **Parameters**:
  - `document_text`: The text content of the document
  - `document_id`: A unique identifier for the document
- **Returns**: 0 on success, negative value on error

### `search_results_t* index_search(char* query)`
Searches the index for documents matching the query.
- **Parameters**:
  - `query`: The search query
- **Returns**: Search results structure

### `void index_cleanup()`
Frees all resources used by the index.

## Parser Module

### `parsed_document_t* parse_document(char* filename)`
Parses a document from a file.
- **Parameters**:
  - `filename`: Path to the document file
- **Returns**: Parsed document structure

### `parsed_document_t* parse_html(char* html_content, int length)`
Parses HTML content.
- **Parameters**:
  - `html_content`: HTML content string
  - `length`: Length of the HTML content
- **Returns**: Parsed document structure

### `char** tokenize(char* text, int* token_count)`
Tokenizes text into an array of tokens.
- **Parameters**:
  - `text`: Text to tokenize
  - `token_count`: Pointer to store the number of tokens
- **Returns**: Array of token strings

## Crawler Module

### `void crawler_init(config_t* config)`
Initializes the crawler with configuration settings.

### `int crawler_add_seed_url(char* url)`
Adds a seed URL to crawl.
- **Parameters**:
  - `url`: Seed URL to start crawling
- **Returns**: 0 on success, negative value on error

### `int crawler_start(int max_pages)`
Starts the crawling process.
- **Parameters**:
  - `max_pages`: Maximum number of pages to crawl
- **Returns**: Number of pages successfully crawled

### `void crawler_cleanup()`
Frees all resources used by the crawler.

## Ranking Module

### `void ranking_init(config_t* config)`
Initializes the ranking module with configuration settings.

### `void ranking_calculate_scores(search_results_t* results, char* query)`
Calculates BM25 scores for search results.
- **Parameters**:
  - `results`: Search results structure
  - `query`: The search query

### `void ranking_sort_results(search_results_t* results)`
Sorts search results by score.
- **Parameters**:
  - `results`: Search results structure
