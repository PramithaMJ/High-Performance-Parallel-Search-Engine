# Search Engine Dashboard Documentation

## Overview

The High-Performance Parallel Search Engine Dashboard is a web-based interface that allows users to interact with different implementations of the search engine (Serial, OpenMP, MPI, and Hybrid versions). This dashboard provides features for searching, comparing performance, and managing the different versions of the search engine.

## Getting Started

### Prerequisites

- Python 3.6 or higher
- Flask and Flask-CORS Python packages
- Web browser with JavaScript enabled
- Access to the search engine executables

### Starting the Dashboard

1. Navigate to the website directory
2. Run the `start_dashboard.sh` script:
   ```bash
   cd website
   chmod +x start_dashboard.sh
   ./start_dashboard.sh
   ```
3. By default, the server will start on port 5000. You can specify a different port:
   ```bash
   ./start_dashboard.sh 8080
   ```
4. Open your web browser and navigate to `http://localhost:5000` (or your specified port)

## Features

### Dashboard Tab

The Dashboard tab provides an overview of:
- System status for each search engine version
- Recent search queries
- Performance metrics summary
- Quick access to frequent actions

### Search Tab

The Search tab allows you to:
- Select which search engine implementation to use
- Enter search queries with custom parameters
- View search results with highlighting
- Export search results

### Performance Tab

The Performance tab provides:
- Detailed performance metrics for each search engine implementation
- Interactive graphs and charts
- Historical performance data
- Ability to run custom performance tests

### Compare Tab

The Compare tab allows you to:
- Run the same search query across multiple implementations
- Compare search results and performance side-by-side
- Generate comparison reports
- Visualize performance differences

### Build Tab

The Build tab provides:
- Status of each search engine implementation
- Options to build/rebuild any implementation
- Build logs and error reporting
- Configuration options

### Settings Tab

The Settings tab allows you to configure:
- Dashboard UI preferences
- Default search parameters
- Performance monitoring settings
- Advanced configuration options

## API Reference

The dashboard communicates with the search engine implementations via a Flask API. The main endpoints are:

### GET /api/status

Returns the status of all search engine implementations.

### POST /api/search

Executes a search query with the specified parameters.

**Parameters:**
- `query`: The search query string
- `version`: The search engine implementation to use (serial, openmp, mpi, hybrid)
- `max_results`: Maximum number of results to return (optional)
- `timeout`: Search timeout in seconds (optional)

### GET /api/metrics

Returns performance metrics for the search engine implementations.

**Parameters:**
- `version`: The specific version to get metrics for (optional)
- `from_date`: Start date for metrics (optional)
- `to_date`: End date for metrics (optional)

### POST /api/build

Builds a specific search engine implementation.

**Parameters:**
- `version`: The implementation to build (serial, openmp, mpi, hybrid)
- `clean`: Whether to perform a clean build (optional)

### POST /api/compare

Runs a comparison between multiple search engine implementations.

**Parameters:**
- `query`: The search query string
- `versions`: Array of versions to compare
- `max_results`: Maximum number of results (optional)
- `timeout`: Search timeout in seconds (optional)

## Configuration

The dashboard can be configured by editing the `config.ini` file in the website directory. Key configuration options include:

- Server settings (port, debug mode)
- Paths to search engine executables
- Default search parameters
- UI preferences
- Performance monitoring settings

## Troubleshooting

### Common Issues

1. **Dashboard won't start**
   - Ensure Python and Flask are properly installed
   - Check that the `api.py` file has execute permissions
   - Verify the port is not in use by another application

2. **Search engine not found**
   - Check the paths in `config.ini` point to the correct executables
   - Ensure the executables are built and have execute permissions

3. **API errors**
   - Check the server logs for detailed error messages
   - Ensure the search engine executables are working correctly
   - Verify that the dataset files are accessible

### Logging

Logs are written to:
- Server log: Console output when running `start_dashboard.sh`
- Performance metrics: File specified in `config.ini` (`PERFORMANCE_LOG_PATH`)
- Client-side errors: Available in the browser console and in the Settings tab error log

## Advanced Usage

### Custom Datasets

To use a custom dataset:
1. Place the dataset files in the appropriate directory for each search engine version
2. Update the configuration files for each search engine implementation
3. Rebuild the search engines through the Build tab

### Performance Testing

For advanced performance testing:
1. Navigate to the Performance tab
2. Select "Custom Test" from the dropdown
3. Configure the test parameters
4. Run the test and analyze the results

### API Integration

The dashboard API can be accessed programmatically:
```python
import requests

# Example: Run a search with the OpenMP implementation
response = requests.post('http://localhost:5000/api/search', json={
    'query': 'your search query',
    'version': 'openmp',
    'max_results': 20
})

results = response.json()
print(results)
```
