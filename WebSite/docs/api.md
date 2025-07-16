# API Documentation

This document provides detailed information about the High-Performance Parallel Search Engine Dashboard API.

## Base URL

All API endpoints are relative to the base URL of the dashboard server. By default, this is:

```
http://localhost:5000
```

## API Endpoints

### Status

#### GET /api/status

Returns the status of all search engine implementations.

**Response:**

```json
{
  "status": "success",
  "data": {
    "serial": {
      "status": "ready",
      "version": "1.0.0",
      "executable": "/path/to/serial/search_engine",
      "last_build": "2025-07-15T10:30:45Z"
    },
    "openmp": {
      "status": "ready",
      "version": "1.0.0",
      "executable": "/path/to/openmp/search_engine",
      "last_build": "2025-07-15T10:35:12Z"
    },
    "mpi": {
      "status": "ready",
      "version": "1.0.0",
      "executable": "/path/to/mpi/search_engine",
      "last_build": "2025-07-15T10:40:33Z"
    },
    "hybrid": {
      "status": "ready",
      "version": "1.0.0",
      "executable": "/path/to/hybrid/search_engine",
      "last_build": "2025-07-15T10:45:55Z"
    }
  }
}
```

### Search

#### POST /api/search

Executes a search query with the specified parameters.

**Request Parameters:**

| Parameter    | Type   | Required | Description                                             |
|--------------|--------|----------|---------------------------------------------------------|
| query        | string | Yes      | The search query string                                 |
| version      | string | Yes      | The search engine implementation to use                 |
| max_results  | number | No       | Maximum number of results to return (default: 10)       |
| timeout      | number | No       | Search timeout in seconds (default: 30)                 |
| ranking      | string | No       | Ranking algorithm (default: bm25)                       |
| highlight    | boolean| No       | Whether to highlight matching terms (default: true)     |

**Example Request:**

```json
{
  "query": "parallel computing algorithms",
  "version": "openmp",
  "max_results": 20,
  "timeout": 15,
  "ranking": "bm25",
  "highlight": true
}
```

**Response:**

```json
{
  "status": "success",
  "data": {
    "query": "parallel computing algorithms",
    "version": "openmp",
    "execution_time_ms": 235,
    "total_results": 42,
    "results": [
      {
        "id": "doc123",
        "title": "Introduction to Parallel Computing",
        "snippet": "...overview of <em>parallel</em> <em>computing</em> <em>algorithms</em> and their applications...",
        "score": 0.89,
        "url": "dataset/doc123.txt"
      },
      // More results...
    ]
  }
}
```

### Metrics

#### GET /api/metrics

Returns performance metrics for the search engine implementations.

**Query Parameters:**

| Parameter  | Type   | Required | Description                                      |
|------------|--------|----------|--------------------------------------------------|
| version    | string | No       | Specific version to get metrics for              |
| from_date  | string | No       | Start date for metrics (ISO format)              |
| to_date    | string | No       | End date for metrics (ISO format)                |
| metric     | string | No       | Specific metric to retrieve                      |

**Example Response:**

```json
{
  "status": "success",
  "data": {
    "serial": {
      "avg_query_time_ms": 450,
      "max_query_time_ms": 1200,
      "avg_memory_usage_mb": 85,
      "queries_per_second": 2.2
    },
    "openmp": {
      "avg_query_time_ms": 150,
      "max_query_time_ms": 400,
      "avg_memory_usage_mb": 110,
      "queries_per_second": 6.7
    },
    // More versions...
  }
}
```

### Build

#### POST /api/build

Builds a specific search engine implementation.

**Request Parameters:**

| Parameter  | Type    | Required | Description                               |
|------------|---------|----------|-------------------------------------------|
| version    | string  | Yes      | The implementation to build               |
| clean      | boolean | No       | Whether to perform a clean build          |
| options    | object  | No       | Additional build options                  |

**Example Request:**

```json
{
  "version": "mpi",
  "clean": true,
  "options": {
    "debug": true,
    "optimizations": "O2"
  }
}
```

**Response:**

```json
{
  "status": "success",
  "data": {
    "version": "mpi",
    "build_time_ms": 3450,
    "output": "Build output...",
    "success": true,
    "executable": "/path/to/mpi/search_engine",
    "timestamp": "2025-07-17T14:30:22Z"
  }
}
```

### Compare

#### POST /api/compare

Runs a comparison between multiple search engine implementations.

**Request Parameters:**

| Parameter    | Type     | Required | Description                                      |
|--------------|----------|----------|--------------------------------------------------|
| query        | string   | Yes      | The search query string                          |
| versions     | string[] | Yes      | Array of versions to compare                     |
| max_results  | number   | No       | Maximum number of results (default: 10)          |
| timeout      | number   | No       | Search timeout in seconds (default: 30)          |
| metrics      | string[] | No       | Specific metrics to compare                      |

**Example Request:**

```json
{
  "query": "distributed algorithms",
  "versions": ["serial", "openmp", "mpi"],
  "max_results": 10,
  "timeout": 20,
  "metrics": ["execution_time", "memory_usage"]
}
```

**Response:**

```json
{
  "status": "success",
  "data": {
    "query": "distributed algorithms",
    "comparison": [
      {
        "version": "serial",
        "execution_time_ms": 550,
        "memory_usage_mb": 76,
        "result_count": 15
      },
      {
        "version": "openmp",
        "execution_time_ms": 180,
        "memory_usage_mb": 105,
        "result_count": 15
      },
      {
        "version": "mpi",
        "execution_time_ms": 120,
        "memory_usage_mb": 210,
        "result_count": 15
      }
    ],
    "result_differences": {
      "identical": true,
      "differences": []
    }
  }
}
```

### Configuration

#### GET /api/config

Retrieves the current configuration settings.

**Response:**

```json
{
  "status": "success",
  "data": {
    "serial": {
      "path": "../Serial Version/bin/search_engine",
      "default_params": {
        "max_results": 10,
        "timeout": 30
      }
    },
    // Other versions...
    "ui": {
      "dark_mode": true,
      "enable_animations": true,
      "auto_refresh_interval": 5000
    }
  }
}
```

#### POST /api/config

Updates configuration settings.

**Request Parameters:**

| Parameter | Type   | Required | Description                    |
|-----------|--------|----------|--------------------------------|
| section   | string | Yes      | Configuration section to update |
| settings  | object | Yes      | New settings to apply          |

**Example Request:**

```json
{
  "section": "ui",
  "settings": {
    "dark_mode": false,
    "auto_refresh_interval": 10000
  }
}
```

**Response:**

```json
{
  "status": "success",
  "data": {
    "updated": ["ui.dark_mode", "ui.auto_refresh_interval"],
    "timestamp": "2025-07-17T15:20:33Z"
  }
}
```

## Error Handling

All API endpoints return appropriate HTTP status codes:

- 200: Success
- 400: Bad request (invalid parameters)
- 404: Resource not found
- 500: Internal server error

Error responses follow this format:

```json
{
  "status": "error",
  "error": {
    "code": "INVALID_PARAMETER",
    "message": "Invalid parameter: version must be one of [serial, openmp, mpi, hybrid]"
  }
}
```

## Rate Limiting

API requests are limited to 60 requests per minute per IP address. When rate limited, the API will return a 429 status code with headers indicating when you can resume making requests.

## Authentication

For production deployments, API authentication can be enabled in the configuration. When enabled, API requests must include an `Authorization` header with a valid API key.

Example:
```
Authorization: Bearer api_key_here
```

## Versioning

The API version is included in all responses in the `X-API-Version` header. The current version is `1.0`.

## WebSocket API

For real-time updates, a WebSocket API is available at:

```
ws://localhost:5000/api/ws
```

This can be used to receive real-time updates for:
- Search progress
- Build status
- Performance metrics

Documentation for the WebSocket API can be found in [websocket_api.md](websocket_api.md).
