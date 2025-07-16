# WebSocket API Documentation

The High-Performance Parallel Search Engine Dashboard provides a WebSocket API for real-time updates and interactions. This document describes how to use this API.

## Connection

Connect to the WebSocket API at:

```
ws://localhost:5000/api/ws
```

## Authentication

If authentication is enabled in the server configuration, include an authentication token in the connection URL:

```
ws://localhost:5000/api/ws?token=your_api_key
```

## Message Format

All messages sent and received through the WebSocket are JSON objects with the following structure:

```json
{
  "type": "message_type",
  "data": {
    // Message specific data
  },
  "id": "optional_message_id",
  "timestamp": "ISO_timestamp"
}
```

## Message Types

### Connection Messages

#### hello

Sent by the server immediately after connection is established.

```json
{
  "type": "hello",
  "data": {
    "server": "SearchEngine WebSocket API",
    "version": "1.0.0",
    "features": ["search_updates", "build_status", "metrics_stream"]
  },
  "timestamp": "2025-07-17T12:00:00Z"
}
```

#### subscribe

Send to subscribe to specific event types.

```json
{
  "type": "subscribe",
  "data": {
    "topics": ["search_updates", "build_status"]
  },
  "id": "sub1"
}
```

Response:

```json
{
  "type": "subscription_ack",
  "data": {
    "topics": ["search_updates", "build_status"],
    "status": "success"
  },
  "id": "sub1",
  "timestamp": "2025-07-17T12:01:00Z"
}
```

#### unsubscribe

Send to unsubscribe from specific event types.

```json
{
  "type": "unsubscribe",
  "data": {
    "topics": ["build_status"]
  },
  "id": "unsub1"
}
```

Response:

```json
{
  "type": "unsubscription_ack",
  "data": {
    "topics": ["build_status"],
    "status": "success"
  },
  "id": "unsub1",
  "timestamp": "2025-07-17T12:02:00Z"
}
```

### Search Related Messages

#### start_search

Send to start a search operation.

```json
{
  "type": "start_search",
  "data": {
    "query": "parallel algorithms",
    "version": "openmp",
    "max_results": 20,
    "timeout": 30
  },
  "id": "search1"
}
```

#### search_update

Received when there are updates to an ongoing search operation.

```json
{
  "type": "search_update",
  "data": {
    "search_id": "search1",
    "status": "in_progress",
    "progress": 45,
    "current_stage": "ranking_results",
    "elapsed_ms": 150,
    "results_found_so_far": 12
  },
  "id": "search1",
  "timestamp": "2025-07-17T12:03:30Z"
}
```

#### search_complete

Received when a search operation completes.

```json
{
  "type": "search_complete",
  "data": {
    "search_id": "search1",
    "status": "complete",
    "execution_time_ms": 350,
    "total_results": 28,
    "results": [
      {
        "id": "doc123",
        "title": "Introduction to Parallel Computing",
        "snippet": "...overview of parallel computing algorithms...",
        "score": 0.89,
        "url": "dataset/doc123.txt"
      },
      // More results...
    ]
  },
  "id": "search1",
  "timestamp": "2025-07-17T12:03:45Z"
}
```

#### search_error

Received when a search operation encounters an error.

```json
{
  "type": "search_error",
  "data": {
    "search_id": "search1",
    "error": "Timeout exceeded",
    "error_details": "Search operation took longer than the specified timeout (30s)"
  },
  "id": "search1",
  "timestamp": "2025-07-17T12:04:15Z"
}
```

### Build Related Messages

#### start_build

Send to start building a search engine version.

```json
{
  "type": "start_build",
  "data": {
    "version": "mpi",
    "clean": true,
    "options": {
      "debug": false
    }
  },
  "id": "build1"
}
```

#### build_update

Received when there are updates to an ongoing build operation.

```json
{
  "type": "build_update",
  "data": {
    "build_id": "build1",
    "status": "in_progress",
    "progress": 65,
    "current_stage": "compiling",
    "elapsed_ms": 2500,
    "log": "Compiling file 23 of 35..."
  },
  "id": "build1",
  "timestamp": "2025-07-17T12:10:30Z"
}
```

#### build_complete

Received when a build operation completes.

```json
{
  "type": "build_complete",
  "data": {
    "build_id": "build1",
    "status": "complete",
    "build_time_ms": 4500,
    "success": true,
    "executable": "/path/to/mpi/search_engine",
    "log": "Build completed successfully..."
  },
  "id": "build1",
  "timestamp": "2025-07-17T12:11:15Z"
}
```

#### build_error

Received when a build operation encounters an error.

```json
{
  "type": "build_error",
  "data": {
    "build_id": "build1",
    "error": "Compilation failed",
    "error_details": "Error in src/indexer.cpp:234: undefined reference to `MPI_Init`",
    "log": "Full build log..."
  },
  "id": "build1",
  "timestamp": "2025-07-17T12:05:45Z"
}
```

### Performance Metrics Messages

#### start_metrics_stream

Send to start receiving real-time metrics.

```json
{
  "type": "start_metrics_stream",
  "data": {
    "versions": ["openmp", "mpi"],
    "metrics": ["cpu_usage", "memory_usage", "query_time"],
    "interval_ms": 1000
  },
  "id": "metrics1"
}
```

#### metrics_update

Received periodically with current performance metrics.

```json
{
  "type": "metrics_update",
  "data": {
    "metrics_id": "metrics1",
    "timestamp": "2025-07-17T12:15:00Z",
    "metrics": {
      "openmp": {
        "cpu_usage": 75.2,
        "memory_usage": 128.5,
        "query_time": 155
      },
      "mpi": {
        "cpu_usage": 85.5,
        "memory_usage": 210.3,
        "query_time": 120
      }
    }
  },
  "id": "metrics1",
  "timestamp": "2025-07-17T12:15:00Z"
}
```

#### stop_metrics_stream

Send to stop receiving real-time metrics.

```json
{
  "type": "stop_metrics_stream",
  "data": {
    "metrics_id": "metrics1"
  },
  "id": "stopmetrics1"
}
```

### System Status Messages

#### system_status

Received periodically or when system status changes.

```json
{
  "type": "system_status",
  "data": {
    "versions": {
      "serial": "ready",
      "openmp": "ready",
      "mpi": "building",
      "hybrid": "ready"
    },
    "system": {
      "cpu_usage": 45.2,
      "memory_available_mb": 4096,
      "disk_space_available_mb": 15360
    },
    "active_operations": {
      "searches": 2,
      "builds": 1
    }
  },
  "timestamp": "2025-07-17T12:20:00Z"
}
```

## Error Handling

If an error occurs with a WebSocket message, you'll receive an error message:

```json
{
  "type": "error",
  "data": {
    "code": "INVALID_REQUEST",
    "message": "Invalid message format or parameters",
    "original_request": {
      "type": "invalid_type",
      "data": {}
    }
  },
  "timestamp": "2025-07-17T12:25:00Z"
}
```

## Connection Management

The WebSocket server will ping clients periodically to keep connections alive. Clients should respond to ping frames according to the WebSocket protocol.

If no activity is detected for 5 minutes, the server will close the connection with code 1000 (normal closure).

## Rate Limiting

WebSocket connections are limited to:
- 1 connection per authenticated user
- 5 active subscriptions per connection
- 10 operations (searches/builds) per minute

If rate limits are exceeded, you'll receive an error message and the connection may be closed.

## Examples

### JavaScript Client Example

```javascript
// Connect to WebSocket API
const ws = new WebSocket('ws://localhost:5000/api/ws');

// Handle connection opening
ws.addEventListener('open', (event) => {
  console.log('Connected to WebSocket API');
  
  // Subscribe to topics
  ws.send(JSON.stringify({
    type: 'subscribe',
    data: {
      topics: ['search_updates', 'system_status']
    },
    id: 'subscription1'
  }));
});

// Handle incoming messages
ws.addEventListener('message', (event) => {
  const message = JSON.parse(event.data);
  
  switch(message.type) {
    case 'hello':
      console.log('Server hello:', message.data);
      break;
    case 'subscription_ack':
      console.log('Subscribed to:', message.data.topics);
      break;
    case 'search_update':
      updateSearchProgress(message.data);
      break;
    case 'search_complete':
      displaySearchResults(message.data);
      break;
    case 'system_status':
      updateSystemStatus(message.data);
      break;
    case 'error':
      console.error('WebSocket error:', message.data);
      break;
  }
});

// Start a search operation
function startSearch(query) {
  ws.send(JSON.stringify({
    type: 'start_search',
    data: {
      query: query,
      version: 'openmp',
      max_results: 20
    },
    id: 'search_' + Date.now()
  }));
}

// Handle WebSocket closure
ws.addEventListener('close', (event) => {
  console.log('WebSocket connection closed:', event.code, event.reason);
});
```

## Security Considerations

- All WebSocket connections should be made over TLS (wss://) in production environments
- Authentication tokens should have appropriate expiration times
- Sensitive information should not be transmitted over WebSocket connections
