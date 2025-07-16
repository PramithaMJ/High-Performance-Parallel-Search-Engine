# User Guide: High-Performance Parallel Search Engine Dashboard

This guide provides instructions for using the High-Performance Parallel Search Engine Dashboard interface.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Dashboard Tab](#dashboard-tab)
3. [Search Tab](#search-tab)
4. [Performance Tab](#performance-tab)
5. [Compare Tab](#compare-tab)
6. [Build Tab](#build-tab)
7. [Settings Tab](#settings-tab)
8. [Tips and Tricks](#tips-and-tricks)
9. [Troubleshooting](#troubleshooting)

## Getting Started

### Accessing the Dashboard

1. Make sure the dashboard server is running by executing `./start_dashboard.sh` in the website directory
2. Open your web browser and navigate to `http://localhost:5000` (or your configured port)
3. The dashboard will load and display the main interface with multiple tabs

### Interface Overview

The dashboard is organized into six main tabs:

- **Dashboard**: Overview and summary information
- **Search**: Execute search queries with different engine implementations
- **Performance**: View detailed performance metrics and charts
- **Compare**: Compare results and performance between implementations
- **Build**: Build and manage search engine implementations
- **Settings**: Configure dashboard settings and preferences

Use the navigation bar at the top to switch between tabs. Each tab provides specific functionality related to the search engine.

## Dashboard Tab

The Dashboard tab provides an overview of the system status and recent activity.

### Key Components

- **System Status**: Shows the status of each search engine implementation
- **Quick Stats**: Displays key performance metrics at a glance
- **Recent Searches**: Lists recent search queries and their performance
- **Performance Snapshot**: Shows a graph of recent performance trends
- **Quick Actions**: Provides shortcuts to common tasks

### Using the Dashboard

1. Check the status indicators to ensure all implementations are operational
2. View performance trends in the charts to identify any issues
3. Use quick actions to navigate to specific features
4. Monitor recent searches to track usage patterns

## Search Tab

The Search tab allows you to execute search queries using different implementations of the search engine.

### Running a Search

1. Select the search engine implementation (Serial, OpenMP, MPI, or Hybrid)
2. Enter your search query in the search box
3. Adjust search parameters if needed:
   - Max Results: Number of results to return
   - Timeout: Maximum search time in seconds
   - Ranking Algorithm: Method used to rank results
4. Click the "Search" button to execute the query
5. View the results in the results panel

### Search Results

The results panel displays:
- Total number of results found
- Execution time
- Document matches with highlighted terms
- Relevance score for each result

### Additional Options

- **Export Results**: Save search results as JSON or CSV
- **Save Query**: Save the current query for future use
- **Query History**: View and rerun previous queries
- **Advanced Parameters**: Configure additional search parameters

## Performance Tab

The Performance tab provides detailed metrics and visualizations for analyzing search engine performance.

### Performance Metrics

View performance data including:
- Query execution time
- Memory usage
- CPU utilization
- Index access patterns
- Throughput (queries per second)

### Performance Charts

Interactive charts allow you to:
1. View historical performance trends
2. Compare metrics across different time periods
3. Analyze performance bottlenecks
4. Export chart data for external analysis

### Running Performance Tests

1. Select the "Run Test" option
2. Choose the test type:
   - Basic Performance Test: Quick overview of performance
   - Stress Test: Test under heavy load
   - Throughput Test: Maximum queries per second
   - Custom Test: Define your own test parameters
3. Configure test parameters
4. Click "Start Test" to begin
5. View real-time results as the test progresses

## Compare Tab

The Compare tab allows you to directly compare different search engine implementations.

### Running Comparisons

1. Enter a search query
2. Select which implementations to compare (check multiple boxes)
3. Set common parameters for all implementations
4. Click "Run Comparison" to execute
5. View side-by-side results and performance metrics

### Comparison Results

The comparison view shows:
- Search results from each implementation
- Performance metrics side-by-side
- Result differences (if any)
- Speedup ratios between implementations

### Comparison Charts

Interactive charts display:
- Performance comparison between implementations
- Scaling efficiency with different parameters
- Resource utilization comparison

### Saving Comparisons

Use the "Save Comparison" button to export the comparison results as:
- PDF report
- CSV data
- JSON data

## Build Tab

The Build tab provides tools for building and managing the search engine implementations.

### Build Status

View the status of each implementation:
- Build date and time
- Version information
- Compilation flags used
- Build success or failure

### Building Implementations

1. Select the implementation to build
2. Choose build options:
   - Clean Build: Rebuild from scratch
   - Debug Mode: Include debug information
   - Optimization Level: Set compiler optimizations
3. Click "Build" to start the build process
4. Monitor the build log for progress and errors

### Configuration

Modify configuration files for each implementation:
1. Select the implementation
2. Edit the configuration parameters
3. Save changes
4. Rebuild the implementation if necessary

## Settings Tab

The Settings tab allows you to configure the dashboard interface and behavior.

### General Settings

- **Theme**: Choose between light and dark mode
- **Refresh Rate**: Set automatic data refresh interval
- **Default Implementation**: Select the default search engine implementation
- **Results Per Page**: Configure default number of results

### Advanced Settings

- **API Settings**: Configure API access and behavior
- **Logging**: Set logging levels and options
- **Performance Monitoring**: Configure metric collection
- **Diagnostics**: Tools for troubleshooting

### User Preferences

- **Saved Queries**: Manage saved search queries
- **Favorite Charts**: Configure dashboard widgets
- **Keyboard Shortcuts**: Customize keyboard navigation
- **Export/Import**: Backup and restore settings

## Tips and Tricks

### Keyboard Shortcuts

- **Ctrl+Enter**: Execute search
- **Ctrl+Tab**: Navigate between tabs
- **Alt+1-6**: Jump to specific tab
- **Ctrl+S**: Save current view
- **F5**: Refresh data
- **Ctrl+F**: Find in results

### Performance Optimization

- Use the OpenMP implementation for multi-core machines
- Use the MPI implementation for distributed computing
- The Hybrid implementation works best on clusters with multi-core nodes
- Filter results to improve search speed
- Index only necessary files to reduce memory usage

### Interface Customization

- Rearrange dashboard widgets by dragging and dropping
- Create custom chart views for frequently monitored metrics
- Save and load dashboard configurations
- Use the comparison template feature for repeated tests

## Troubleshooting

### Common Issues

#### Search Engine Not Responding

1. Check if the executable exists in the expected location
2. Verify that the executable has execute permissions
3. Check system resource usage (CPU, memory)
4. Look for errors in the build log

#### Poor Performance

1. Check if other processes are using system resources
2. Verify that the correct implementation is selected
3. Try with a smaller dataset or query
4. Check configuration parameters for optimization

#### Build Failures

1. Check the build log for compiler errors
2. Verify that all dependencies are installed
3. Check file permissions in source directories
4. Try a clean build

#### Dashboard Not Loading

1. Verify that the server is running
2. Check for JavaScript errors in the browser console
3. Clear browser cache and reload
4. Try a different browser

### Getting Help

If you encounter issues not covered in this guide:

1. Check the logs in the server terminal
2. Review the error messages in the Settings > Diagnostics tab
3. Consult the project documentation in the docs directory
4. Submit an issue with detailed information about the problem
