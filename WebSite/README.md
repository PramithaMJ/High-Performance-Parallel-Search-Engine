# High-Performance Parallel Search Engine Dashboard

This dashboard provides a modern web-based interface to demonstrate, compare, and control the three versions of the search engine (Serial, OpenMP, and MPI).

## Features

- **Interactive Dashboard**: Visualize performance comparisons between all three search engine versions
- **Live Search**: Execute searches in real-time and compare results
- **Build Integration**: Build any version directly from the dashboard
- **Performance Metrics**: Track and visualize detailed performance metrics
- **Execution Control**: Configure and run different versions with custom parameters
- **Comparative Analysis**: Run benchmarks to compare scaling, performance, and memory usage

## Prerequisites

- Python 3.6 or higher
- Flask and Flask-CORS packages
- Modern web browser (Chrome, Firefox, Safari, or Edge)
- Compiled binaries for all three search engine versions

## Getting Started

1. Make sure all three versions of the search engine are built:
   ```
   cd "../Serial Version" && make
   cd "../OpenMP Version" && make
   cd "../MPI Version" && make
   ```

2. Install Python requirements:
   ```
   pip3 install flask flask-cors
   ```

3. Start the dashboard:
   ```
   chmod +x start_dashboard.sh
   ./start_dashboard.sh
   ```

4. Open your browser and navigate to `http://localhost:5001` if not automatically opened

## Dashboard Sections

### 1. Main Dashboard

- Overview of performance metrics for all versions
- Quick comparison charts
- System status and recent searches

### 2. Search Engine

- Execute searches with any version
- Configure thread count, process count, and other parameters
- View and compare search results

### 3. Performance Metrics

- Detailed performance charts for all versions
- Memory usage tracking
- CPU utilization statistics
- Query processing breakdown

### 4. Version Comparison

- Side-by-side comparison of all versions
- Scaling efficiency charts
- Memory usage analysis
- Result accuracy verification

### 5. Build & Deploy

- Build any version from the dashboard
- Configure build parameters
- View build logs
- MPI deployment configuration

### 6. Settings

- Configure default parameters
- Set dataset paths
- Adjust BM25 ranking parameters

## API Endpoints

The dashboard includes a backend API with the following endpoints:

- **GET /api/status**: Check the status of all search engine versions
- **POST /api/search**: Execute a search with specified parameters
- **GET /api/metrics**: Retrieve all recorded performance metrics
- **POST /api/build**: Build a specific version
- **POST /api/compare**: Run a comparison between versions

## Architecture

The dashboard consists of:

1. **Frontend**: HTML, CSS, JavaScript with Bootstrap and Chart.js
2. **Backend**: Python Flask API that interfaces with the search engine executables
3. **Bridge**: API layer that translates web requests into command-line operations

## Troubleshooting

- If the search engine binaries are not found, you'll be prompted to build them
- Check the console logs for any API errors
- Ensure all required Python packages are installed
- For MPI execution issues, verify your MPI installation (OpenMPI or MPICH)

## License

This dashboard is released under the same license as the search engine project.
