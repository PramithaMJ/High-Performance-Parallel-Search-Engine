#!/usr/bin/env python3
# api.py - Bridge between web interface and search engine executables

import os
import sys
import json
import time
import datetime
import subprocess
import argparse
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Base directory for the search engine project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERSIONS = ["Serial", "OpenMP", "MPI", "Hybrid"]
METRICS_FILE = os.path.join(BASE_DIR, "data", "performance_metrics.json")

# Initialize metrics storage if it doesn't exist
def init_metrics():
    if not os.path.exists(METRICS_FILE):
        data = {
            "runs": [],
            "latest": {
                "serial": {},
                "openmp": {},
                "mpi": {},
                "hybrid": {}
            }
        }
        os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
        with open(METRICS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        return data
    else:
        with open(METRICS_FILE, 'r') as f:
            return json.load(f)

# Save metrics to file
def save_metrics(data):
    with open(METRICS_FILE, 'w') as f:
        json.dump(data, f, indent=2)

# Record metrics from a search run
def record_metrics(version, metrics_data):
    data = init_metrics()
    
    # Update latest metrics for this version
    data["latest"][version] = metrics_data
    
    # Add to historical runs
    run_data = {
        "version": version,
        "timestamp": metrics_data["timestamp"],
        "query": metrics_data["query"],
        "metrics": metrics_data["metrics"],
        "result_count": metrics_data["result_count"]
    }
    
    data["runs"].append(run_data)
    
    # Keep only last 100 runs to avoid file growth
    if len(data["runs"]) > 100:
        data["runs"] = data["runs"][-100:]
    
    save_metrics(data)

# Get the path to the search engine executable for a specific version
def get_executable_path(version):
    if version.lower() == "hybrid":
        # Special case for Hybrid version
        version_dir = "Hybrid Version"
    else:
        version_dir = version.capitalize() + " Version"
    
    return os.path.join(BASE_DIR, version_dir, "bin", "search_engine")

# Run the search engine command and capture output
def run_search_engine(version, query, options=None):
    executable = get_executable_path(version)
    version_dir = f"{version.capitalize()} Version" if version.lower() != "hybrid" else "Hybrid Version"
    work_dir = os.path.join(BASE_DIR, version_dir)
    
    if not os.path.exists(executable):
        return {
            "error": f"Executable for {version} version not found at {executable}"
        }
    
    # Default values if options is None
    if options is None:
        options = {}
    
    # Prepare parameters with defaults
    crawl_url = options.get("crawlUrl", "https://medium.com/@lpramithamj")
    crawl_depth = str(options.get("crawlDepth", 2))
    crawl_max_pages = str(options.get("crawlMaxPages", 10))
    thread_count = str(options.get("threads", 6))
    num_processes = str(options.get("processes", 4))
    
    # Prepare command based on version
    cmd = []
    
    if version.lower() == "serial":
        # Serial version: ./bin/search_engine -c https://medium.com/@lpramithamj -d 2 -p 10
        cmd = [executable, "-c", crawl_url, "-d", crawl_depth, "-p", crawl_max_pages]
    elif version.lower() == "openmp":
        # OpenMP version: ./bin/search_engine -t 6 -c https://medium.com/@lpramithamj -d 3 -p 20
        cmd = [executable, "-t", thread_count, "-c", crawl_url, "-d", crawl_depth, "-p", crawl_max_pages]
    elif version.lower() == "mpi":
        # MPI version: mpirun -np 4 ./bin/search_engine -m @lpramithamj -d 2 -p 10
        cmd = ["mpirun", "-np", num_processes, executable, "-m", "@lpramithamj", "-d", crawl_depth, "-p", crawl_max_pages]
    elif version.lower() == "hybrid":
        # Hybrid version: mpirun -np 8 ./bin/search_engine -t 8 -m @lpramithamj -d 2 -p 10
        hybrid_processes = str(options.get("processes", 8))
        hybrid_threads = str(options.get("threads", 8))
        cmd = ["mpirun", "-np", hybrid_processes, executable, "-t", hybrid_threads, "-m", "@lpramithamj", "-d", crawl_depth, "-p", crawl_max_pages]
    
    # Add query if provided
    if query and query.strip():
        cmd.extend(["-q", query])
    
    # Add result limit if specified
    if "limit" in options:
        cmd.extend(["-l", str(options["limit"])])
    
    # Additional options
    if options:
        # Add result limit if specified
        if "limit" in options:
            cmd.extend(["-l", str(options["limit"])])
        
        # Add data source if specified
        if options.get("dataSource") == "custom" and "dataPath" in options:
            cmd.extend(["-i", options["dataPath"]])
        
        # Add crawl URL if specified
        if options.get("dataSource") == "crawl" and "crawlUrl" in options:
            print(f"Crawling URL: {options['crawlUrl']}")
            cmd.extend(["-c", options["crawlUrl"]])
            # Add crawl depth and max pages if specified
            if "crawlDepth" in options:
                cmd.extend(["-d", str(options["crawlDepth"])])
            if "crawlMaxPages" in options:
                cmd.extend(["-p", str(options["crawlMaxPages"])])
    
    # Start timing
    start_time = time.time()
    
    # Determine timeout based on options
    search_timeout = 120  # Default timeout
    if options and options.get('extendedTimeout'):
        search_timeout = 240  # Extended timeout for complex searches (4 minutes)
        print(f"Using extended timeout of {search_timeout} seconds for search: {query}")
    
    try:
        # Run the command with appropriate timeout
        process = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=search_timeout
        )
        
        # End timing
        execution_time = time.time() - start_time
        
        # Extract results from output
        results = []
        lines = process.stdout.split("\n")
        
        # Parse output for results and metrics
        metrics = {
            "query_time_ms": 0,
            "indexing_time_ms": 0,
            "memory_usage_mb": 0,
            "total_time_ms": execution_time * 1000
        }
        
        current_result = None
        for line in lines:
            # Check for metrics
            if "Query processed in" in line:
                try:
                    metrics["query_time_ms"] = float(line.split("in")[1].split("ms")[0].strip())
                except:
                    pass
            elif "Indexing time:" in line:
                try:
                    metrics["indexing_time_ms"] = float(line.split(":")[1].split("ms")[0].strip())
                except:
                    pass
            elif "Memory usage:" in line:
                try:
                    metrics["memory_usage_mb"] = float(line.split(":")[1].split("MB")[0].strip())
                except:
                    pass
            
            # Check for result entries
            if line.startswith("Document:"):
                if current_result:
                    results.append(current_result)
                current_result = {"title": line.replace("Document:", "").strip(), "path": "", "score": 0, "snippet": ""}
            elif line.startswith("Path:") and current_result:
                current_result["path"] = line.replace("Path:", "").strip()
            elif line.startswith("Score:") and current_result:
                try:
                    current_result["score"] = float(line.replace("Score:", "").strip())
                except:
                    current_result["score"] = 0
            elif line.startswith("Snippet:") and current_result:
                current_result["snippet"] = line.replace("Snippet:", "").strip()
        
        # Add the last result if there is one
        if current_result:
            results.append(current_result)
        
        # Record metrics
        record_metrics(version.lower(), {
            "query": query,
            "timestamp": datetime.datetime.now().isoformat(),
            "metrics": metrics,
            "result_count": len(results)
        })
        
        return {
            "success": True,
            "version": version,
            "query": query,
            "execution_time_ms": execution_time * 1000,
            "metrics": metrics,
            "result_count": len(results),
            "results": results
        }
    except subprocess.TimeoutExpired:
        error_message = f"Process timed out after {search_timeout} seconds"
        # Provide more detailed error information for timeouts
        if options and options.get('dataSource') == 'crawl':
            crawl_url = options.get('crawlUrl', 'unknown URL')
            crawl_depth = options.get('crawlDepth', 'unknown depth')
            error_message += f" while crawling {crawl_url} with depth {crawl_depth}"
        
        print(f"SEARCH ERROR: {error_message} for query: {query}")
        return {
            "error": error_message,
            "query": query,
            "version": version,
            "recommendation": "Try using a more specific query or reducing crawl depth"
        }
    except Exception as e:
        return {
            "error": str(e)
        }

# Build a specific version of the search engine
def build_version(version):
    if version.lower() == "hybrid":
        # Special case for Hybrid version
        version_dir = "Hybrid Version"
    else:
        version_dir = version.capitalize() + " Version"
    
    build_dir = os.path.join(BASE_DIR, version_dir)
    
    if not os.path.exists(build_dir):
        return {
            "error": f"Directory for {version} version not found at {build_dir}"
        }
    
    try:
        # Run make in the version directory
        process = subprocess.run(
            ["make", "-j4"],  # Use 4 threads for faster building
            text=True,
            capture_output=True,
            timeout=60,  # Give it a minute to build
            cwd=build_dir  # Run in the version directory
        )
        
        if process.returncode == 0:
            return {
                "success": True,
                "version": version,
                "output": process.stdout,
                "executable": os.path.join(build_dir, "bin", "search_engine")
            }
        else:
            return {
                "error": f"Build failed for {version} version",
                "output": process.stdout,
                "error_output": process.stderr
            }
    except subprocess.TimeoutExpired:
        return {
            "error": f"Build process timed out for {version} version"
        }
    except Exception as e:
        return {
            "error": str(e)
        }

# Get status of all versions
def get_versions_status():
    status = {}
    
    for version in VERSIONS:
        executable = get_executable_path(version)
        status[version.lower()] = {
            "name": version,
            "available": os.path.exists(executable),
            "executable": executable
        }
        
        # Get version-specific status
        if version.lower() == "mpi":
            # Check if MPI is available
            try:
                process = subprocess.run(
                    ["mpirun", "--version"],
                    text=True,
                    capture_output=True,
                    timeout=5
                )
                status[version.lower()]["mpi_available"] = (process.returncode == 0)
                status[version.lower()]["mpi_version"] = process.stdout.split("\n")[0] if process.returncode == 0 else "Unknown"
            except:
                status[version.lower()]["mpi_available"] = False
                status[version.lower()]["mpi_version"] = "Not found"
        
        if version.lower() == "openmp" or version.lower() == "hybrid":
            # Check OpenMP availability by running a simple test
            if os.path.exists(executable):
                try:
                    process = subprocess.run(
                        [executable, "--openmp-info"],
                        text=True,
                        capture_output=True,
                        timeout=5
                    )
                    status[version.lower()]["openmp_available"] = "OpenMP available" in process.stdout
                except:
                    status[version.lower()]["openmp_available"] = False
    
    return status

# API Routes

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('.', path)

@app.route('/api/status', methods=['GET'])
def api_status():
    try:
        return jsonify({
            "status": "ok",
            "versions": get_versions_status(),
            "api_version": "1.0"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/api/search', methods=['POST'])
def api_search():
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data or 'query' not in data or 'version' not in data:
            return jsonify({
                "status": "error",
                "error": "Missing required fields: query and version"
            }), 400
        
        query = data['query']
        version = data['version'].lower()
        options = data.get('options', {})
        
        # Validate version
        if version not in [v.lower() for v in VERSIONS]:
            return jsonify({
                "status": "error",
                "error": f"Invalid version: {version}. Must be one of: {', '.join(VERSIONS)}"
            }), 400
        
        # Run the search
        result = run_search_engine(version, query, options)
        
        if 'error' in result:
            return jsonify({
                "status": "error",
                **result
            }), 500
        else:
            return jsonify({
                "status": "ok",
                **result
            })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/api/metrics', methods=['GET'])
def api_metrics():
    try:
        data = init_metrics()
        return jsonify({
            "status": "ok",
            "metrics": data
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/api/build', methods=['POST'])
def api_build():
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data or 'version' not in data:
            return jsonify({
                "status": "error",
                "error": "Missing required field: version"
            }), 400
        
        version = data['version'].lower()
        
        # Validate version
        if version not in [v.lower() for v in VERSIONS]:
            return jsonify({
                "status": "error",
                "error": f"Invalid version: {version}. Must be one of: {', '.join(VERSIONS)}"
            }), 400
        
        # Build the version
        result = build_version(version)
        
        if 'error' in result:
            return jsonify({
                "status": "error",
                **result
            }), 500
        else:
            return jsonify({
                "status": "ok",
                **result
            })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/api/compare', methods=['POST'])
def api_compare():
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data or 'query' not in data or 'versions' not in data:
            return jsonify({
                "status": "error",
                "error": "Missing required fields: query and versions"
            }), 400
        
        query = data['query']
        versions = data['versions']
        options = data.get('options', {})
        
        # Validate versions
        for version in versions:
            if version.lower() not in [v.lower() for v in VERSIONS]:
                return jsonify({
                    "status": "error",
                    "error": f"Invalid version: {version}. Must be one of: {', '.join(VERSIONS)}"
                }), 400
        
        # Run searches for all versions
        results = {}
        for version in versions:
            results[version] = run_search_engine(version, query, options)
        
        return jsonify({
            "status": "ok",
            "query": query,
            "results": results
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

# Main entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='High-Performance Parallel Search Engine Dashboard API')
    parser.add_argument('--port', type=int, default=5001, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()
    
    # Initialize metrics file if it doesn't exist
    init_metrics()
    
    # Print server information
    print(f"Starting High-Performance Search Engine Dashboard API on port {args.port}")
    print(f"Base directory: {BASE_DIR}")
    print(f"Available versions: {', '.join(VERSIONS)}")
    
    # Run the Flask server
    app.run(host='0.0.0.0', port=args.port, debug=args.debug)
