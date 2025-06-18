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
VERSIONS = ["Serial", "OpenMP", "MPI"]
METRICS_FILE = os.path.join(BASE_DIR, "data", "performance_metrics.json")

# Initialize metrics storage if it doesn't exist
def init_metrics():
    if not os.path.exists(METRICS_FILE):
        data = {
            "runs": [],
            "latest": {
                "serial": {},
                "openmp": {},
                "mpi": {}
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

# Get the path to the search engine executable for a specific version
def get_executable_path(version):
    version_dir = version.lower() + " Version"
    return os.path.join(BASE_DIR, version_dir, "bin", "search_engine")

# Run the search engine command and capture output
def run_search_engine(version, query, options=None):
    executable = get_executable_path(version)
    
    if not os.path.exists(executable):
        return {
            "error": f"Executable for {version} version not found at {executable}"
        }
    
    # Prepare command based on version and options
    cmd = []
    
    if version.lower() == "mpi":
        # For MPI version, use mpirun
        num_processes = options.get("processes", 4) if options else 4
        cmd = ["mpirun", "-np", str(num_processes), executable, "-q", query]
    elif version.lower() == "openmp":
        # For OpenMP version, pass thread count
        cmd = [executable, "-q", query]
        if options and "threads" in options:
            cmd.extend(["-t", str(options["threads"])])
    else:
        # Serial version
        cmd = [executable, "-q", query]
    
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
            cmd.extend(["-c", options["crawlUrl"]])
    
    # Start timing
    start_time = time.time()
    
    try:
        # Run the command
        process = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=30  # 30 second timeout
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
        return {
            "error": f"Process timed out after 30 seconds"
        }
    except Exception as e:
        return {
            "error": str(e)
        }

# Record metrics for a run
def record_metrics(version, data):
    metrics_data = init_metrics()
    metrics_data["latest"][version] = data
    metrics_data["runs"].append({
        "version": version,
        **data
    })
    save_metrics(metrics_data)

# API endpoint to get the status of the search engine
@app.route('/api/status', methods=['GET'])
def get_status():
    status = {
        "versions": {}
    }
    
    for version in VERSIONS:
        executable = get_executable_path(version)
        status["versions"][version.lower()] = {
            "available": os.path.exists(executable),
            "path": executable
        }
    
    return jsonify(status)

# API endpoint to search using a specific version
@app.route('/api/search', methods=['POST'])
def search():
    data = request.json
    version = data.get('version', 'serial')
    query = data.get('query', '')
    options = data.get('options', None)
    
    if not query:
        return jsonify({"error": "No query provided"})
    
    result = run_search_engine(version, query, options)
    return jsonify(result)

# API endpoint to get metrics
@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    metrics_data = init_metrics()
    return jsonify(metrics_data)

# API endpoint to run a build
@app.route('/api/build', methods=['POST'])
def build():
    data = request.json
    version = data.get('version', 'all')
    build_type = data.get('buildType', 'production')
    clean = data.get('clean', False)
    
    versions_to_build = VERSIONS if version == 'all' else [version.capitalize()]
    results = {}
    
    for v in versions_to_build:
        version_dir = v.lower() + " Version"
        version_path = os.path.join(BASE_DIR, version_dir)
        
        if not os.path.exists(version_path):
            results[v.lower()] = {"success": False, "error": f"Version directory not found: {version_path}"}
            continue
        
        cmd = []
        if clean:
            cmd.append(f"cd {version_path} && make clean && ")
        else:
            cmd.append(f"cd {version_path} && ")
        
        if build_type == 'production':
            cmd.append("make production")
        elif build_type == 'debug':
            cmd.append("make debug")
        else:
            cmd.append("make")
        
        try:
            process = subprocess.run(
                " ".join(cmd),
                text=True,
                capture_output=True,
                timeout=120,  # 2 minute timeout
                shell=True
            )
            results[v.lower()] = {
                "success": process.returncode == 0,
                "output": process.stdout,
                "error": process.stderr
            }
        except Exception as e:
            results[v.lower()] = {"success": False, "error": str(e)}
    
    return jsonify(results)

# API endpoint to run a comparison
@app.route('/api/compare', methods=['POST'])
def compare():
    data = request.json
    query_set = data.get('querySet', 'default')
    comparison_type = data.get('comparisonType', 'performance')
    
    # Default query set
    queries = ["circuit breaker", "distributed tracing", "linux wake up", "parallel computing", "microservices"]
    
    if query_set == 'simple':
        queries = ["web", "search", "engine", "performance", "parallel"]
    elif query_set == 'complex':
        queries = ["distributed tracing with zipkin", "high performance computing parallel algorithms", 
                  "circuit breaker pattern microservices architecture", "linux kernel boot process",
                  "machine learning model optimization techniques"]
    
    results = {
        "serial": {"avg_time": 0, "results": []},
        "openmp": {"avg_time": 0, "results": []},
        "mpi": {"avg_time": 0, "results": []}
    }
    
    # Run each query on each version
    for version in ["serial", "openmp", "mpi"]:
        total_time = 0
        for query in queries:
            result = run_search_engine(version, query)
            if "execution_time_ms" in result:
                total_time += result["execution_time_ms"]
                results[version]["results"].append({
                    "query": query,
                    "time_ms": result["execution_time_ms"],
                    "result_count": result.get("result_count", 0),
                    "metrics": result.get("metrics", {})
                })
        results[version]["avg_time"] = total_time / len(queries) if queries else 0
    
    # Calculate speedups
    if results["serial"]["avg_time"] > 0:
        results["openmp"]["speedup"] = results["serial"]["avg_time"] / results["openmp"]["avg_time"]
        results["mpi"]["speedup"] = results["serial"]["avg_time"] / results["mpi"]["avg_time"]
    
    return jsonify(results)

# API endpoint to serve static files from the WebSite directory
@app.route('/<path:path>')
def send_static(path):
    try:
        website_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"Serving static file: {path} from {website_dir}")
        return send_from_directory(website_dir, path)
    except Exception as e:
        print(f"Error serving static file {path}: {str(e)}")
        return f"Error serving file: {str(e)}", 500

# Default route serves the main HTML file
@app.route('/')
def index():
    try:
        website_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"Serving index.html from {website_dir}")
        return send_from_directory(website_dir, 'index.html')
    except Exception as e:
        print(f"Error serving index.html: {str(e)}")
        return f"Error serving file: {str(e)}", 500

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Search Engine API Server')
    parser.add_argument('--port', type=int, default=5001, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()
    
    # Initialize metrics file
    init_metrics()
    
    print(f"Starting server on http://localhost:{args.port}")
    print(f"Press Ctrl+C to stop the server")
    
    # Start the server with explicit host binding to all interfaces
    app.run(host='0.0.0.0', port=args.port, debug=args.debug, threaded=True)
