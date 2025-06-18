#!/usr/bin/env python3
"""
Search Monitoring Script - Helps diagnose search engine execution issues

This script runs the search engine with verbose logging to help diagnose
timeout and performance issues.
"""

import os
import sys
import json
import time
import signal
import datetime
import subprocess
import argparse

# Base directory for the search engine project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERSIONS = ["Serial", "OpenMP", "MPI"]

def get_executable_path(version):
    """Get the path to the search engine executable for a specific version"""
    version_dir = version.lower() + " Version"
    return os.path.join(BASE_DIR, version_dir, "bin", "search_engine")

def run_search_with_monitoring(version, query, options=None, timeout=60):
    """Run search engine with detailed monitoring and diagnostics"""
    executable = get_executable_path(version)
    
    if not os.path.exists(executable):
        print(f"⚠️ Error: Executable for {version} version not found at {executable}")
        return 1
    
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
            print(f"🌐 Crawling URL: {options['crawlUrl']}")
            cmd.extend(["-c", options["crawlUrl"]])
            # Add crawl depth and max pages if specified
            if "crawlDepth" in options:
                cmd.extend(["-d", str(options["crawlDepth"])])
            if "crawlMaxPages" in options:
                cmd.extend(["-p", str(options["crawlMaxPages"])])
    
    print(f"🚀 Executing command: {' '.join(cmd)}")
    print(f"⏱️  Timeout set to: {timeout} seconds")
    
    # Start timing
    start_time = time.time()
    
    # Handle SIGALRM for timeout monitoring
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Process timed out after {timeout} seconds")
    
    # Set timeout alarm
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
    try:
        # Run the command with incremental output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1  # Line buffered
        )
        
        # Poll process while capturing output incrementally
        stdout = []
        stderr = []
        
        print("\n--- Output Stream ---")
        while process.poll() is None:
            # Read and display stdout
            for line in process.stdout:
                stdout.append(line)
                print(f"STDOUT: {line.strip()}")
                sys.stdout.flush()
                
            # Read and display stderr
            for line in process.stderr:
                stderr.append(line)
                print(f"STDERR: {line.strip()}")
                sys.stdout.flush()
                
        # Cancel the alarm
        signal.alarm(0)
        
        # End timing
        execution_time = time.time() - start_time
        
        # Display summary
        print(f"\n✅ Process completed in {execution_time:.2f} seconds")
        print(f"Exit code: {process.returncode}")
        
        return 0
        
    except TimeoutError as e:
        print(f"\n⚠️ {e}")
        
        # Try to terminate the process if it's still running
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(5)  # Wait up to 5 seconds for termination
            except subprocess.TimeoutExpired:
                process.kill()  # Force kill if termination takes too long
        
        return 1
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    finally:
        # Ensure the alarm is cancelled
        signal.alarm(0)

def main():
    parser = argparse.ArgumentParser(description='Search Engine Monitor for debugging timeouts')
    parser.add_argument('-v', '--version', choices=['serial', 'openmp', 'mpi'], default='serial', help='Search engine version')
    parser.add_argument('-q', '--query', required=True, help='Search query')
    parser.add_argument('-t', '--timeout', type=int, default=60, help='Timeout in seconds')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads (OpenMP)')
    parser.add_argument('--processes', type=int, default=4, help='Number of processes (MPI)')
    parser.add_argument('-l', '--limit', type=int, default=10, help='Result limit')
    parser.add_argument('-c', '--crawl', help='Web crawling URL')
    parser.add_argument('-d', '--depth', type=int, default=2, help='Web crawling depth')
    parser.add_argument('-p', '--max-pages', type=int, default=10, help='Max pages to crawl')
    
    args = parser.parse_args()
    
    # Prepare options
    options = {
        'threads': args.threads,
        'processes': args.processes,
        'limit': args.limit,
    }
    
    # Add crawl parameters if provided
    if args.crawl:
        options['dataSource'] = 'crawl'
        options['crawlUrl'] = args.crawl
        options['crawlDepth'] = args.depth
        options['crawlMaxPages'] = args.max_pages
    
    # Run search with monitoring
    sys.exit(run_search_with_monitoring(args.version, args.query, options, args.timeout))

if __name__ == '__main__':
    main()
