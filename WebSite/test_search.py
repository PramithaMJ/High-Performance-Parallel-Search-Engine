#!/usr/bin/env python3
"""
Test script for the search functionality in the dashboard.
This script will send search requests to the API and verify the responses.
"""

import os
import sys
import json
import time
import requests
import argparse

def test_search(query, version="serial", options=None):
    """
    Send a search request to the API and process the response
    """
    if options is None:
        options = {}
    
    url = "http://localhost:5001/api/search"
    payload = {
        "version": version,
        "query": query,
        "options": options
    }
    
    print(f"\n[TEST] Searching for '{query}' using {version.upper()} version")
    print(f"[TEST] Options: {json.dumps(options)}")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        response = requests.post(url, json=payload, timeout=180)
        
        # Check if successful
        if response.status_code == 200:
            result = response.json()
            
            if "error" in result:
                print(f"❌ Error: {result['error']}")
                return False
                
            execution_time = time.time() - start_time
            print(f"✅ Search completed in {execution_time:.2f} seconds")
            print(f"   API reported execution time: {result.get('execution_time_ms', 0)/1000:.2f} seconds")
            print(f"   Result count: {result.get('result_count', 0)}")
            
            # Display results
            if "results" in result and result["results"]:
                print("\nTop results:")
                for i, res in enumerate(result["results"][:3], 1):  # Show top 3
                    print(f"{i}. {res.get('title', 'Untitled')} (Score: {res.get('score', 0):.2f})")
                    print(f"   Path: {res.get('path', 'No path')}")
                    print(f"   Snippet: {res.get('snippet', 'No snippet')[:100]}...")
            else:
                print("\nNo results found.")
                
            return True
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(response.text)
            return False
            
    except requests.exceptions.Timeout:
        print(f"❌ Request timed out after {time.time() - start_time:.2f} seconds")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ Connection error. Make sure the API server is running.")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def run_test_suite():
    """Run a series of tests to verify search functionality"""
    print("=" * 80)
    print("SEARCH FUNCTIONALITY TEST SUITE")
    print("=" * 80)
    
    # Test 1: Simple search with Serial version
    test_search("python programming", "serial")
    
    # Test 2: Search with OpenMP version
    test_search("parallel computing", "openmp", {"threads": 4})
    
    # Test 3: Search with MPI version
    test_search("high performance", "mpi", {"processes": 4})
    
    # Test 4: Search with crawl option (this might timeout)
    crawl_options = {
        "dataSource": "crawl",
        "crawlUrl": "https://medium.com",
        "crawlDepth": 1,
        "crawlMaxPages": 5
    }
    test_search("medium articles", "openmp", crawl_options)
    
    # Test 5: Search with extended timeout
    extended_options = {
        "extendedTimeout": True,
        "threads": 8
    }
    test_search("complex query with many terms and specific phrases", "openmp", extended_options)
    
    print("\nAll tests completed!")

def main():
    parser = argparse.ArgumentParser(description='Test the search functionality')
    parser.add_argument('-q', '--query', help='The search query')
    parser.add_argument('-v', '--version', choices=['serial', 'openmp', 'mpi'], default='serial', 
                        help='The search engine version to use')
    parser.add_argument('-t', '--threads', type=int, default=4, 
                        help='Number of threads for OpenMP version')
    parser.add_argument('-p', '--processes', type=int, default=4, 
                        help='Number of processes for MPI version')
    parser.add_argument('-c', '--crawl', help='URL to crawl')
    parser.add_argument('-d', '--depth', type=int, default=1, 
                        help='Crawl depth')
    parser.add_argument('--suite', action='store_true', 
                        help='Run the full test suite')
    
    args = parser.parse_args()
    
    if args.suite:
        run_test_suite()
    elif args.query:
        # Build options
        options = {
            "threads": args.threads,
            "processes": args.processes
        }
        
        # Add crawl options if specified
        if args.crawl:
            options["dataSource"] = "crawl"
            options["crawlUrl"] = args.crawl
            options["crawlDepth"] = args.depth
        
        # Run the search
        test_search(args.query, args.version, options)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
