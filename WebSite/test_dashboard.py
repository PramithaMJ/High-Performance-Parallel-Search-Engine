#!/usr/bin/env python3
# Test script for the search engine dashboard

import os
import sys
import json
import requests
import unittest

PORT = 5001
BASE_URL = f"http://localhost:{PORT}"

class TestDashboard(unittest.TestCase):
    """Test cases for the search engine dashboard API"""
    
    def test_status(self):
        """Test that the API status endpoint returns correctly"""
        resp = requests.get(f"{BASE_URL}/api/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("versions", data)
        
    def test_search(self):
        """Test that the search API endpoint works"""
        payload = {
            "version": "serial",
            "query": "microservices",
            "options": {
                "limit": 5
            }
        }
        resp = requests.post(f"{BASE_URL}/api/search", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        if "error" in data:
            print(f"Search API returned error: {data['error']}")
        else:
            print(f"Search result count: {data.get('result_count', 0)}")
            print(f"Execution time: {data.get('execution_time_ms', 0)} ms")
    
    def test_web_crawl(self):
        """Test that the web crawl functionality works"""
        payload = {
            "version": "serial", 
            "query": "welcome",
            "options": {
                "dataSource": "crawl",
                "crawlUrl": "https://example.com",
                "crawlDepth": 1,
                "crawlMaxPages": 2,
                "limit": 5
            }
        }
        resp = requests.post(f"{BASE_URL}/api/search", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        if "error" in data:
            print(f"Crawl API returned error: {data['error']}")
        else:
            print(f"Crawl result count: {data.get('result_count', 0)}")
            print(f"Execution time: {data.get('execution_time_ms', 0)} ms")
    
    def test_compare(self):
        """Test that the compare API endpoint works"""
        payload = {
            "querySet": "simple",
            "comparisonType": "performance"
        }
        resp = requests.post(f"{BASE_URL}/api/compare", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        if "serial" not in data or "openmp" not in data or "mpi" not in data:
            print("Compare API missing expected version data")
        else:
            print(f"Serial avg time: {data['serial'].get('avg_time', 0)} ms")
            print(f"OpenMP avg time: {data['openmp'].get('avg_time', 0)} ms")
            print(f"MPI avg time: {data['mpi'].get('avg_time', 0)} ms")
            
            if "speedup" in data["openmp"]:
                print(f"OpenMP speedup: {data['openmp']['speedup']}x")
            if "speedup" in data["mpi"]:
                print(f"MPI speedup: {data['mpi']['speedup']}x")

if __name__ == "__main__":
    print(f"Testing dashboard API at {BASE_URL}")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
