#!/usr/bin/env python3
# Simple test script to verify port 5001 is working

from flask import Flask, send_file
import os

app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello World! Server is running on port 5001'

@app.route('/test.html')
def test_html():
    return send_file('test.html')

if __name__ == '__main__':
    print("Starting test server on http://localhost:5001")
    print("Press Ctrl+C to stop the server")
    app.run(host='0.0.0.0', port=5001, debug=True)
