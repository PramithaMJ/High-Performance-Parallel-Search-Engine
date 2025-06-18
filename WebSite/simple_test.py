#!/usr/bin/env python3
from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello World! Flask is working!"

if __name__ == '__main__':
    print("Starting simple Flask server at http://localhost:5050")
    app.run(host='0.0.0.0', port=5050, debug=True)
