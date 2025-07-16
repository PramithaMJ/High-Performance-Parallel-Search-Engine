#!/bin/bash
# Start the High-Performance Parallel Search Engine Dashboard
# Usage: ./start_dashboard.sh [port]

PORT=${1:-8080}  # Changed default from 5000 to 8080 to avoid AirPlay conflict on macOS
BASE_DIR="$(dirname "$(realpath "$0")")"
PYTHON_CMD=$(command -v python3 || command -v python)

echo "Starting High-Performance Parallel Search Engine Dashboard..."
echo "Base directory: $BASE_DIR"

# Check if Python is available
if [ -z "$PYTHON_CMD" ]; then
    echo "Error: Python 3 is not installed or not in PATH."
    exit 1
fi

# Check if Flask is installed
if ! $PYTHON_CMD -c "import flask" &>/dev/null; then
    echo "Flask is not installed. Installing required packages..."
    $PYTHON_CMD -m pip install flask flask-cors || { echo "Failed to install requirements"; exit 1; }
fi

# Create config.ini if it doesn't exist
CONFIG_FILE="$BASE_DIR/config.ini"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Creating default config.ini..."
    cat > "$CONFIG_FILE" << EOL
# Website Dashboard Configuration

# Server settings
PORT=$PORT
DEBUG=True

# Search engine paths
SERIAL_PATH=../Serial\ Version/bin/search_engine
OPENMP_PATH=../OpenMP\ Version/bin/search_engine
MPI_PATH=../MPI\ Version/bin/search_engine
HYBRID_PATH=../Hybrid\ Version/bin/search_engine

# MPI settings
MPI_HOSTFILE=../MPI\ Version/hostfile
MPI_NUM_PROCESSES=4

# Default parameters
DEFAULT_MAX_RESULTS=10
DEFAULT_TIMEOUT=30
EOL
    echo "Created default config.ini"
fi

# Make sure the data directory exists
mkdir -p "$BASE_DIR/../data"

echo "Starting server on port $PORT..."
echo "Note: Using port 8080 by default to avoid conflicts with AirPlay Receiver on macOS"
cd "$BASE_DIR" && $PYTHON_CMD -m flask --app api run --host=0.0.0.0 --port=$PORT
