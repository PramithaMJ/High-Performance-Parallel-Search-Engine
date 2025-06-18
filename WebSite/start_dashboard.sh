#!/bin/bash

# start_dashboard.sh - Script to start the search engine dashboard

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Python 3 could not be found. Please install Python 3 to run the dashboard."
    exit 1
fi

# Check for required Python packages
echo "Checking for required Python packages..."
REQUIRED_PACKAGES=("flask" "flask-cors")
MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! python3 -c "import $package" &> /dev/null; then
        MISSING_PACKAGES+=("$package")
    fi
done

# Install missing packages if needed
if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo "Installing required packages: ${MISSING_PACKAGES[*]}"
    pip3 install "${MISSING_PACKAGES[@]}"
fi

# Check if any binaries need to be built
if [ ! -f "$BASE_DIR/Serial Version/bin/search_engine" ] || \
   [ ! -f "$BASE_DIR/OpenMP Version/bin/search_engine" ] || \
   [ ! -f "$BASE_DIR/MPI Version/bin/search_engine" ]; then
    echo "Some search engine binaries are missing. Would you like to build them now? (y/n)"
    read -r build_choice
    if [[ "$build_choice" =~ ^[Yy]$ ]]; then
        # Build all versions
        echo "Building Serial version..."
        (cd "$BASE_DIR/Serial Version" && make clean && make)
        
        echo "Building OpenMP version..."
        (cd "$BASE_DIR/OpenMP Version" && make clean && make)
        
        echo "Building MPI version..."
        (cd "$BASE_DIR/MPI Version" && make clean && make)
    else
        echo "Warning: Some features may not work without the compiled binaries."
    fi
fi

# Create data directory if it doesn't exist
mkdir -p "$BASE_DIR/data"

# Start the API server
echo "Starting Search Engine Dashboard API server on port 5001..."
cd "$SCRIPT_DIR"
python3 "api.py" --debug &
API_PID=$!

# Open the dashboard in the browser (wait a few seconds for server to start)
sleep 3
echo "Opening dashboard in your default browser..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    open "http://localhost:5001"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    xdg-open "http://localhost:5001"
elif [[ "$OSTYPE" == "msys"* ]]; then
    start "http://localhost:5001"
else
    echo "Please open http://localhost:5001 in your browser to access the dashboard."
fi

echo "Dashboard is running. Press Ctrl+C to stop..."

# Handle cleanup on exit
function cleanup {
    echo "Stopping API server..."
    kill $API_PID
    exit 0
}

trap cleanup INT

# Wait for user to press Ctrl+C
while true; do
    sleep 1
done
