#!/bin/bash

# Simple test to verify MPI process behavior
echo "Testing MPI process behavior..."

cd "/Users/pramithajayasooriya/Desktop/High-Performance-Parallel-Search-Engine/MPI Version"

echo "=== Test 1: Check if mpirun is working ==="
if command -v mpirun &> /dev/null; then
    echo "✓ mpirun found"
    mpirun --version | head -1
else
    echo "✗ mpirun not found"
    exit 1
fi

echo ""
echo "=== Test 2: Simple MPI test ==="
echo 'int main() { return 0; }' > /tmp/test_mpi.c
mpicc /tmp/test_mpi.c -o /tmp/test_mpi
if mpirun -np 2 /tmp/test_mpi; then
    echo "✓ Basic MPI execution works"
else
    echo "✗ Basic MPI execution failed"
fi
rm -f /tmp/test_mpi.c /tmp/test_mpi

echo ""
echo "=== Test 3: Check if binary exists ==="
if [ -f "./bin/search_engine" ]; then
    echo "✓ search_engine binary found"
    ls -la ./bin/search_engine
else
    echo "✗ search_engine binary not found"
    echo "Available files in bin/:"
    ls -la ./bin/ 2>/dev/null || echo "bin/ directory not found"
fi

echo ""
echo "=== Test 4: Check dataset ==="
if [ -d "./dataset" ]; then
    echo "✓ dataset directory found"
    echo "Files in dataset: $(ls ./dataset | wc -l)"
    ls ./dataset | head -3
else
    echo "✗ dataset directory not found"
fi

echo ""
echo "Test completed."
