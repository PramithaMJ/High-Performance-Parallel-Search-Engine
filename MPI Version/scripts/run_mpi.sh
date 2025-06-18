#!/bin/bash

# Compile the MPI version
echo "Compiling MPI version..."
make clean
make

# Check if compilation was successful
if [ $? -ne 0 ]; then
    echo "Compilation failed. Please check errors."
    exit 1
fi

# Get the number of processes from command line or use default
NUM_PROCESSES=${1:-4}
echo "Running search engine with $NUM_PROCESSES MPI processes..."

# Run with MPI
mpirun -np $NUM_PROCESSES ./bin/search_engine "$@"
