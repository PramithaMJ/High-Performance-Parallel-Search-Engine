#!/bin/bash

# MPI Search Engine Runner Script
# This script helps run the MPI version with the correct mpirun command

# Default values
NUM_PROCESSES=4
ARGS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -np|--num-processes)
      NUM_PROCESSES="$2"
      shift # past argument
      shift # past value
      ;;
    *)
      ARGS="$ARGS $1"
      shift # past argument
      ;;
  esac
done

# Check if binary exists
if [ ! -f "./bin/search_engine" ]; then
    echo "Error: ./bin/search_engine not found. Please compile first with 'make'"
    exit 1
fi

# Check if mpirun is available
if ! command -v mpirun &> /dev/null; then
    echo "Error: mpirun not found. Please install MPI (e.g., brew install open-mpi)"
    exit 1
fi

echo "Running MPI Search Engine with $NUM_PROCESSES processes..."
echo "Command: mpirun -np $NUM_PROCESSES ./bin/search_engine$ARGS"
echo "----------------------------------------"

# Run the MPI application
mpirun -np $NUM_PROCESSES ./bin/search_engine$ARGS
