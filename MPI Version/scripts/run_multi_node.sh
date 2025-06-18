#!/bin/bash

# Get the number of nodes and processes per node from command line or use default
NODES=${1:-3}          # Default: 3 nodes
PROCS_PER_NODE=${2:-4} # Default: 4 processes per node
TOTAL_PROCS=$((NODES * PROCS_PER_NODE))

# Parameters for the search engine
PARAMS="${@:3}"        # All remaining parameters are passed to the application

echo "Running search engine on $NODES nodes with $PROCS_PER_NODE processes per node (total: $TOTAL_PROCS processes)"

# Compile the MPI version
echo "Compiling MPI version..."
make clean
make

# Check if compilation was successful
if [ $? -ne 0 ]; then
    echo "Compilation failed. Please check errors."
    exit 1
fi

# Run with MPI across multiple nodes
mpirun --hostfile hostfile -np $TOTAL_PROCS ./bin/search_engine $PARAMS

echo "Job completed."
