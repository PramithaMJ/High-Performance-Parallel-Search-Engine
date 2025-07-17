#!/bin/bash
# filepath: /Users/pramithajayasooriya/Library/CloudStorage/OneDrive-Personal/Academic/Academic/Semester 7/HPC/hpc/HPC Project/High-Performance-Parallel-Search-Engine/Hybrid Version/scripts/run_hybrid.sh
#
# Helper script to run the search engine with both MPI processes and OpenMP threads
# specified on the command line.
#
# Usage: ./run_hybrid.sh <MPI_PROCS> <OMP_THREADS> [other arguments]
# Example: ./run_hybrid.sh 4 8 -q "artificial intelligence"
#
# The script now uses positional parameters for MPI processes and OpenMP threads
# First parameter: Number of MPI processes
# Second parameter: Number of OpenMP threads
# All remaining parameters are passed directly to the search engine

# Default values
MPI_PROCS=4
OMP_THREADS=4
OTHER_ARGS=""

# Check if we have at least two arguments
if [[ $# -ge 2 ]]; then
    MPI_PROCS=$1
    OMP_THREADS=$2
    shift 2
    OTHER_ARGS="$@"
else
    # If not enough arguments, use defaults
    if [[ $# -eq 1 ]]; then
        # If just one argument, assume it's the MPI process count
        MPI_PROCS=$1
        shift
    fi
    # Add -np and -t options to OTHER_ARGS so they're passed to the search engine
    OTHER_ARGS="-np $MPI_PROCS -t $OMP_THREADS $@"
fi

# Print configuration
echo " Running hybrid MPI+OpenMP search engine with:"
echo "   - MPI Processes: $MPI_PROCS (distributed memory parallelism)"
echo "   - OpenMP Threads per process: $OMP_THREADS (shared memory parallelism)"
echo "   - Total parallel units: $(($MPI_PROCS * $OMP_THREADS))"
echo "   - Additional arguments: $OTHER_ARGS"
echo ""

# Set OpenMP threads environment variable
export OMP_NUM_THREADS=$OMP_THREADS

# Run the search engine
# Note: -np flag is passed to mpirun, not as an argument to the search_engine
mpirun -np $MPI_PROCS ../bin/search_engine -t $OMP_THREADS $OTHER_ARGS
