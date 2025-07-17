#!/bin/bash

# Enhanced MPI Distributed Monitoring Script
# This script helps monitor distributed execution across machines

echo " Starting Enhanced Distributed MPI Search Engine"
echo "════════════════════════════════════════════════════════════════════"

# Check if MPI is available
if ! command -v mpirun &> /dev/null; then
    echo "❌ Error: mpirun not found. Please install MPI (Open MPI or MPICH)"
    exit 1
fi

# Default values
NUM_PROCESSES=2
HOSTFILE=""
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -np|--processes)
            NUM_PROCESSES="$2"
            shift 2
            ;;
        -hostfile|--hostfile)
            HOSTFILE="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -np, --processes NUM     Number of MPI processes (default: 2)"
            echo "  -hostfile, --hostfile    Path to MPI hostfile for multi-machine execution"
            echo "  -v, --verbose           Enable verbose output"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 -np 4                           # Run with 4 processes on local machine"
            echo "  $0 -np 8 -hostfile hosts          # Run with 8 processes across machines in hostfile"
            echo "  $0 -np 4 -v                       # Run with verbose monitoring"
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Validate number of processes
if ! [[ "$NUM_PROCESSES" =~ ^[0-9]+$ ]] || [ "$NUM_PROCESSES" -lt 1 ]; then
    echo "❌ Error: Number of processes must be a positive integer"
    exit 1
fi

# Check if hostfile exists (if specified)
if [[ -n "$HOSTFILE" && ! -f "$HOSTFILE" ]]; then
    echo "❌ Error: Hostfile '$HOSTFILE' not found"
    exit 1
fi

# Check if the search engine binary exists
if [[ ! -f "bin/search_engine" ]]; then
    echo "⚠️  Binary not found. Building the search engine..."
    if ! make; then
        echo "❌ Error: Failed to build the search engine"
        exit 1
    fi
fi

echo " Configuration:"
echo "   ├─ Processes: $NUM_PROCESSES"
if [[ -n "$HOSTFILE" ]]; then
    echo "   ├─ Hostfile: $HOSTFILE"
    echo "   ├─ Machines in hostfile:"
    cat "$HOSTFILE" | grep -v '^#' | grep -v '^$' | sed 's/^/   │  /'
else
    echo "   ├─ Mode: Single machine (localhost)"
fi
echo "   └─ Verbose: $VERBOSE"
echo ""

# Prepare MPI command
MPI_CMD="mpirun -np $NUM_PROCESSES"

if [[ -n "$HOSTFILE" ]]; then
    MPI_CMD="$MPI_CMD -hostfile $HOSTFILE"
fi

if [[ "$VERBOSE" == true ]]; then
    MPI_CMD="$MPI_CMD -verbose"
fi

# Add the executable
MPI_CMD="$MPI_CMD ./bin/search_engine"

echo " Executing distributed search engine..."
echo "Command: $MPI_CMD"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Set environment variables for better MPI output
export OMPI_MCA_btl_vader_single_copy_mechanism=none 2>/dev/null || true
export OMPI_MCA_rmaps_base_oversubscribe=1 2>/dev/null || true

# Execute the MPI command
if [[ "$VERBOSE" == true ]]; then
    echo "📝 Starting with verbose monitoring..."
    $MPI_CMD 2>&1 | while IFS= read -r line; do
        echo "$(date '+%H:%M:%S') | $line"
    done
else
    $MPI_CMD
fi

# Check exit status
EXIT_CODE=$?
echo ""
echo "════════════════════════════════════════════════════════════════════"
if [ $EXIT_CODE -eq 0 ]; then
    echo " Distributed execution completed successfully!"
else
    echo "❌ Distributed execution failed with exit code: $EXIT_CODE"
fi

echo "🏁 Monitoring session ended"
