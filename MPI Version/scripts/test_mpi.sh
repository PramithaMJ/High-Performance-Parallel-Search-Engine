#!/bin/bash
# Test MPI communication
mpirun --hostfile hostfile -np 12 hostname
