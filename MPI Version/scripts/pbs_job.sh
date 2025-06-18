#!/bin/bash
#PBS -N search_engine_job
#PBS -l nodes=3:ppn=4
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -o search_engine_job.log

# Change to the submission directory
cd $PBS_O_WORKDIR

# Load MPI module (may be different on your cluster)
module load openmpi

# Run the application
mpirun -np $PBS_NP ./bin/search_engine -m @lpramithamj
