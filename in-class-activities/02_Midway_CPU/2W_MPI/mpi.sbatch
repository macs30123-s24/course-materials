#!/bin/bash

#SBATCH --job-name=mpi
#SBATCH --output=mpi.out
#SBATCH --error=mpi.err
#SBATCH --account=macs30123
#SBATCH --partition=caslake
#SBATCH --ntasks=4

# Load Python and MPI modules
module load python mpich

# Run the python program with mpirun. The -n flag is not required;
# mpirun will automatically figure out the best configuration from the
# Slurm environment variables.
mpirun python3 ./mpi_rand_walk.py
