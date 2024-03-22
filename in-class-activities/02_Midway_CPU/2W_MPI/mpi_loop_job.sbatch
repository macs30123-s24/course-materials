#!/bin/bash

#SBATCH --job-name=mpi_loop_job
#SBATCH --error=mpi_loop.err
#SBATCH --output=mpi_loop.out
#SBATCH --account=macs30123
#SBATCH --partition=caslake
#SBATCH --ntasks=5
#SBATCH --nodes=1 # Run ntasks on same single node

# Load Python and MPI modules
module load python mpich

for i in {1..5}
  do
    mpirun -n $i python3 mpi_rand_walk.py >> mpi_loop.out
  done
