#!/bin/bash

#SBATCH --job-name=ssd-cpu-example
#SBATCH --output=ssd-cpu.out
#SBATCH --error=ssd-cpu.err
#SBATCH --ntasks=10
#SBATCH --partition=ssd
#SBATCH --account=ssd

module load python mpich

mpirun python3 mpi_rand_walk.py
