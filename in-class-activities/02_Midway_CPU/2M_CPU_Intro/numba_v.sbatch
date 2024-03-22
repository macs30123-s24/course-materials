#!/bin/bash

#SBATCH --job-name=numba_v
#SBATCH --output=numba_v.out
#SBATCH --error=numba_v.err
#SBATCH --account=macs30123
#SBATCH --partition=caslake
#SBATCH --ntasks=1

# Load Python and MPI modules
module load python

# Compile Vectorized Numba function and compare performance to regular NumPy
python3 ./distance_functions.py
