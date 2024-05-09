#!/bin/bash

#SBATCH --job-name=ssd-gpu-example
#SBATCH --output=ssd-gpu.out
#SBATCH --error=ssd-gpu.err
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=ssd-gpu
#SBATCH --account=ssd
#SBATCH --qos=ssd

module load python cuda

python3 cupy_rand_walk.py
