#!/bin/bash
#SBATCH --job-name=pyopencl_activity
#SBATCH --output=pyopencl_activity.out
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --account=macs30123

module load python cuda

python3 pyopencl_activity.py
