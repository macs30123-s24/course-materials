#!/bin/bash
#SBATCH --job-name=gpu      # job name
#SBATCH --output=info.out   # output log file
#SBATCH --error=info.err    # error file
#SBATCH --time=00:05:00     # 5 minutes of wall time
#SBATCH --nodes=1           # 1 GPU node
#SBATCH --partition=gpu     # GPU partition
#SBATCH --ntasks=1          # 1 CPU core to drive GPU
#SBATCH --gres=gpu:1        # Request 1 GPU
#SBATCH --account=macs30123

module load python cuda

python3 ./print_gpu_info.py
