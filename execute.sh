#!/bin/bash
#
#SBATCH --partition=a100        # Use GPU partition "a100"
#SBATCH --gres gpu:2            # set 2 GPUs per job
#SBATCH -c 15                   # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH -t 0-01:00              # Maximum run-time in D-HH:MM
#SBATCH --mem=30G               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o logs_output/Unified_Train/Train_%j.out      # File to which STDOUT will be written
#SBATCH -e logs_error/Unified_Train/Train_%j.err      # File to which STDERR will be written

eval "$(conda shell.bash hook)"
conda activate sentinel # run a regular command
python Train_final_stacked.py  # run a regular command
conda deactivate # run a regular command