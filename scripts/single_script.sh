#!/bin/bash

# Slurm sbatch options
#SBATCH -o logs/batch_script.sh.log-%j

#SBATCH --gres=gpu:volta:1
#SBATCH -n 20 

# Loading the required module
module load anaconda/Python-ML-2023b

# Run the script
python segment_utils.py
