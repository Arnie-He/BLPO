#!/bin/bash
#SBATCH -J RBC
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=8:30:00
#SBATCH --mem=64GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -o RBC_job_%j.o
#SBATCH -e RBC_job_%j.e

# Print the current LD_LIBRARY_PATH for debugging
echo "Original LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Unset LD_LIBRARY_PATH (if needed)
unset LD_LIBRARY_PATH
echo "Unset LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Navigate to your home directory
cd ~

# Change to the HyperGradient-RL directory
cd Documents/HyperGradient-RL

# Activate the jax.venv virtual environment
source jax.venv/bin/activate

# Run the Python scripts
python sweep.py