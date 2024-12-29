#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:a5500:1
#SBATCH --mem=100G

#SBATCH -t 24:00:00
#SBATCH --output=slurm_logs/wandb_%j.out  # Standard output log



module load python/3.11
export PYTHONPATH=$(pwd):$PYTHONPATH

API_KEY=$(cat wandbkey.txt)

# Define the seeds to be used for the experiments

source venv10/bin/activate
wandb login $API_KEY

python Stackelberg_RL/continuous_Hypergrad_ppo.py