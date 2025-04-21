#!/bin/bash

#SBATCH --output=slurm_logs/wandb_%j.out # Standard output log
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=8:30:00
#SBATCH --mem=64GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
  

# Load Python module

module load python/3.11
export PYTHONPATH=$(pwd):$PYTHONPATH

# Get the API key for WandB
API_KEY=$(cat wandbkey.txt)
# Activate the Python virtual environment
source venv10/bin/activate

# run.sh
export PYTHONPATH=$(pwd):$PYTHONPATH

seeds=(0 10 20 30 40)

# @param ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
for seed in "${seeds[@]}"; do
    env_name="humanoid"
    echo "Running task=$env_name with seed=$seed"
    # Vanilla
    python main_brax.py --task=$env_name --seed=$seed --algo=ppo --steps=2e7
    # Nested
    python main_brax.py --task=$env_name --seed=$seed --algo=nested --steps=2e7 --nested=5 
    # CG
    python main_brax.py --task=$env_name  --seed=$seed --algo=cg --steps=2e7 --nested=5 --ihvp=1.0 --clipf=0.8
    # Nystrom1
    python main_brax.py --task=$env_name --seed=$seed --algo=nystrom --steps=2e7 --nested=5 --ihvp=1.0 --clipf=0.8
done


for seed in "${seeds[@]}"; do
    env_name="halfcheetah"
    echo "Running task=$env_name with seed=$seed"
    # Vanilla
    python main_brax.py --task=$env_name --seed=$seed --algo=ppo --steps=2e7
    # Nested
    python main_brax.py --task=$env_name --seed=$seed --algo=nested --steps=2e7 --nested=7 
    # CG
    python main_brax.py --task=$env_name  --seed=$seed --algo=cg --steps=2e7 --nested=7 --ihvp=1.4 --clipf=0.7
    # Nystrom1
    python main_brax.py --task=$env_name --seed=$seed --algo=nystrom --steps=2e7 --nested=7 --ihvp=1.4 --clipf=0.7
done