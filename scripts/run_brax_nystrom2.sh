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

for seed in "${seeds[@]}"; do
    env_name="walker2d"
    echo "Running task=$env_name with seed=$seed"
    # CG
    python main_brax.py --task=$env_name  --seed=$seed --algo=nystrom2 --steps=2e7 --nested=3 --ihvp=0.8 --clipf=0.84 --group=Nystrom2 --lam=0.02
done

for seed in "${seeds[@]}"; do
    env_name="humanoid"
    echo "Running task=$env_name with seed=$seed"

    # CG
    python main_brax.py --task=$env_name  --seed=$seed --algo=nystrom2 --steps=2e7 --nested=3 --ihvp=0.8 --clipf=0.84 --group=Nystrom2 --lam=0.02
done

for seed in "${seeds[@]}"; do
    env_name="pusher"
    echo "Running task=$env_name with seed=$seed"
    #   CG
    python main_brax.py --task=$env_name --nested=3 --steps=8e6 --algo=nystrom2 --seed=$seed --group=Nystrom2 --lam=0.02

done