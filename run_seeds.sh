#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:a5500:1
#SBATCH --mem=100G

#SBATCH -t 24:00:00
#SBATCH --output=slurm_logs/wandb_%j.out  # Standard output log

# Load Python module
module load python/3.11
export PYTHONPATH=$(pwd):$PYTHONPATH

# Get the API key for WandB
API_KEY=$(cat wandbkey.txt)

# Activate the Python virtual environment
source venv10/bin/activate

# Log in to WandB
wandb login $API_KEY

# Read the environment variable passed as an argument
ENV=$1

# Check if the environment is provided
if [ -z "$ENV" ]; then
  echo "Error: Please provide the environment as the first argument."
  echo "Usage: sbatch script_name.sh <env>"
  exit 1
fi

# Define the seeds
SEEDS=(42 123 456 789 1011)

for SEED in "${SEEDS[@]}"; do
echo "Running for seed=$SEED, and environment=$ENV"
# python Stackelberg_RL/run_brax.py --env "$ENV" --seed "$SEED" --vanilla --nested_updates 10

python Stackelberg_RL/run_brax.py --env "$ENV" --seed "$SEED" --vanilla --nested_updates 1

#python Stackelberg_RL/run_brax.py --env "$ENV" --seed "$SEED" --nested_updates 1

done


# for SEED in "${SEEDS[@]}"; do
#     echo "Running for seed=$SEED, and environment=$ENV"
#     python Stackelberg_RL/run_brax.py --env "$ENV" --seed "$SEED"
# done

