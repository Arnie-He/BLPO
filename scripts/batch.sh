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
source venv10/bin/activate

python main_brax.py --task=walker2d --algo=nystrom
python main_brax.py --task=walker2d --algo=ppo