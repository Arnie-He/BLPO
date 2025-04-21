#!/bin/bash

#SBATCH --output=slurm_logs/wandb_%j.out # Standard output log
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
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

###### algo = {nystrom, nested, cg, ppo}
###### task = {cartpole, acrobot}, {$task, humanoid, humanoidstandup, inverted_pendulum, inverted_double_pendulum, pusher, hopper, reacher}

task="acrobot" 
group_ver="April21"

python main.py --task=$task --algo=nystrom --group_ver=$group_ver
python main.py --task=$task --algo=ppo --group_ver=$group_ver
python main.py --task=$task --algo=nested --group_ver=$group_ver
python main.py --task=$task --algo=cg --group_ver=$group_ver