#!/bin/bash

#SBATCH --output=slurm_logs/wandb_%j.out # Standard output log
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=16:00:00
#SBATCH --mem=64GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Load Python module

module load python/3.11
export PYTHONPATH=$(pwd):$PYTHONPATH
source .venv/bin/activate
###### algo = {nystrom, nested, cg, ppo}
###### task = {cartpole, acrobot}, {walker2d, humanoid, humanoidstandup, inverted_pendulum, inverted_double_pendulum, pusher, hopper, reacher}


# python main.py --task=walker2d --algo=nystrom --group_ver=April20
# python main.py --task=walker2d --algo=ppo --group_ver=April20
# python main.py --task=walker2d --algo=nested --group_ver=April20
# python main.py --task=walker2d --algo=cg --group_ver=April20

python main.py --task=walker2d --algo=nested --group_ver=criticp_April20
# python main.py --task=walker2d --algo=cgcriticp --group_ver=criticp_April20