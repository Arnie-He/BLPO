#!/bin/bash
#SBATCH -J RBC
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=5:30:00
#SBATCH --mem=64GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -o RBC_job_%j.o
#SBATCH -e RBC_job_%j.e
#SBATCH -o my-output-%j.out

# run.sh
export PYTHONPATH=$(pwd):$PYTHONPATH

seeds=(0 10 20 30 40)

# @param ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
for seed in "${seeds[@]}"; do
    env_name="reacher"
    echo "Running task=$env_name with seed=$seed"
    # Nystrom1
    python main_brax.py --task=$env_name --nested=3 --steps=8e6 --seed=$seed
    # Nested
    python main_brax.py --task=$env_name --nested=3 --steps=8e6 --vanilla=True --seed=$seed
    # Nystrom2
    python main_brax.py --task=$env_name --nested=3 --steps=8e6 --algo=test --seed=$seed
    # CG
    python main_brax.py --task=$env_name --nested=3 --steps=8e6 --algo=cg --seed=$seed
    # Vanilla
    python main_brax.py --task=$env_name --steps=8e6 --algo=ppo --seed=$seed
done


for seed in "${seeds[@]}"; do
    env_name="inverted_pendulum"
    echo "Running task=$env_name with seed=$seed"
    # Nystrom1
    python main_brax.py --task=$env_name --nested=5 --steps=8e6 --seed=$seed
    # Nested
    python main_brax.py --task=$env_name --nested=5 --steps=8e6 --vanilla=True --seed=$seed
    # Nystrom2
    python main_brax.py --task=$env_name --nested=5 --steps=8e6 --algo=test --seed=$seed
    # CG
    python main_brax.py --task=$env_name --nested=5 --steps=8e6 --algo=cg --seed=$seed
    # Vanilla
    python main_brax.py --task=$env_name --steps=8e6 --algo=ppo --seed=$seed
done