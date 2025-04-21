#!/bin/bash

# run.sh
export PYTHONPATH=$(pwd):$PYTHONPATH

seeds=(0 10 20 30 40)

# @param ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
for seed in "${seeds[@]}"; do
    env_name="walker2d"
    echo "Running task=$env_name with seed=$seed"
    # Vanilla
    python main_brax.py --task=$env_name --seed=$seed --algo=ppo --steps=2e7
    # Nested
    python main_brax.py --task=$env_name --seed=$seed --algo=nested --steps=2e7 --nested=3 
    # CG
    python main_brax.py --task=$env_name  --seed=$seed --algo=cg --steps=2e7 --nested=3 --ihvp=0.8 --clipf=0.84
    # Nystrom1
    python main_brax.py --task=$env_name --seed=$seed --algo=nystrom --steps=2e7 --nested=3 --ihvp=0.8 --clipf=0.84
done


for seed in "${seeds[@]}"; do
    env_name="hopper"
    echo "Running task=$env_name with seed=$seed"
    # Vanilla
    python main_brax.py --task=$env_name --seed=$seed --algo=ppo --steps=1.2e7
    # Nested
    python main_brax.py --task=$env_name --seed=$seed --algo=nested --steps=1.2e7 --nested=3 
    # CG
    python main_brax.py --task=$env_name  --seed=$seed --algo=cg --steps=1.2e7 --nested=3 --ihvp=0.8 --clipf=0.84
    # Nystrom1
    python main_brax.py --task=$env_name --seed=$seed --algo=nystrom --steps=1.2e7 --nested=3 --ihvp=0.8 --clipf=0.84
done