#!/bin/bash

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