#!/bin/bash

# run.sh
export PYTHONPATH=$(pwd):$PYTHONPATH

seeds=(0 10 20 30 40)

# @param ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
for seed in "${seeds[@]}"; do
    env_name="cartpole"
    echo "Running task=$env_name with seed=$seed"
    # Nystrom1
    python main.py --task=$env_name --algo=nystrom --nested=10 --steps=5e5 --seed=$seed --ihvp=1.4 --clipf=1.0
    # Nested
    python main.py --task=$env_name --algo=nested --nested=10 --steps=5e5 --seed=$seed
    # CG
    python main.py --task=$env_name --algo=cg --nested=10 --steps=5e5 --seed=$seed --ihvp=1.4 --clipf=1.0
    # Vanilla
    python main.py --task=$env_name --algo=ppo --steps=5e5 --seed=$seed
done


for seed in "${seeds[@]}"; do
    env_name="acrobot"
    echo "Running task=$env_name with seed=$seed"
    # Nystrom1
    python main.py --task=$env_name --algo=nystrom --nested=10 --steps=5e5 --seed=$seed --ihvp=2 --clipf=1.16662 
    # Nested
    python main.py --task=$env_name --algo=nested --nested=10 --steps=5e5 --seed=$seed
    # CG
    python main.py --task=$env_name --algo=cg --nested=10 --steps=5e5 --seed=$seed --ihvp=2 --clipf=1.16662 
    # Vanilla
    python main.py --task=$env_name --algo=ppo --steps=5e5 --seed=$seed
done