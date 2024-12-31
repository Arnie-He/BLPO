#!/bin/bash

# run.sh
export PYTHONPATH=$(pwd):$PYTHONPATH

seeds=(0 10 20 30 40)

# @param ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
for seed in "${seeds[@]}"; do
    env_name="walker2d"
    echo "Running task=$env_name with seed=$seed"
    # Nystrom1
    python main_brax.py --task=$env_name --nested=2 --steps=8e6 --seed=$seed
    # Nested
    python main_brax.py --task=$env_name --nested=2 --steps=8e6 --vanilla=True --seed=$seed
    # Nystrom2
    python main_brax.py --task=$env_name --nested=2 --steps=8e6 --algo=test --seed=$seed
    # CG
    python main_brax.py --task=$env_name --nested=2 --steps=8e6 --algo=cg --seed=$seed
    # Vanilla
    python main_brax.py --task=$env_name --steps=8e6 --algo=ppo --seed=$seed
done


for seed in "${seeds[@]}"; do
    env_name="halfcheetah"
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