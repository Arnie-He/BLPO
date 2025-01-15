#!/bin/bash

# run.sh
export PYTHONPATH=$(pwd):$PYTHONPATH

seeds=(0 10 20 30 40)

# @param ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
for seed in "${seeds[@]}"; do
    env_name="cartpole"
    echo "Running task=$env_name with seed=$seed"
    # Nystrom1
    python main.py --task=$env_name --nested=5 --steps=5e5 --seed=$seed
    # Nested
    python main.py --task=$env_name --nested=5 --steps=5e5 --vanilla=True --seed=$seed
    # CG
    python main.py --task=$env_name --nested=5 --steps=5e5 --algo=cg --seed=$seed
    # Vanilla
    python main.py --task=$env_name --steps=5e5 --algo=ppo --seed=$seed
done


# for seed in "${seeds[@]}"; do
#     env_name="acrobot"
#     echo "Running task=$env_name with seed=$seed"
#     # Nystrom1
#     python main.py --task=$env_name --nested=5 --steps=4e5 --seed=$seed
#     # Nested
#     python main.py --task=$env_name --nested=5 --steps=4e5 --vanilla=True --seed=$seed
#     # CG
#     python main.py --task=$env_name --nested=5 --steps=4e5 --algo=cg --seed=$seed
#     # Vanilla
#     python main.py --task=$env_name --steps=4e5 --algo=ppo --seed=$seed
# done