#!/bin/bash

# run.sh
export PYTHONPATH=$(pwd):$PYTHONPATH

seeds=(0 10 20 30 40)

# @param ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
for seed in "${seeds[@]}"; do
    env_name="cartpole"
    echo "Running task=$env_name with seed=$seed"
    # Nystrom1
    python main_ac.py --task=$env_name --algo=nystrom --nested=10 --steps=5e5 --seed=$seed --ihvp=0.2 --clipf=0.2 --group=NystromAC2
    # Nested
    # python main_ac.py --task=$env_name --algo=nested --nested=10 --steps=5e5 --seed=$seed --group=NestedAC2
done


for seed in "${seeds[@]}"; do
    env_name="acrobot"
    echo "Running task=$env_name with seed=$seed"
    # Nystrom1
    python main_ac.py --task=$env_name --algo=nystrom --nested=10 --steps=5e5 --seed=$seed --ihvp=0.2 --clipf=0.2 --group=NystromAC2
    # Nested
    # python main_ac.py --task=$env_name --algo=nested --nested=10 --steps=5e5 --seed=$seed --group=NestedAC2
done