#!/bin/bash

# run.sh
export PYTHONPATH=$(pwd):$PYTHONPATH

seeds=(0 10 20 30 40)
env_names=("walker2d" "halfcheetah" "hopper" "humanoid")

# @param ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
for env_name in "${env_names[@]}"; do
    for seed in "${seeds[@]}"; do
        # Vanilla
        python main_brax.py --task=$env_name --steps=8e6 --algo=ppo --seed=$seed
    done
done