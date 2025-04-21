#!/bin/bash

# run.sh
export PYTHONPATH=$(pwd):$PYTHONPATH

seeds=(0 10 20 30 40)

# @param ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
for seed in "${seeds[@]}"; do
    env_name="walker2d"
    echo "Running task=$env_name with seed=$seed"
    # Nystromcorrect
    python main_brax.py --task=$env_name --seed=$seed --algo=nys_correct --steps=2e7 --nested=3 --ihvp=0.8 --clipf=0.84 --group=NystromCVer2
done

for seed in "${seeds[@]}"; do
    env_name="hopper"
    echo "Running task=$env_name with seed=$seed"
    # Nystrom1
    python main_brax.py --task=$env_name --seed=$seed --algo=nys_correct --steps=1.2e7 --nested=3 --ihvp=0.8 --clipf=0.84 --group=NystromCVer2
done

for seed in "${seeds[@]}"; do
    env_name="pusher"
    echo "Running task=$env_name with seed=$seed"
    # Nystrom1
    python main_brax.py --task=$env_name --nested=3 --steps=8e6 --algo=nys_correct --seed=$seed --group=NystromCVer2
done

for seed in "${seeds[@]}"; do
    env_name="inverted_double_pendulum"
    echo "Running task=$env_name with seed=$seed"
    # Nystrom1
    python main_brax.py --task=$env_name --nested=3 --steps=8e6 --algo=nys_correct --seed=$seed --group=NystromCVer2
done