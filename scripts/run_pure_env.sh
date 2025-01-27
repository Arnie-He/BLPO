#!/bin/bash

# Set PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH

# Define seeds and experiments
seeds=(0 10 20)
EXP_NAME=("CartPole" "Acrobot")
BEXP_NAME=("walker2d"
           "halfcheetah" 
           "hopper" 
           "humanoid" 
           # "inverted_pendulum" 
           "reacher" 
           # "humanoidstandup" 
           "pusher" 
           "inverted_double_pendulum")

# Loop through seeds
for seed in "${seeds[@]}"; do
    # Run CartPole and Acrobot experiments
    python main.py --algo=env --task=cartpole --steps=5e5 --group=pureenv --seed=$seed
    python main.py --algo=env --task=acrobot --steps=4e5 --group=pureenv --seed=$seed

    # Loop through BEXP_NAME experiments
    for bexp_name in "${BEXP_NAME[@]}"; do
        python main.py --algo=env --task=$bexp_name --steps=8e6 --group=pureenv --seed=$seed
    done
done