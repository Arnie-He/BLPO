#!/bin/bash

# run.sh
export PYTHONPATH=$(pwd):$PYTHONPATH

env_name="breakout"

python main.py --task=$env_name --nested=3 --step=5e6
python main.py --task=$env_name --vanilla=True --nested=3 --step=5e6
python main.py --task=$env_name --algo=ppo --step=5e6