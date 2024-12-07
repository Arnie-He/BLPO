#!/bin/bash

# run.sh
export PYTHONPATH=$(pwd):$PYTHONPATH

python main.py --task=cartpole
python main.py --task=cartpole --vanilla=True
python main.py --task=cartpole --algo=ppo