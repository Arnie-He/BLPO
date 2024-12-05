#!/bin/bash

# run.sh
export PYTHONPATH=$(pwd):$PYTHONPATH

python main.py 
python main.py --vanilla=True
python main.py --algo=ppo