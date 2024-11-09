#!/bin/bash

# run_experiments.sh

# Define the list of algorithms
algorithms=("actor_critic" "stac_actor")

# Define tasks
tasks=(
    "cartpole"
    "catch"
    "acrobot"
    # Add other tasks as needed
)

# Hyperparameters to sweep
actor_learning_rates=(0.001 0.0025 0.003 0.005)
critic_learning_rates=(0.005 0.008 0.01)
nested_updates_list=(10 25 50)
advantage_rates=(0.9 0.95 0.99)
nystrom_ranks=(20 50 80 100 150)
nystrom_rhos=(10 50 80 100)

# Seeds to use
seeds=(0 1 2 3 4)

# Loop over each task
for task in "${tasks[@]}"; do
    echo "Processing task: $task"

    # Run each algorithm for the current task
    for algo in "${algorithms[@]}"; do
        echo "Running algorithm: $algo on task: $task"

        # Loop over hyperparameter combinations
        for actor_lr in "${actor_learning_rates[@]}"; do
            for critic_lr in "${critic_learning_rates[@]}"; do
                for nested_updates in "${nested_updates_list[@]}"; do
                    for adv_rate in "${advantage_rates[@]}"; do
                        for nystrom_rank in "${nystrom_ranks[@]}"; do
                            for nystrom_rho in "${nystrom_rhos[@]}"; do
                                for seed in "${seeds[@]}"; do

                                    # Construct a unique description for this combination
                                    description="seed${seed}_actorlr${actor_lr}_criticlr${critic_lr}_nested${nested_updates}_adv${adv_rate}_nystromrank${nystrom_rank}_nystromrho${nystrom_rho}"

                                    # Run the experiment
                                    python main.py \
                                        --algo="$algo" \
                                        --task="$task" \
                                        --actor_learning_rate="$actor_lr" \
                                        --critic_learning_rate="$critic_lr" \
                                        --nested_updates="$nested_updates" \
                                        --advantage_rate="$adv_rate" \
                                        --nystrom_rank="$nystrom_rank" \
                                        --nystrom_rho="$nystrom_rho" \
                                        --seed="$seed"

                                done
                            done
                        done
                    done
                done
            done
        done

        echo "Completed runs for algorithm: $algo on task: $task. Configuration: $description"
        echo "-----------------------------------"
    done
done
