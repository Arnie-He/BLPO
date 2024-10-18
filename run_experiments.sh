#!/bin/bash

# run_experiments.sh

# Define the list of algorithms
algorithms=("actor_critic" "ratliff" "stac-actor" "a2c_no_nest")

# Combine tasks
tasks=(
    "cartpole" 
    # "pendulum" 
    "acrobot"

    # "4rooms"
    # "pong"

    "catch"
    # "deepsea"
    # "bandit"

    # "space_invaders"
    # "breakout"
    # "asterix"
)

# Loop over each task
for task in "${tasks[@]}"; do
    echo "Processing task: $task"

    # Run each algorithm for the current task
    for algo in "${algorithms[@]}"; do
        echo "Running algorithm: $algo on task: $task"

        python main.py --algo="$algo" --task="$task"
    done

    # Prepare data file names based on the algorithm and task
    data_files=()
    for algo in "${algorithms[@]}"; do
        if [ "$algo" == "actor_critic" ]; then
            data_file="data/${task}/${algo}_nested.csv"
        elif [ "$algo" == "a2c_no_nest" ]; then
            data_file="data/${task}/${algo}_nonest.csv"
        else
            data_file="data/${task}/${algo}_lambda10e5.csv"
        fi
        data_files+=("$data_file")
    done

    # Convert the data_files array to a space-separated string
    data_files_str="${data_files[@]}"

    # Run the plotter and save the plot
    output_file="data/${task}/reward.png"
    echo "Generating plot for task: $task"
    python data/plotter.py $data_files_str --output="$output_file"
    echo "Plot saved as $output_file"
    echo "-----------------------------------"
done
