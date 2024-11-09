#!/bin/bash

# run_experiments.sh

# Define the list of algorithms
algorithms=("ratliff" "stac-actor" "actor_critic" "a2c_no_nest")

# Define tasks
tasks=(
    # "cartpole" 
    # "pendulum" 
    # "acrobot"
    "catch"
    # Add other tasks as needed
)

# Loop over each task
for task in "${tasks[@]}"; do
    echo "Processing task: $task"

    # Run each algorithm for the current task
    for algo in "${algorithms[@]}"; do
        echo "Running algorithm: $algo on task: $task"
        
        if [[ "$algo" == "actor_critic" || "$algo" == "a2c_no_nest" ]]; then
            python main.py --algo="$algo" --task="$task"
        else
            for lam in 0.0 10.0 100.0 1000.0 100000.0; do
                python main.py --algo="$algo" --task="$task" --description="lambda${lam}" --lam="$lam"
            done
        fi
    done

    # Prepare data file names based on the algorithm and task
    data_files=()
    for algo in "${algorithms[@]}"; do
        if [[ "$algo" == "actor_critic" ]]; then
            data_files+=("data/${task}/${algo}_nested.csv")
        elif [[ "$algo" == "a2c_no_nest" ]]; then
            data_files+=("data/${task}/${algo}_nonest.csv")
        else
            for lam in 0.0 10.0 100.0 1000.0 100000.0; do
                data_files+=("data/${task}/${algo}_lambda${lam}.csv")
            done
        fi
    done

    # Convert the data_files array to a space-separated string
    data_files_str="${data_files[@]}"

    # Run the plotter and save the plot for rewards
    output_file="data/${task}/reward.png"
    echo "Generating reward plot for task: $task"
    python data/plotter.py $data_files_str --output="$output_file"
    echo "Plot saved as $output_file"
    echo "-----------------------------------"

    # Now, for each algorithm (excluding 'actor_critic' and 'a2c_no_nest'), plot 'cosine_similarities' and 'final_product'
    for algo in "${algorithms[@]}"; do
        if [[ "$algo" == "actor_critic" || "$algo" == "a2c_no_nest" ]]; then
            continue  # Skip these algorithms
        fi

        echo "Generating additional plots for algorithm: $algo on task: $task"

        # Collect data files for different lambda values for this algorithm
        algo_data_files=()
        for lam in 0.0 10.0 100.0 1000.0 100000.0; do
            algo_data_files+=("data/${task}/${algo}_lambda${lam}.csv")
        done
        algo_data_files_str="${algo_data_files[@]}"

        # Plot 'cosine_similarities'
        output_file="data/${task}/${algo}_cosine_similarities.png"
        echo "Plotting cosine_similarities for $algo on $task"
        python data/plotter.py $algo_data_files_str --output="$output_file" --column=cosine_similarities
        echo "Plot saved as $output_file"

        # Plot 'final_product'
        output_file="data/${task}/${algo}_final_product.png"
        echo "Plotting final_product for $algo on $task"
        python data/plotter.py $algo_data_files_str --output="$output_file" --column=final_product
        echo "Plot saved as $output_file"
        echo "-----------------------------------"
    done
done
