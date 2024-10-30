# Define the list of algorithms
algorithms=("ratliff" "stac-actor" "actor_critic" "a2c_no_nest")

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

task="cartpole"

echo "Processing task: $task"

# Prepare data file names based on the algorithm and task
data_files=()

data_files+=("data/${task}/actor_critic_nested.csv")
for algo in "${algorithms[@]}"; do
    if [[ "$algo" == "actor_critic" ]]; then
        data_files+=("data/${task}/${algo}_nested.csv")
    elif [[ "$algo" == "a2c_no_nest" ]]; then
        #data_files+=("data/${task}/${algo}_nonest.csv")
        :
    else
        for lam in 0.0 10.0 100.0 10000.0 1000000.0; do
            data_files+=("data/${task}/${algo}_lambda${lam}.csv")
        done
        # for lam in 0.0; do
        #     data_files+=("data/${task}/${algo}_lambda${lam}.csv")
        # done
    fi
done

algo="stac-actor"
# for lam in 0.0 10.0 100.0 10000.0 1000000.0; do
#     data_files+=("data/${task}/${algo}_lambda${lam}.csv")
# done

# Convert the data_files array to a space-separated string
data_files_str="${data_files[@]}"

# Run the plotter and save the plot
output_file="data/${task}/${algo}_lambdasweep.png"
echo "Generating plot for task: $task"
python data/plotter.py $data_files_str --output="$output_file"
echo "Plot saved as $output_file"
echo "-----------------------------------"