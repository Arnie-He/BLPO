import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import itertools

def plot_rewards(csv_files, output_file):
    # Generate a cycle of colors for multiple files
    colors = itertools.cycle(plt.get_cmap('tab10').colors)

    plt.figure(figsize=(10, 6))
    
    # Loop over each CSV file
    for csv_file in csv_files:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Extract steps and rewards
        steps = df['step']
        rewards = df['reward']
        
        # Plot rewards with a different color for each CSV file
        color = next(colors)
        plt.plot(steps, rewards, label=f'Rewards from {csv_file}', color=color)

    # Add labels and title
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.title('Rewards Comparison from Multiple CSV Files')
    plt.legend()

    # Show the plot
    plt.savefig(output_file)
    print(f"Plot saved as {output_file}")

if __name__ == "__main__":
    # Argument parser to take CSV file paths as input
    parser = argparse.ArgumentParser(description="Plot rewards from multiple CSV files for comparison.")
    parser.add_argument('csv_files', nargs='+', help='List of CSV files to compare')
    parser.add_argument('--output', type=str, default='data/reward_plot.png', help='Output file for the plot (default: reward_plot.png)')
    
    args = parser.parse_args()

    # Ensure the output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Call the plotting function with the list of CSV files
    plot_rewards(args.csv_files, args.output)
