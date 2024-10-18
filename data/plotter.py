import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_rewards_with_variance(csv_files, output_file, smoothing_window=100, downsample=10):
    # Set up the plot
    plt.figure(figsize=(10, 6))
    
    # Get a list of colors from the 'tab10' colormap
    colors = plt.get_cmap('tab10').colors
    line_styles = ['-', '--', '-.', ':']  # Different line styles

    # Loop over each CSV file and its index
    for i, csv_file in enumerate(csv_files):
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Extract steps, rewards, and calculate statistics
        steps = df['step']
        rewards = df['reward']
        smoothed_rewards = rewards.rolling(window=smoothing_window).mean()
        smoothed_std = rewards.rolling(window=smoothing_window).std()  # Standard deviation for variance
        
        # Downsample to reduce clutter (plot every nth point)
        downsampled_steps = steps[::downsample]
        downsampled_rewards = smoothed_rewards[::downsample]
        downsampled_std = smoothed_std[::downsample]
        
        # Assign a color and line style based on the index, cycling if necessary
        color = colors[i % len(colors)]
        line_style = line_styles[i % len(line_styles)]
        
        # Plot rewards as a solid line
        plt.plot(downsampled_steps, downsampled_rewards, label=f'Rewards from {os.path.basename(csv_file)}', 
                 color=color, linestyle=line_style)
        
        # Plot the variance as a shaded region
        plt.fill_between(downsampled_steps, downsampled_rewards - downsampled_std, downsampled_rewards + downsampled_std, 
                         color=color, alpha=0.2)

    # Add labels, title, and legend
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.title('Rewards Comparison with Variance from Multiple CSV Files')
    plt.legend()

    # Save the plot to the specified output file
    plt.savefig(output_file)
    print(f"Plot saved as {output_file}")

if __name__ == "__main__":
    # Argument parser to take CSV file paths as input
    parser = argparse.ArgumentParser(description="Plot rewards with variance from multiple CSV files for comparison.")
    parser.add_argument('csv_files', nargs='+', help='List of CSV files to compare')
    parser.add_argument('--output', type=str, default='data/reward_plot_with_variance.png', help='Output file for the plot (default: data/reward_plot_with_variance.png)')
    parser.add_argument('--smoothing_window', type=int, default=100, help='Window size for smoothing the rewards (default: 100)')
    parser.add_argument('--downsample', type=int, default=10, help='Plot every nth point to reduce clutter (default: 10)')
    
    args = parser.parse_args()

    # Ensure the output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Call the plotting function with the list of CSV files
    plot_rewards_with_variance(args.csv_files, args.output, args.smoothing_window, args.downsample)
