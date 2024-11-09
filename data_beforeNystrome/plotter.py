from typing import Callable, List, Optional, Tuple
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt

def rolling_window(array: np.ndarray, window: int) -> np.ndarray:
    """
    Apply a rolling window to a 1D numpy array.

    :param array: the input array
    :param window: length of the rolling window
    :return: a 2D array where each row is a window of the input array
    """
    if window > array.size:
        raise ValueError("Window size cannot be larger than the array size.")
    shape = (array.size - window + 1, window)
    strides = (array.strides[0], array.strides[0])
    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)


def window_func(var_1: np.ndarray, var_2: np.ndarray, window: int, func: Callable) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a function to the rolling window of 2 arrays.

    :param var_1: variable 1 (e.g., steps)
    :param var_2: variable 2 (e.g., values)
    :param window: length of the rolling window
    :param func: function to apply on the rolling window on variable 2
    :return: the rolled variable 1 and the function applied to variable 2
    """
    if len(var_2) < window:
        return var_1, var_2  # Not enough data for rolling window
    var_2_window = rolling_window(var_2, window)
    function_on_var2 = func(var_2_window, axis=-1)
    return var_1[window - 1:], function_on_var2

def ts2xy(data_frame: pd.DataFrame, x_axis: str, y_axis_column: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose a data frame variable to x and y arrays.

    :param data_frame: the input data frame
    :param x_axis: the column name to use for x-axis data
    :param y_axis_column: the column name to use for y-axis data
    :return: the x and y arrays
    """
    x_var = data_frame[x_axis].values
    y_var = data_frame[y_axis_column].values
    return x_var, y_var

def plot_curves(
    output: str,
    xy_list: List[Tuple[np.ndarray, np.ndarray]],
    labels: List[str],
    x_axis_label: str,
    y_axis_label: str,
    title: str,
    figsize: Tuple[int, int] = (8, 6),
    rolling_window_size: Optional[int] = None,
    plot_every_n: int = 1,
    naive_cosine: bool = False,
) -> None:
    """
    Plot the curves.

    :param output: the output file path to save the plot
    :param xy_list: the x and y coordinates to plot
    :param labels: list of labels for each run
    :param x_axis_label: label for the x-axis
    :param y_axis_label: label for the y-axis
    :param title: the title of the plot
    :param figsize: Size of the figure (width, height)
    :param rolling_window_size: window size for rolling mean and variance.
    :param plot_every_n: plot every nth data point
    :param naive_cosine: if True and y_axis_label is 'cosine_similarities', plot only dots, no lines, no rolling window
    """
    plt.figure(figsize=figsize)
    for idx, (x, y) in enumerate(xy_list):
        label = labels[idx]

        # Downsample data by plotting every nth point
        x = x[::plot_every_n]
        y = y[::plot_every_n]

        if y_axis_label == 'cosine_similarities' and naive_cosine:
            # For 'cosine_similarities' with naive_cosine True, plot only dots without lines or rolling window
            plt.scatter(x, y, label=label)
        else:
            # For 'cosine_similarities' with naive_cosine False, plot absolute value
            if y_axis_label == 'cosine_similarities':
                y = np.abs(y)

            if rolling_window_size and len(y) >= rolling_window_size:
                x_smooth, y_mean = window_func(x, y, rolling_window_size, np.mean)
                x_smooth, y_std = window_func(x, y, rolling_window_size, np.std)
                plt.plot(x_smooth, y_mean, label=label)
                plt.fill_between(x_smooth, y_mean - y_std, y_mean + y_std, alpha=0.2)
            else:
                plt.plot(x, y, label=label)
    plt.title(title)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    plt.close()  # Close the figure to free up memory
    print(f"Plot saved as {output}")

def plot_results(
    output: str,
    files: List[str],
    x_axis: str,
    y_axis_column: str,
    title: str,
    figsize: Tuple[int, int] = (8, 6),
    rolling_window_size: Optional[int] = None,
    plot_every_n: int = 1,
    naive_cosine: bool = False,
) -> None:
    """
    Plot the results from CSV files.

    :param output: the output file path to save the plot
    :param files: list of CSV files to plot
    :param x_axis: the column name to use for x-axis data (e.g., 'step')
    :param y_axis_column: the column name to use for y-axis data (e.g., 'reward')
    :param title: the title of the plot
    :param figsize: Size of the figure (width, height)
    :param rolling_window_size: window size for rolling mean and variance.
    :param plot_every_n: plot every nth data point
    :param naive_cosine: if True and y_axis_column is 'cosine_similarities', plot only dots
    """
    data_frames = []
    labels = []
    for file in files:
        data_frame = pd.read_csv(file)
        data_frames.append(data_frame)
        # Extract label from filename
        label = os.path.splitext(os.path.basename(file))[0]
        labels.append(label)
    xy_list = [ts2xy(data_frame, x_axis, y_axis_column) for data_frame in data_frames]

    # Adjust rolling_window_size based on naive_cosine
    if y_axis_column == 'cosine_similarities' and naive_cosine:
        rolling_window_size = None

    plot_curves(
        output,
        xy_list,
        labels,
        x_axis_label=x_axis,
        y_axis_label=y_axis_column,
        title=title,
        figsize=figsize,
        rolling_window_size=rolling_window_size,
        plot_every_n=plot_every_n,
        naive_cosine=naive_cosine,
    )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot data from CSV files.")
    parser.add_argument('csv_files', nargs='+', help='List of CSV files to plot')
    parser.add_argument('--x-axis', type=str, default='step', help='Column name for x-axis (default: step)')
    parser.add_argument('--y', type=str, default='reward', help='Column name for y-axis (default: reward)')
    parser.add_argument('--title', type=str, default='Plot', help='Title of the plot')
    parser.add_argument('--figsize', type=float, nargs=2, default=[8, 6], help='Figure size (width height)')
    parser.add_argument('--window', type=int, default=4, help='Window size for rolling mean and variance (default: 10)')
    parser.add_argument('--output', type=str, default="data/figure.png", help='Output file path for the plot')
    parser.add_argument('--every-n', type=int, default=10, help='Plot every nth data point (default: 1)')
    parser.add_argument('--naive',type=bool, default=False, help='If set, plot cosine_similarities naively (scatter plot of raw values).')

    args = parser.parse_args()

    plot_results(
        output=args.output,
        files=args.csv_files,
        x_axis=args.x_axis,
        y_axis_column=args.y,
        title=args.title,
        figsize=tuple(args.figsize),
        rolling_window_size=args.window,
        plot_every_n=args.every_n,
        naive_cosine=args.naive,
    )
