import matplotlib.pyplot as plt

class ChartLogger:
    """A logger that saves metrics during training and plots them as charts."""
    def __init__(self, metric_names):
        self.interval = 1
        self.data = { name: [] for name in metric_names }
        self.titles = {}
        self.save_paths = {}

    def set_interval(self, interval):
        """Sets the step interval. This method should only be called once before logging."""
        self.interval = interval

    def set_info(self, name, title = None, save_path = None):
        """Sets metadata about a metric."""
        if title is not None:
            self.titles[name] = title
        if save_path is not None:
            self.save_paths[name] = save_path

    def log_metrics(self, metrics):
        """Logs values for metrics by name. The values are stored internally as floats."""
        for name in metrics:
            if name in self.data:
                self.data[name] += [float(v) for v in metrics[name]]

    def plot_metric(self, name):
        """Plots a metric as a line chart at a file path."""
        assert name in self.data
        assert name in self.titles
        assert name in self.save_paths
        values = self.data[name]
        step_counts = list(s * self.interval for s in range(1, len(values) + 1))

        figure, axes = plt.subplots()
        axes.plot(step_counts, values)
        axes.set_title(self.titles[name])
        figure.savefig(self.save_paths[name])