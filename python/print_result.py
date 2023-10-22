import matplotlib.pyplot as plt
import numpy as np

def plot_tensor(tensor):
    if tensor.shape != (1, tensor.shape[1], 3):
        raise ValueError(f"The input tensor of size: {tensor.shape} does not have the required shape (1, n, 3)")

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 12))

    # Define intervals. Here, for example, every 100 points:
    interval_points = 100
    x_ticks = np.arange(0, tensor.shape[1], interval_points)
    x_tick_labels = [f"{16*val}ms" for val in x_ticks]

    for i in range(3):
        ax = axes[i]
        ax.plot(np.arange(tensor.shape[1]), tensor[0, :, i], label=f"Plot {i+1}")
        ax.set_ylim(-2, 2)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels, rotation=45)  # Rotate the x-axis labels for better visibility.
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel(f"Value for Plot {i+1}")
        ax.legend()
        ax.grid(True, which='both')  # Display major and minor grid lines
        ax.minorticks_on()  # Turn minor ticks on for finer grid
        ax.grid(which='minor', linestyle=':', linewidth=0.5)  # Set minor grid style

    plt.tight_layout()
    plt.savefig('channels_plot.png')
    plt.show()

# Test the function with a sample tensor.
# tensor = np.random.rand(1, 1000, 3) - 0.5
# plot_tensor(tensor)
