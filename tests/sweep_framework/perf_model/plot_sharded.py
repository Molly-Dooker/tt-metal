import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the CSV file
data_sharded = pd.read_csv("memory_statistics_sharded.csv")

# Create a figure for the subplots
plt.figure(figsize=(18, 6))


# Define a function to plot data for each type of sharded memory
def plot_sharded_data(ax, data, title, input_memory):
    filtered_data = data[data["input_memory"] == input_memory]

    ax.scatter(filtered_data["num_tiles"], filtered_data["mean_duration"], s=100, label=f"Samples for {input_memory}")

    # Fit a linear regression line
    x = filtered_data["num_tiles"]
    y = filtered_data["mean_duration"]
    coefficients = np.polyfit(x, y, 1)
    polynomial = np.poly1d(coefficients)

    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = polynomial(x_line)

    ax.plot(x_line, y_line, linestyle="--", label=f"{input_memory} Line")

    ax.set_title(title)
    ax.set_xlabel("Number of Tiles")
    ax.set_ylabel("Mean Duration (ns)")
    ax.grid(True)
    ax.legend(title="Memory Configuration", loc="upper left")


# Plot for L1_BLOCK_SHARDED
plt.subplot(1, 3, 1)
plot_sharded_data(plt.gca(), data_sharded, "L1 Block Sharded Memory Statistics", "L1_BLOCK_SHARDED")

# Plot for L1_HEIGHT_SHARDED
plt.subplot(1, 3, 2)
plot_sharded_data(plt.gca(), data_sharded, "L1 Height Sharded Memory Statistics", "L1_HEIGHT_SHARDED")

# Plot for L1_WIDTH_SHARDED
plt.subplot(1, 3, 3)
plot_sharded_data(plt.gca(), data_sharded, "L1 Width Sharded Memory Statistics", "L1_WIDTH_SHARDED")

# Adjust layout
plt.tight_layout()

# Save the plot as sharded_stats.png
plt.savefig("sharded_stats.png")
plt.show()
