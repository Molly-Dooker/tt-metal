import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the CSV file
data = pd.read_csv("memory_statistics_interleaved.csv")

# Define unique combinations of input_memory and output_memory
categories = data[["input_memory", "output_memory"]].drop_duplicates()

# Set up the plot
plt.figure(figsize=(12, 8))

# Loop through each unique category
for _, category in categories.iterrows():
    input_mem = category["input_memory"]
    output_mem = category["output_memory"]

    # Filter the data for the current category
    filtered_data = data[(data["input_memory"] == input_mem) & (data["output_memory"] == output_mem)]

    # Plot mean_duration against num_tiles
    plt.scatter(filtered_data["num_tiles"], filtered_data["mean_duration"], s=100)  # Plot points

    # Linear regression
    x = filtered_data["num_tiles"]
    y = filtered_data["mean_duration"]
    coefficients = np.polyfit(x, y, 1)  # Linear fit (degree 1)
    polynomial = np.poly1d(coefficients)

    # Create a line for the linear approximation
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = polynomial(x_line)

    # Plot the linear approximation
    plt.plot(x_line, y_line, linestyle="--", label=f"input: {input_mem} output: {output_mem}")

# Customize the plot
plt.title("Mean Duration vs. Number of Tiles with Linear Approximation")
plt.xlabel("Number of Tiles")
plt.ylabel("Mean Duration (ns)")
plt.legend(title="Memory Configuration")
plt.grid(True)

# Save the plot as interleaved_stats.png
plt.tight_layout()
plt.savefig("interleaved_stats.png")
