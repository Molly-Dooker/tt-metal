import matplotlib.pyplot as plt
import csv
import numpy as np
import os

# File path to the aggregated results
aggregated_file = "tests/sweep_framework/perf_model/rmsre_aggregated_results.csv"
output_image_path = "tests/sweep_framework/perf_model/error_distribution/rmsre_plots.png"  # Path to save the image


# Function to read and process the aggregated results from CSV
def read_aggregated_data(aggregated_file):
    data = {}
    with open(aggregated_file, mode="r") as infile:
        reader = csv.reader(infile)
        header = next(reader)  # Skip the header row

        for row in reader:
            input_memory = row[0]
            output_memory = row[1]
            input_datatype = row[2]
            num_tiles = int(row[3])
            mean_rmsre = float(row[4])
            std_rmsre = float(row[5])
            std_mean_ratio = float(row[6])

            # Create a key for the (input_memory, output_memory, input_datatype)
            key = (input_memory, output_memory, input_datatype)

            # Initialize dictionary entry if not already present
            if key not in data:
                data[key] = {}

            # Store the results for the given key and num_tiles
            data[key][num_tiles] = (mean_rmsre, std_rmsre, std_mean_ratio)

    return data


# Function to plot the RMSRE with Standard Deviation for all combinations
def plot_rmsre_with_std(data):
    # Define the 12 combinations of input memory, output memory, and input datatype
    plot_order = [
        ("L1_INTERLEAVED", "L1_INTERLEAVED", "BFLOAT16"),
        ("L1_INTERLEAVED", "L1_INTERLEAVED", "BFLOAT4_B"),
        ("L1_INTERLEAVED", "L1_INTERLEAVED", "BFLOAT8_B"),
        ("L1_INTERLEAVED", "DRAM_INTERLEAVED", "BFLOAT16"),
        ("L1_INTERLEAVED", "DRAM_INTERLEAVED", "BFLOAT4_B"),
        ("L1_INTERLEAVED", "DRAM_INTERLEAVED", "BFLOAT8_B"),
        ("DRAM_INTERLEAVED", "L1_INTERLEAVED", "BFLOAT16"),
        ("DRAM_INTERLEAVED", "L1_INTERLEAVED", "BFLOAT4_B"),
        ("DRAM_INTERLEAVED", "L1_INTERLEAVED", "BFLOAT8_B"),
        ("DRAM_INTERLEAVED", "DRAM_INTERLEAVED", "BFLOAT16"),
        ("DRAM_INTERLEAVED", "DRAM_INTERLEAVED", "BFLOAT4_B"),
        ("DRAM_INTERLEAVED", "DRAM_INTERLEAVED", "BFLOAT8_B"),
    ]

    # Number of plots (12 combinations)
    num_plots = len(plot_order)

    # Set up subplots (4 rows and 3 columns)
    num_cols = 3
    num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate required number of rows
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 5 * num_rows), squeeze=False)

    # Flatten axes for easy indexing
    axes = axes.flatten()

    # Iterate over the combinations and plot
    for plot_index, (input_memory, output_memory, input_datatype) in enumerate(plot_order):
        ax = axes[plot_index]

        # Get the data for the current combination
        key = (input_memory, output_memory, input_datatype)
        if key not in data:
            continue

        # Extract num_tiles, mean_rmsre and std_rmsre
        num_tiles = sorted(data[key].keys())
        mean_rmsre = [data[key][nt][0] for nt in num_tiles]
        std_rmsre = [data[key][nt][1] for nt in num_tiles]

        # Create a bar plot with error bars
        ax.bar(
            num_tiles,
            mean_rmsre,
            yerr=std_rmsre,
            capsize=5,
            color="lightblue",
            edgecolor="black",
            label="Mean RMSRE ± Std",
        )

        # Set title and labels
        ax.set_title(f"Input: {input_memory}\nOutput: {output_memory}\nFormat: {input_datatype}", fontsize=10, pad=20)
        ax.set_xlabel("Num Tiles", fontsize=10)
        ax.set_ylabel("RMSRE (%)", fontsize=10)
        ax.grid(True, axis="y")
        ax.legend()

    # Remove any unused subplots
    for i in range(plot_index + 1, len(axes)):
        fig.delaxes(axes[i])

    # Adjust layout to prevent overlap and make it more readable
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(hspace=0.8)  # Increase vertical spacing

    # Save the figure as an image
    plt.savefig(output_image_path)
    print(f"Plot saved to {output_image_path}")

    # Optionally, show the plot (you can disable this to just save the image)
    plt.show()


# Main function to execute the plotting
def main():
    # Read the aggregated data
    data = read_aggregated_data(aggregated_file)

    # Plot the RMSRE values with standard deviation
    plot_rmsre_with_std(data)


# Run the main function
if __name__ == "__main__":
    main()
