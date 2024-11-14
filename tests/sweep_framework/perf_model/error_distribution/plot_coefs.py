import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# Folder where the coefficient files are stored
output_folder = "tests/sweep_framework/perf_model/error_distribution/coef_results"

# Define the 12 combinations of input memory, output memory, and input datatype
memory_combinations = [
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


# Function to read coefficients from a file
def read_coefficients(coefficients_file):
    coefs = {}

    with open(coefficients_file, mode="r") as file:
        reader = csv.DictReader(file)  # Using DictReader to access columns by name
        for row in reader:
            # Extract information from the row
            input_memory = row["INPUT_0_MEMORY"]
            output_memory = row["OUTPUT_0_MEMORY"]
            input_datatype = row["INPUT_0_DATATYPE"]
            coef_a = float(row["SLOPE (A)"])
            coef_b = float(row["INTERCEPT (B)"])

            # Store coefficients in a dictionary with key as tuple of (input_memory, output_memory, input_datatype)
            coefs[(input_memory, output_memory, input_datatype)] = (coef_a, coef_b)

    return coefs


# Function to plot the coefficients for each combination
def plot_coefficients(all_coefs):
    # Define the combinations of input_memory, output_memory, and input_datatype
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))  # 12 subplots (3x4 grid)
    axes = axes.flatten()  # Flatten to easily index into subplots

    # Loop over each subplot (12 in total)
    for i, (input_memory, output_memory, input_datatype) in enumerate(memory_combinations):
        ax = axes[i]

        # Generate a line for each CSV file
        for file_name, coefs in all_coefs.items():
            # Check if the current combination is in the current coefficient file
            if (input_memory, output_memory, input_datatype) in coefs:
                coef_a, coef_b = coefs[(input_memory, output_memory, input_datatype)]

                # Generate num_tiles and the corresponding device kernel duration
                num_tiles = np.linspace(0, 15000, 1000)  # 1000 points from 0 to 15000
                kernel_duration = coef_a * num_tiles + coef_b  # Linear equation

                # Plot the line for this combination and file
                ax.plot(num_tiles, kernel_duration, label=f"{file_name}")

        # Customize the plot
        ax.set_title(f"{input_memory} -> {output_memory} [{input_datatype}]")
        ax.set_xlabel("NUM_TILES")
        ax.set_ylabel("DEVICE KERNEL DURATION [ns]")
        ax.set_xlim(0, 15000)  # X axis range (num_tiles)
        ax.set_ylim(0, 300000)  # Y axis range (kernel duration)
        ax.legend()
        ax.grid(True)

    # Adjust layout for better spacing between plots
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "kernel_duration_vs_num_tiles.png"))  # Save the plot as a PNG
    plt.show()


def main():
    # Get all CSV files in the output folder
    coefficient_files = [f for f in os.listdir(output_folder) if f.endswith(".csv")]

    # Dictionary to store all the coefficients, file-wise
    all_coefs = {}

    # Process each coefficient file
    for file_name in coefficient_files:
        file_path = os.path.join(output_folder, file_name)
        print(f"Reading coefficients from: {file_path}")
        coefs = read_coefficients(file_path)

        # Store coefficients by file name
        all_coefs[file_name] = coefs

    # Plot the coefficients for all the combinations
    plot_coefficients(all_coefs)


# Run the main function
if __name__ == "__main__":
    main()
