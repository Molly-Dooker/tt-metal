import os
import csv
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# Paths to your directories and files
fitting_data_file = "tests/sweep_framework/perf_model/fitting_data.csv"
coef_results_dir = "tests/sweep_framework/perf_model/error_distribution/coef_results"
output_file = "tests/sweep_framework/perf_model/error_distribution/predicted_kernel_durations_multiple.csv"
aggregated_output_file = "tests/sweep_framework/perf_model/error_distribution/aggregated_predicted_kernel_durations.csv"
output_image_path = (
    "tests/sweep_framework/perf_model/error_distribution/rmsre_dif_coeff_plots.png"  # Path to save the image
)

# List of num_tiles to predict for
num_tiles_list = [5, 20, 200, 500, 3000, 4000, 7000, 9000, 11000]


# Function to read the fitting data coefficients (actual values)
def read_fitting_data():
    fitting_data = {}
    with open(fitting_data_file, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            input_datatype = row["INPUT_0_DATATYPE"]
            input_memory = row["INPUT_0_MEMORY"]
            output_memory = row["OUTPUT_0_MEMORY"]
            coef_a = float(row["COEFFICIENT_A"])
            coef_b = float(row["COEFFICIENT_B"])
            fitting_data[(input_memory, output_memory, input_datatype)] = (coef_a, coef_b)
    return fitting_data


# Function to read coefficient data from multiple CSV files in the directory
def read_coef_results():
    coef_data = []
    for file_name in os.listdir(coef_results_dir):
        if file_name.endswith(".csv"):
            file_path = os.path.join(coef_results_dir, file_name)
            with open(file_path, mode="r") as file:
                reader = csv.DictReader(file)
                file_coef_data = {}
                for row in reader:
                    input_memory = row["INPUT_0_MEMORY"]
                    output_memory = row["OUTPUT_0_MEMORY"]
                    input_datatype = row["INPUT_0_DATATYPE"]
                    slope = float(row["SLOPE (A)"])
                    intercept = float(row["INTERCEPT (B)"])
                    file_coef_data[(input_memory, output_memory, input_datatype)] = (slope, intercept)
                coef_data.append(file_coef_data)
    return coef_data


# Function to calculate predicted kernel durations for all num_tiles values
def calculate_predicted_durations(fitting_data, coef_data, num_tiles_list):
    results = []

    for (input_memory, output_memory, input_datatype), (coef_a, coef_b) in fitting_data.items():
        # Loop through each file's coefficients
        for file_coef_data in coef_data:
            if (input_memory, output_memory, input_datatype) in file_coef_data:
                slope, intercept = file_coef_data[(input_memory, output_memory, input_datatype)]

                # For each num_tiles, calculate the predicted duration
                for num_tiles in num_tiles_list:
                    # Actual kernel duration (from fitting data)
                    actual_duration = coef_a * num_tiles + coef_b

                    # Predicted kernel duration (from the current file's coefficients)
                    predicted_duration = slope * num_tiles + intercept

                    # Store results with actual and predicted durations
                    results.append(
                        (input_memory, output_memory, input_datatype, num_tiles, actual_duration, predicted_duration)
                    )

    return results


# Function to calculate RMSRE for each combination of input/output memory, input datatype, and num_tiles
def calculate_rmsre(results):
    rmsre_results = []

    # Group results by (input_memory, output_memory, input_datatype, num_tiles)
    grouped_results = {}
    for result in results:
        input_memory, output_memory, input_datatype, num_tiles, actual_duration, predicted_duration = result
        key = (input_memory, output_memory, input_datatype, num_tiles)
        if key not in grouped_results:
            grouped_results[key] = []
        grouped_results[key].append((actual_duration, predicted_duration))

    # For each group, calculate RMSRE using all predictions for that group
    for key, values in grouped_results.items():
        input_memory, output_memory, input_datatype, num_tiles = key

        actual_values = [v[0] for v in values]
        predicted_values = [v[1] for v in values]

        if (
            input_memory == "L1_INTERLEAVED"
            and output_memory == "L1_INTERLEAVED"
            and input_datatype == "BFLOAT16"
            and num_tiles == 20
        ):
            print(actual_values)
            print(predicted_values)

        # Calculate RMSRE: sqrt(mean((predicted - actual) / actual)^2)
        squared_relative_errors = [(pred - actual) / actual for pred, actual in zip(predicted_values, actual_values)]
        rmsre = math.sqrt(sum([x**2 for x in squared_relative_errors]) / len(squared_relative_errors)) * 100

        if (
            input_memory == "L1_INTERLEAVED"
            and output_memory == "L1_INTERLEAVED"
            and input_datatype == "BFLOAT16"
            and num_tiles == 20
        ):
            print(rmsre)

        # Store the RMSRE for this num_tiles and memory combination
        rmsre_results.append(
            (
                input_memory,  # INPUT_0_MEMORY
                output_memory,  # OUTPUT_0_MEMORY
                input_datatype,  # INPUT_0_DATATYPE
                num_tiles,  # NUM_TILES
                actual_values[0],  # ACTUAL_KERNEL_DURATION_NS (assuming all actual values are the same for a group)
                predicted_values,  # List of PREDICTED_KERNEL_DURATION_NS (all predictions for that num_tiles)
                rmsre,  # RMSRE for this group
            )
        )

    return rmsre_results


# Function to save the results to CSV
def save_results_to_csv(results, output_file):
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow(
            [
                "INPUT_0_MEMORY",
                "OUTPUT_0_MEMORY",
                "INPUT_0_DATATYPE",
                "NUM_TILES",
                "ACTUAL_KERNEL_DURATION_NS",
                "PREDICTED_KERNEL_DURATION_NS",
                "RMSRE",
            ]
        )
        # Write the data rows
        writer.writerows(results)


# Function to plot RMSRE vs num_tiles for each memory combination
def plot_rmsre_vs_num_tiles(rmsre_results, memory_combinations, output_image_path="rmsre_vs_num_tiles.png"):
    # Create a dictionary to store RMSRE values for each memory combination
    rmsre_data = {mem_comb: [] for mem_comb in memory_combinations}

    # Filter data for each memory combination and prepare data for plotting
    for result in rmsre_results:
        input_memory, output_memory, input_datatype, num_tiles, actual_duration, predicted_duration, rmsre = result
        # Filter results for the specific memory combination
        if (input_memory, output_memory, input_datatype) in memory_combinations:
            rmsre_data[(input_memory, output_memory, input_datatype)].append((num_tiles, rmsre))

    # Create a figure with 4 rows and 3 columns for the subplots
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(16, 18))
    axes = axes.flatten()  # Flatten the axes array for easier indexing

    # Loop through each memory combination and plot on the appropriate subplot
    for idx, mem_comb in enumerate(memory_combinations):
        ax = axes[idx]  # Get the current axis to plot on
        data = rmsre_data[mem_comb]

        if data:  # Ensure there is data to plot
            num_tiles = [entry[0] for entry in data]
            rmsre = [entry[1] for entry in data]  # RMSRE already in percentage
            ax.plot(num_tiles, rmsre, marker="o", label=f"{mem_comb[0]} -> {mem_comb[1]} -> {mem_comb[2]}")

        # Set subplot labels
        ax.set_xlabel("Number of Tiles")
        ax.set_ylabel("RMSRE (%)")
        ax.set_title(f"{mem_comb[0]} -> {mem_comb[1]} -> {mem_comb[2]}")
        ax.set_xscale("linear")  # Linear scale for the x-axis
        ax.set_yscale("linear")  # Linear scale for the y-axis
        ax.grid(True)
        ax.legend(loc="upper right")

        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))

    # Adjust layout for better spacing between subplots
    plt.tight_layout()

    # Save the plot to the output image path
    plt.savefig(output_image_path)
    print(f"Plot saved to {output_image_path}")

    # Show the plot
    plt.show()


# Main function to orchestrate everything
def main():
    # Read fitting data and coefficient results
    fitting_data = read_fitting_data()
    coef_data = read_coef_results()

    # Calculate the predicted kernel durations
    results = calculate_predicted_durations(fitting_data, coef_data, num_tiles_list)

    # Calculate RMSRE for each group
    rmsre_results = calculate_rmsre(results)

    # Save the results to a CSV file
    save_results_to_csv(rmsre_results, aggregated_output_file)

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

    # Plot the RMSRE vs num_tiles for the specified memory combinations
    plot_rmsre_vs_num_tiles(rmsre_results, memory_combinations, output_image_path)

    print(f"Results with RMSRE saved to: {aggregated_output_file}")


if __name__ == "__main__":
    main()
