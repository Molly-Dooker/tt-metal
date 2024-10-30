import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description="Process performance results CSV files.")
parser.add_argument("-f", "--file", type=str, help="Path to the input CSV file", required=False)
args = parser.parse_args()

# Path to the directory containing the CSV files
reports_dir = "generated/profiler/reports/"

# Determine the CSV file to process
if args.file:
    # Use the provided file if specified
    last_csv_file = os.path.abspath(args.file)
else:
    # Get all CSV files recursively from the specified directory that start with 'ops_perf_results_'
    csv_files = []
    for dirpath, _, filenames in os.walk(reports_dir):
        for f in filenames:
            if f.endswith(".csv") and f.startswith("ops_perf_results_"):
                csv_files.append(os.path.abspath(os.path.join(dirpath, f)))

    # Sort the files by their modification time
    csv_files.sort(key=lambda f: os.path.getmtime(f))

    if not csv_files:
        print("No CSV files found in the specified directory.")
        exit()

    last_csv_file = csv_files[-1]  # Get the most recently modified CSV file

output_file = "tests/sweep_framework/perf_model/filtered_rows_new_data.csv"
fitting_file = "tests/sweep_framework/perf_model/fitting_data.csv"  # File containing fitting coefficients

# To hold data for plotting and aggregation
agg_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

# Load fitting coefficients
fitting_coeffs = {}
with open(fitting_file, mode="r") as fit_file:
    fit_reader = csv.DictReader(fit_file)
    for row in fit_reader:
        key = (row["INPUT_0_MEMORY"], row["OUTPUT_0_MEMORY"], row["INPUT_0_DATATYPE"])
        fitting_coeffs[key] = (float(row["COEFFICIENT_A"]), float(row["COEFFICIENT_B"]))

# Open the selected CSV file for reading
with open(last_csv_file, mode="r") as infile:
    reader = csv.reader(infile)
    header = next(reader)

    # Define the indices of the columns we want to keep
    device_kernel_duration_index = (
        header.index("DEVICE KERNEL DURATION [ns]") if "DEVICE KERNEL DURATION [ns]" in header else None
    )
    input_0_y_index = header.index("INPUT_0_Y") if "INPUT_0_Y" in header else None
    input_0_x_index = header.index("INPUT_0_X") if "INPUT_0_X" in header else None
    input_0_memory_index = header.index("INPUT_0_MEMORY") if "INPUT_0_MEMORY" in header else None
    output_0_memory_index = header.index("OUTPUT_0_MEMORY") if "OUTPUT_0_MEMORY" in header else None
    input_0_datatype_index = header.index("INPUT_0_DATATYPE") if "INPUT_0_DATATYPE" in header else None

    # Open the output CSV file for writing (this will overwrite it if it exists)
    with open(output_file, mode="w", newline="") as outfile:
        writer = csv.writer(outfile)

        # Write the header for the filtered columns
        writer.writerow(
            [
                "DEVICE KERNEL DURATION [ns]",
                "ESTIMATED KERNEL DURATION [ns]",
                "NUM_TILES",
                "INPUT_0_DATATYPE",
                "INPUT_0_MEMORY",
                "OUTPUT_0_MEMORY",
            ]
        )

        actual_durations = []
        estimated_durations = []

        # Iterate over each row in the input file
        for row in reader:
            if not row[0].startswith("(torch)"):
                # Calculate NUM_TILES
                num_tiles = None
                if input_0_x_index is not None and input_0_y_index is not None:
                    input_0_x = int(row[input_0_x_index]) if row[input_0_x_index] else 0
                    input_0_y = int(row[input_0_y_index]) if row[input_0_y_index] else 0
                    num_tiles = (input_0_x * input_0_y) // 1024

                # Prepare the filtered row
                input_memory = (
                    row[input_0_memory_index].replace("DEV_0_", "") if input_0_memory_index is not None else ""
                )
                output_memory = (
                    row[output_0_memory_index].replace("DEV_0_", "") if output_0_memory_index is not None else ""
                )
                input_datatype = row[input_0_datatype_index] if input_0_datatype_index is not None else ""

                # Actual duration
                actual_duration = (
                    int(row[device_kernel_duration_index]) if device_kernel_duration_index is not None else 0
                )
                actual_durations.append(actual_duration)

                # Calculate estimated duration
                key = (input_memory, output_memory, input_datatype)
                estimated_duration = 0
                if key in fitting_coeffs:
                    a, b = fitting_coeffs[key]
                    estimated_duration = a * num_tiles + b

                estimated_durations.append(estimated_duration)

                # Write the filtered row to the output file
                writer.writerow(
                    [actual_duration, estimated_duration, num_tiles, input_datatype, input_memory, output_memory]
                )


# Function to calculate RRMSE
def rrmse(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)

    # Calculate mean of actual values
    mean_actual = np.mean(actual)

    # Calculate RMSE
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    # Calculate RRMSE
    rrmse_value = (rmse / mean_actual) * 100

    return rrmse_value


# Function to calculate RMSRE
def rmsre(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)

    # Calculate relative errors
    relative_errors = np.abs((actual - predicted) / actual)

    # Calculate RMSRE
    rmsre_value = np.sqrt(np.mean(relative_errors**2)) * 100  # Percentage

    return rmsre_value


# Calculate and print RRMSE and RMSRE
rrmse_value = rrmse(actual_durations, estimated_durations)
rmsre_value = rmsre(actual_durations, estimated_durations)

print(f"RRMSE: {rrmse_value:.2f}%")
print(f"RMSRE: {rmsre_value:.2f}%")

print(f"Filtered data has been written to {output_file}.")
