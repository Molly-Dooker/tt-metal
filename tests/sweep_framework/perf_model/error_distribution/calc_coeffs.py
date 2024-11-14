import csv
import os
import numpy as np
from collections import defaultdict


# Define the input and output directories
input_folder = "tests/sweep_framework/perf_model/error_distribution/csvs"  # Folder where your 10 CSV files are located
output_folder = "tests/sweep_framework/perf_model/error_distribution/coef_results"  # Folder where we will store the coefficient results

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get the list of all the CSV files in the input folder
csv_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".csv")])


# Function to calculate NUM_TILES from the X and Y columns
def calculate_num_tiles(x, y):
    try:
        return (int(x) * int(y)) // 1024
    except ValueError:
        return None  # Return None if conversion fails


# Function to calculate the coefficients for a given CSV file
def calculate_coefficients(input_csv_file):
    with open(input_csv_file, mode="r") as infile:
        reader = csv.reader(infile)
        header = next(reader)  # Skip the header

        # Define the indices for the columns we want to process
        device_kernel_duration_index = header.index("DEVICE KERNEL DURATION [ns]")
        input_0_x_index = header.index("INPUT_0_X")
        input_0_y_index = header.index("INPUT_0_Y")
        input_0_memory_index = header.index("INPUT_0_MEMORY")
        output_0_memory_index = header.index("OUTPUT_0_MEMORY")
        input_0_datatype_index = header.index("INPUT_0_DATATYPE")

        # Dictionary to store data for each combination of input memory, output memory, and datatype
        data = defaultdict(list)

        # Iterate over each row in the CSV
        for row in reader:
            # Calculate NUM_TILES based on INPUT_0_X and INPUT_0_Y
            num_tiles = calculate_num_tiles(row[input_0_x_index], row[input_0_y_index])
            if num_tiles is None:
                continue  # Skip rows with invalid NUM_TILES

            # Get the DEVICE KERNEL DURATION [ns]
            device_kernel_duration = float(row[device_kernel_duration_index])

            # Get the values for input memory, output memory, and input datatype
            input_memory = row[input_0_memory_index]
            output_memory = row[output_0_memory_index]
            input_datatype = row[input_0_datatype_index]

            # Store data in the dictionary
            data[(input_memory, output_memory, input_datatype)].append((num_tiles, device_kernel_duration))

        return data


# Function to perform linear regression and save coefficients
def perform_linear_regression_and_save(data, output_file):
    with open(output_file, mode="w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["INPUT_0_MEMORY", "OUTPUT_0_MEMORY", "INPUT_0_DATATYPE", "SLOPE (A)", "INTERCEPT (B)"])

        # Iterate over each combination of input memory, output memory, and datatype
        for (input_memory, output_memory, input_datatype), values in data.items():
            # Prepare data for linear regression (NUM_TILES vs DEVICE KERNEL DURATION)
            num_tiles = np.array([v[0] for v in values])
            kernel_durations = np.array([v[1] for v in values])

            # Perform linear regression (fit a line)
            if len(num_tiles) > 1:
                coeffs = np.polyfit(num_tiles, kernel_durations, 1)  # Linear fit
                slope, intercept = coeffs

                # Write the coefficients to the output CSV
                writer.writerow([input_memory, output_memory, input_datatype, slope, intercept])

                print(f"Coefficients for {input_memory} -> {output_memory} [{input_datatype}] saved.")
            else:
                print(
                    f"Not enough data to calculate coefficients for {input_memory} -> {output_memory} [{input_datatype}]"
                )


# Iterate over all CSV files in the input folder
for i, csv_file in enumerate(csv_files):
    input_csv_file = os.path.join(input_folder, csv_file)

    # Calculate coefficients for the current CSV file
    data = calculate_coefficients(input_csv_file)

    if data:
        # Prepare the output filename and write the results
        output_csv_file = os.path.join(output_folder, f"coef_{i+1}.csv")
        perform_linear_regression_and_save(data, output_csv_file)
    else:
        print(f"No valid data found in {csv_file}.")

print("Coefficient calculation complete for all files.")
