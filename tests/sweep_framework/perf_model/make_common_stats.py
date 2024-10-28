import os
import pandas as pd
import sys

# Check if a value was provided as a command-line argument
if len(sys.argv) != 2:
    print("Usage: python script.py VALUE")
    exit()

value = sys.argv[1]

# Path to the directory containing the CSV files
reports_dir = "generated/profiler/reports/"

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

# Read the last CSV file
df_last = pd.read_csv(last_csv_file)

# Initialize a new DataFrame to store the results
result_data = []

# Iterate through the rows of the last CSV file
for index, row in df_last.iterrows():
    # Extract INPUT_0_MEMORY and OUTPUT_0_MEMORY
    input_memory = row["INPUT_0_MEMORY"].replace("DEV_0_", "")
    output_memory = row["OUTPUT_0_MEMORY"].replace("DEV_0_", "")

    # Extract INPUT_0_X and INPUT_0_Y to compute num_tiles
    input_0_x = row["INPUT_0_X"]
    input_0_y = row["INPUT_0_Y"]
    num_tiles = input_0_x * input_0_y // 1024

    # Extract DEVICE KERNEL DURATION [ns]
    device_kernel_duration = row["DEVICE KERNEL DURATION [ns]"]

    # Append the row to the result data, including the VALUE
    result_data.append(
        {
            "num_tiles": num_tiles,
            "device_kernel_duration_ns": device_kernel_duration,
            "value": value,  # Add the VALUE to the row
        }
    )

# Create a new DataFrame from the result data
df_result = pd.DataFrame(result_data)

# Save the result data to a new CSV file without headers
output_csv_file = "filtered_memory_values_common.csv"
df_result.to_csv(output_csv_file, index=False, header=False)  # Exclude headers

if not df_result.empty:
    print(f"Filtered data saved to {output_csv_file}")
else:
    print("No records found.")
