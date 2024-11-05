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

output_file = "tests/sweep_framework/perf_model/filtered_rows_sharded.csv"
aggregated_file = "tests/sweep_framework/perf_model/aggregated_values_sharded.csv"
fitting_file = "tests/sweep_framework/perf_model/fitting_data_sharded.csv"  # New file for fitting data
image_file = "tests/sweep_framework/perf_model/fitting_plot_sharded.png"

# Define the number of cores
CORE_COUNT = 4  # Update this value as needed

# To hold data for plotting and aggregation
agg_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

# Open the selected CSV file for reading
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
    core_count_index = (
        header.index("CORE COUNT") if "CORE COUNT" in header else None
    )  # New line to get CORE_COUNT index

    # Read the first data row to get the CORE_COUNT value
    # first_data_row = next(reader, None)
    # if first_data_row and core_count_index is not None:
    #     CORE_COUNT = int(first_data_row[core_count_index])  # Set CORE_COUNT from the first row

    # Now reset the reader to re-read the CSV from the beginning
    infile.seek(0)
    reader = csv.reader(infile)
    header = next(reader)  # Skip the header again

    # Open the output CSV file for writing (this will overwrite it if it exists)
    with open(output_file, mode="w", newline="") as outfile:
        writer = csv.writer(outfile)

        # Write the header for the filtered columns
        writer.writerow(
            ["DEVICE KERNEL DURATION [ns]", "NUM_TILES", "INPUT_0_DATATYPE", "INPUT_0_MEMORY", "OUTPUT_0_MEMORY"]
        )

        # Iterate over each row in the input file
        for row in reader:
            if not row[0].startswith("(torch)"):
                # Calculate NUM_TILES and divide by CORE_COUNT
                num_tiles = None
                if input_0_x_index is not None and input_0_y_index is not None:
                    input_0_x = int(row[input_0_x_index]) if row[input_0_x_index] else 0
                    input_0_y = int(row[input_0_y_index]) if row[input_0_y_index] else 0
                    core_count = int(row[core_count_index]) if row[core_count_index] else 1
                    num_tiles = (input_0_x * input_0_y) // (1024 * core_count)

                # Prepare the filtered row
                filtered_row = [
                    row[device_kernel_duration_index] if device_kernel_duration_index is not None else "",
                    num_tiles,
                    row[input_0_datatype_index] if input_0_datatype_index is not None else "",
                    row[input_0_memory_index].replace("DEV_0_", "") if input_0_memory_index is not None else "",
                    row[output_0_memory_index].replace("DEV_0_", "") if output_0_memory_index is not None else "",
                ]

                # Write the filtered row to the output file
                writer.writerow(filtered_row)

                # Store data for aggregation
                input_memory = filtered_row[3]
                output_memory = filtered_row[4]
                input_datatype = filtered_row[2]
                agg_data[input_memory][output_memory][input_datatype].append((num_tiles, int(filtered_row[0])))


# Open the aggregated values file for writing (this will overwrite it if it exists)
with open(aggregated_file, mode="w", newline="") as agg_file:
    agg_writer = csv.writer(agg_file)
    agg_writer.writerow(
        ["INPUT_0_DATATYPE", "INPUT_0_MEMORY", "OUTPUT_0_MEMORY", "NUM_TILES", "AVG_DEVICE_KERNEL_DURATION [ns]"]
    )

    for input_memory, output_dict in agg_data.items():
        for output_memory, datatype_dict in output_dict.items():
            for input_datatype, values in datatype_dict.items():
                num_tiles_avg = {}
                for num_tiles, duration in values:
                    if num_tiles not in num_tiles_avg:
                        num_tiles_avg[num_tiles] = []
                    num_tiles_avg[num_tiles].append(duration)

                for num_tiles, durations in num_tiles_avg.items():
                    avg_duration = sum(durations) / len(durations) if durations else 0
                    agg_writer.writerow([input_datatype, input_memory, output_memory, num_tiles, avg_duration])

# New section to save fitting data
with open(fitting_file, mode="w", newline="") as fit_file:
    fit_writer = csv.writer(fit_file)
    fit_writer.writerow(["INPUT_0_DATATYPE", "INPUT_0_MEMORY", "OUTPUT_0_MEMORY", "COEFFICIENT_A", "COEFFICIENT_B"])

    # Plotting from aggregated data
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(20, 15))  # 4 rows and 3 columns
    axes = axes.flatten()
    plot_count = 0

    # Loop to create plots for each combination of INPUT_0_MEMORY and OUTPUT_0_MEMORY
    for input_memory, output_dict in agg_data.items():
        for output_memory, datatype_dict in output_dict.items():
            for input_datatype, values in datatype_dict.items():
                if plot_count < len(axes):
                    # Prepare data for plotting
                    num_tiles_avg = {}

                    for num_tiles, duration in values:
                        num_tiles_avg[num_tiles] = duration

                    num_tiles = np.array(list(num_tiles_avg.keys()))
                    avg_durations = np.array(list(num_tiles_avg.values()))

                    # Plotting aggregated values
                    axes[plot_count].scatter(
                        num_tiles, avg_durations, label=f"{input_memory} -> {output_memory} [{input_datatype}]"
                    )

                    # Fit line
                    if len(num_tiles) > 1:  # Ensure there's enough data for fitting
                        coeffs = np.polyfit(num_tiles, avg_durations, 1)  # Linear fit
                        fit_line = np.polyval(coeffs, num_tiles)
                        axes[plot_count].plot(num_tiles, fit_line, color="red", linestyle="--", label="Fit Line")

                        # Save fitting coefficients
                        fit_writer.writerow(
                            [
                                input_datatype,
                                input_memory,
                                output_memory,
                                coeffs[0],  # Slope (A)
                                coeffs[1],  # Intercept (B)
                            ]
                        )

                    # Set titles and labels
                    axes[plot_count].set_title(
                        f"AVG KERNEL DURATION vs NUM_TILES\n{input_memory} -> {output_memory} [{input_datatype}]"
                    )
                    axes[plot_count].set_xlabel("NUM_TILES")
                    axes[plot_count].set_ylabel("AVG DEVICE KERNEL DURATION [ns]")
                    axes[plot_count].legend()
                    axes[plot_count].grid()

                    plot_count += 1

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(image_file)
    plt.close()

print(f"Filtered data has been written to {output_file}.")
print(f"Aggregated data has been written to {aggregated_file}.")
print(f"Fitting data has been written to {fitting_file}.")
print(f"Plots have been saved to {image_file}.")
