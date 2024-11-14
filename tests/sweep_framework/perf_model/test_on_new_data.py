import os
import csv
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import argparse

# Argument parsing and file setup
parser = argparse.ArgumentParser(description="Process performance results CSV files.")
parser.add_argument("-f", "--file", type=str, help="Path to the input CSV file", required=False)
args = parser.parse_args()

reports_dir = "generated/profiler/reports/"
last_csv_file = (
    args.file
    if args.file
    else max(
        [os.path.join(dp, f) for dp, dn, filenames in os.walk(reports_dir) for f in filenames if f.endswith(".csv")],
        key=os.path.getmtime,
    )
)

output_file = "tests/sweep_framework/perf_model/filtered_rows_new_data.csv"
stats_file = "tests/sweep_framework/perf_model/kernel_duration_stats.csv"
plots_dir = "tests/sweep_framework/perf_model/plots"
os.makedirs(plots_dir, exist_ok=True)  # Ensure the plots directory exists

# Load fitting coefficients
fitting_file = "tests/sweep_framework/perf_model/fitting_data.csv"
fitting_coeffs = {}
with open(fitting_file, mode="r") as fit_file:
    fit_reader = csv.DictReader(fit_file)
    for row in fit_reader:
        key = (row["INPUT_0_MEMORY"], row["OUTPUT_0_MEMORY"], row["INPUT_0_DATATYPE"])
        fitting_coeffs[key] = (float(row["COEFFICIENT_A"]), float(row["COEFFICIENT_B"]))

# Process the selected CSV file
data = []
kernel_durations = defaultdict(lambda: defaultdict(list))

with open(last_csv_file, mode="r") as infile:
    reader = csv.reader(infile)
    header = next(reader)

    device_kernel_duration_index = header.index("DEVICE KERNEL DURATION [ns]")
    input_0_y_index = header.index("INPUT_0_Y")
    input_0_x_index = header.index("INPUT_0_X")
    input_0_memory_index = header.index("INPUT_0_MEMORY")
    output_0_memory_index = header.index("OUTPUT_0_MEMORY")
    input_0_datatype_index = header.index("INPUT_0_DATATYPE")

    for row in reader:
        if not row[0].startswith("(torch)"):
            input_0_x = int(row[input_0_x_index]) if row[input_0_x_index] else 0
            input_0_y = int(row[input_0_y_index]) if row[input_0_y_index] else 0
            num_tiles = (input_0_x * input_0_y) // 1024

            input_memory = row[input_0_memory_index].replace("DEV_0_", "")
            output_memory = row[output_0_memory_index].replace("DEV_0_", "")
            input_datatype = row[input_0_datatype_index]

            actual_duration = int(row[device_kernel_duration_index])
            key = (input_memory, output_memory, input_datatype)

            estimated_duration = fitting_coeffs.get(key, (0, 0))
            estimated_duration = estimated_duration[0] * num_tiles + estimated_duration[1]

            # Store the values for output
            data.append((input_memory, output_memory, input_datatype, num_tiles, actual_duration, estimated_duration))
            kernel_durations[(input_memory, output_memory, input_datatype)][num_tiles].append(actual_duration)

# Write detailed results to a CSV file
with open(output_file, mode="w", newline="") as out_file:
    writer = csv.writer(out_file)
    writer.writerow(
        [
            "Input Memory",
            "Output Memory",
            "Input Datatype",
            "Num Tiles",
            "Actual Duration [ns]",
            "Estimated Duration [ns]",
        ]
    )
    for entry in data:
        writer.writerow(entry)

# Calculate statistics and write to a new CSV file
with open(stats_file, mode="w", newline="") as stats_out_file:
    stats_writer = csv.writer(stats_out_file)
    stats_writer.writerow(
        [
            "Input Memory",
            "Output Memory",
            "Input Datatype",
            "Num Tiles",
            "Mean Duration [ns]",
            "Std Dev Duration [ns]",
            "Std Dev/Mean",
        ]
    )

    for key, num_tiles_dict in kernel_durations.items():
        for num_tiles, durations in num_tiles_dict.items():
            mean_duration = np.mean(durations)
            std_duration = np.std(durations)
            std_dev_over_mean = std_duration / mean_duration if mean_duration > 0 else np.nan
            stats_writer.writerow([key[0], key[1], key[2], num_tiles, mean_duration, std_duration, std_dev_over_mean])

# Prepare for plotting
plot_data = defaultdict(lambda: defaultdict(dict))

for key, num_tiles_dict in kernel_durations.items():
    for num_tiles, durations in num_tiles_dict.items():
        mean_duration = np.mean(durations)
        std_duration = np.std(durations)

        # Calculate RRMSE
        rrmse_value = (std_duration / mean_duration * 100) if mean_duration > 0 else 0

        # Calculate RMSRE
        estimated_durations = []
        for actual_duration in durations:
            estimated_duration = fitting_coeffs.get(key, (0, 0))
            estimated_duration = estimated_duration[0] * num_tiles + estimated_duration[1]
            estimated_durations.append(estimated_duration)

        rmsre_value = (
            (np.sqrt(np.mean((np.array(durations) - np.array(estimated_durations)) ** 2)) / mean_duration * 100)
            if mean_duration > 0
            else 0
        )

        plot_data[key][num_tiles] = (rrmse_value, rmsre_value)

# Prepare for plotting
# Explicitly define the 12 combinations you want to plot
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

# Filter plot data to match the specified order
ordered_plot_data = [(key, plot_data[key]) for key in plot_order if key in plot_data]

num_plots = len(ordered_plot_data)
num_cols = 3  # Number of columns for the subplots
num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate required number of rows
fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 5 * num_rows), squeeze=False)

# Flatten axes for easy indexing
axes = axes.flatten()

# Plotting
for plot_index, (key, num_tiles_metrics) in enumerate(ordered_plot_data):
    input_memory, output_memory, input_datatype = key
    num_tiles = sorted(num_tiles_metrics.keys())
    rrmse_values = [num_tiles_metrics[nt][0] for nt in num_tiles]
    rmsre_values = [num_tiles_metrics[nt][1] for nt in num_tiles]

    ax = axes[plot_index]  # Get the axis for the current plot
    ax.plot(num_tiles, rrmse_values, "bo-", label="RRMSE (%)", alpha=0.7)
    ax.plot(num_tiles, rmsre_values, "ro-", label="RMSRE (%)", alpha=0.7)

    # Set title and labels
    ax.set_title(
        f"Input: {input_memory}\nOutput: {output_memory}\nFormat: {input_datatype}", fontsize=12, pad=20
    )  # Multi-line title
    ax.set_xlabel("Num Tiles")
    ax.set_ylabel("Error (%)")
    ax.grid()
    ax.legend()

# Remove any unused subplots
for i in range(plot_index + 1, len(axes)):
    fig.delaxes(axes[i])

# Adjust layout to prevent overlap
plt.tight_layout(pad=3.0)
plt.subplots_adjust(hspace=0.8)  # Increase vertical spacing
plt.savefig(os.path.join(plots_dir, "all_plots_ordered.png"))
print(f"All ordered plots have been saved to {os.path.join(plots_dir, 'all_plots_ordered.png')}.")

# Prepare for plotting RMSE without dividing by mean
plot_data_rmse_without_mean = defaultdict(dict)

for key, num_tiles_dict in kernel_durations.items():
    for num_tiles, durations in num_tiles_dict.items():
        # Calculate RMSE directly
        estimated_durations = []
        for actual_duration in durations:
            estimated_duration = fitting_coeffs.get(key, (0, 0))
            estimated_duration = estimated_duration[0] * num_tiles + estimated_duration[1]
            estimated_durations.append(estimated_duration)

        # Calculate RMSE
        rmse_value = np.sqrt(np.mean((np.array(durations) - np.array(estimated_durations)) ** 2))

        plot_data_rmse_without_mean[key][num_tiles] = rmse_value

# Prepare for plotting
ordered_plot_data_rmse_without_mean = [
    (key, plot_data_rmse_without_mean[key]) for key in plot_order if key in plot_data_rmse_without_mean
]

num_plots_rmse_without_mean = len(ordered_plot_data_rmse_without_mean)
num_cols = 3  # Number of columns for the subplots
num_rows_rmse_without_mean = (
    num_plots_rmse_without_mean + num_cols - 1
) // num_cols  # Calculate required number of rows
fig, axes = plt.subplots(
    num_rows_rmse_without_mean, num_cols, figsize=(18, 5 * num_rows_rmse_without_mean), squeeze=False
)

# Flatten axes for easy indexing
axes = axes.flatten()

# Plotting
for plot_index, (key, num_tiles_metrics) in enumerate(ordered_plot_data_rmse_without_mean):
    input_memory, output_memory, input_datatype = key
    num_tiles = sorted(num_tiles_metrics.keys())
    rmse_values = [num_tiles_metrics[nt] for nt in num_tiles]

    ax = axes[plot_index]  # Get the axis for the current plot
    ax.plot(num_tiles, rmse_values, "bo-", label="RMSE (ns)", alpha=0.7)

    # Set title and labels
    ax.set_title(
        f"RMSE - Input: {input_memory}\nOutput: {output_memory}\nFormat: {input_datatype}", fontsize=12, pad=20
    )  # Multi-line title
    ax.set_xlabel("Num Tiles")
    ax.set_ylabel("RMSE [ns]")  # Updated y-axis label
    ax.grid()
    ax.legend()

# Remove any unused subplots
for i in range(plot_index + 1, len(axes)):
    fig.delaxes(axes[i])

# Adjust layout to prevent overlap
plt.tight_layout(pad=3.0)
plt.subplots_adjust(hspace=0.8)  # Increase vertical spacing
plt.savefig(os.path.join(plots_dir, "rmse_without_mean.png"))
print(f"RMSE without mean plot has been saved to {os.path.join(plots_dir, 'rmse_without_mean.png')}.")
