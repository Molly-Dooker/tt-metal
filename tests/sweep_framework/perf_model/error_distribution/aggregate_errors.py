import csv
import numpy as np
from collections import defaultdict

# File paths
input_file = "tests/sweep_framework/perf_model/rmsre_mean_results.csv"
output_file = "tests/sweep_framework/perf_model/rmsre_aggregated_results.csv"


# Function to aggregate data and compute statistics
def aggregate_rmsre_data(input_file, output_file):
    # Dictionary to hold the RMSRE values for each (input_memory, output_memory, input_datatype, num_tiles) key
    rmsre_data = defaultdict(list)

    # Read input CSV file
    with open(input_file, mode="r") as infile:
        reader = csv.reader(infile)
        header = next(reader)  # Skip header row

        for row in reader:
            input_memory, output_memory, input_datatype, num_tiles, rmsre = row
            rmsre = float(rmsre)  # Convert RMSRE to float

            # Group by (input_memory, output_memory, input_datatype, num_tiles)
            key = (input_memory, output_memory, input_datatype, int(num_tiles))
            rmsre_data[key].append(rmsre)

    # Open the output file for writing aggregated results
    with open(output_file, mode="w", newline="") as outfile:
        writer = csv.writer(outfile)

        # Write header
        writer.writerow(
            ["Input Memory", "Output Memory", "Input Datatype", "Num Tiles", "Mean RMSRE", "Std RMSRE", "Std/Mean"]
        )

        # Process each group to compute mean, std, and std/mean
        for key, rmsre_values in rmsre_data.items():
            mean_rmsre = np.mean(rmsre_values)
            std_rmsre = np.std(rmsre_values)
            std_mean_ratio = std_rmsre / mean_rmsre if mean_rmsre != 0 else 0

            # Write the results to the output file
            writer.writerow([key[0], key[1], key[2], key[3], mean_rmsre, std_rmsre, std_mean_ratio])

    print(f"Aggregated RMSRE results saved to {output_file}")


# Run the aggregation function
aggregate_rmsre_data(input_file, output_file)
