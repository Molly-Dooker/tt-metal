import pandas as pd
import os

# Load data from input CSV file
input_csv_file = "filtered_memory_values_common.csv"
df = pd.read_csv(input_csv_file, header=None, names=["first", "second", "third"])

# Group by 'first' and 'third', then calculate the average of 'second'
df_aggregated = df.groupby(["first", "third"], as_index=False)["second"].mean()

# Define the output CSV file
output_csv_file = "all_machines_results.csv"

# Check if the output file exists
file_exists = os.path.isfile(output_csv_file)

# Append the aggregated results to the output file, creating it if it does not exist
df_aggregated.to_csv(output_csv_file, mode="a", index=False, header=not file_exists)

print(f"Aggregated data appended to {output_csv_file}")
