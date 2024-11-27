import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv("tests/sweep_framework/perf_model/binary_add/iis_extracted.csv")

# Group by the specified columns
grouped = df.groupby(["INPUT_0_MEMORY", "INPUT_1_MEMORY", "CORE COUNT", "TILES PER CORE"])

# Calculate mean, std, and std/mean for each group
result = grouped["DEVICE KERNEL DURATION [ns]"].agg(mean="mean", std="std").reset_index()

# Calculate std/mean ratio, avoid division by zero
result["std/mean"] = result["std"] / result["mean"].replace(0, pd.NA)

# Save the result to a new CSV file
result.to_csv("output_statistics.csv", index=False)

print("Results have been saved to 'output_statistics.csv'.")
