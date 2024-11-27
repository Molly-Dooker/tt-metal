import pandas as pd
import numpy as np

# Load the dataset
estimations_file = (
    "/localdev/skrstic/tt-metal/generated/profiler/reports/2024_11_22_14_18_02/ops_perf_results_2024_11_22_14_18_02.csv"
)
df = pd.read_csv(estimations_file)

# Preprocess the data
# Ensure that the required columns exist
df = df[["INPUT_0_X", "INPUT_0_Y", "CORE COUNT", "DEVICE KERNEL DURATION [ns]"]]
df = df.dropna()  # Drop any rows with missing values

# Calculate the num_tiles_per_core (assuming 'INPUT_0_Y' is used for the calculation)
df["num_tiles_per_core"] = (df["INPUT_0_Y"] * df["INPUT_0_X"]) // 1024 // df["CORE COUNT"]

# Load the coefficients from the linear_model_coefficients.txt file
coefficients_file = "tests/sweep_framework/perf_model/binary_add/sss/linear_model_coefficients.txt"

# Read the coefficients from the file (assuming the format is 'Intercept: value' and 'Slope: value')
with open(coefficients_file, "r") as f:
    coefficients = f.readlines()

# Extract the intercept and slope from the file (strip any whitespace and convert to float)
intercept = float(coefficients[0].strip().split(":")[1])  # Extract the intercept value
slope = float(coefficients[1].strip().split(":")[1])  # Extract the slope value

# Make predictions using the linear model (y = slope * num_tiles_per_core + intercept)
df["estimated_time"] = slope * df["num_tiles_per_core"] + intercept

# Calculate the real time (just for the sake of example, let's use the 'DEVICE KERNEL DURATION [ns]' column as the real time)
df["real_time"] = df["DEVICE KERNEL DURATION [ns]"]

# Now, create a new CSV file with the required columns
output_file = "tests/sweep_framework/perf_model/binary_add/sss/predictions_output.csv"

df_output = df[["INPUT_0_X", "INPUT_0_Y", "CORE COUNT", "num_tiles_per_core", "real_time", "estimated_time"]]

# Save to a new CSV file
df_output.to_csv(output_file, index=False)

print(f"Predictions saved to {output_file}")
