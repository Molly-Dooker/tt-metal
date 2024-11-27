import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

# File paths
estimations_file = (
    "/localdev/skrstic/tt-metal/generated/profiler/reports/2024_11_22_12_14_33/ops_perf_results_2024_11_22_12_14_33.csv"
)
output_dir = "tests/sweep_framework/perf_model/binary_add/sss/"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load the dataset
df = pd.read_csv(estimations_file)

print(df.columns)

# Compute num_tiles_per_core as per the formula
df["num_tiles_per_core"] = (df["INPUT_0_Y"] * df["INPUT_0_Y"]) // 1024 // df["CORE COUNT"]

# Take the mean of DEVICE KERNEL DURATION [ns] for each num_tiles_per_core
df_mean = df.groupby("num_tiles_per_core")["DEVICE KERNEL DURATION [ns]"].mean().reset_index()

# Prepare data for linear regression
X = df_mean[["num_tiles_per_core"]].values
y = df_mean["DEVICE KERNEL DURATION [ns]"].values

# Perform linear regression
model = LinearRegression()
model.fit(X, y)

# Get the model coefficients (intercept and slope)
intercept = model.intercept_
slope = model.coef_[0]

# Save the coefficients in a file
coeff_file = os.path.join(output_dir, "linear_model_coefficients.txt")
with open(coeff_file, "w") as f:
    f.write(f"Intercept: {intercept}\n")
    f.write(f"Slope: {slope}\n")

print(f"Model coefficients saved to {coeff_file}")

# Plot the data and the fitted linear model
plt.figure(figsize=(10, 6))

# Plot the actual data points
plt.scatter(df_mean["num_tiles_per_core"], df_mean["DEVICE KERNEL DURATION [ns]"], label="Measured", color="blue")

# Plot the linear fit
plt.plot(df_mean["num_tiles_per_core"], model.predict(X), label="Linear Fit", color="red", linestyle="--")

# Set plot labels and title
plt.xlabel("num_tiles_per_core", fontsize=12)
plt.ylabel("DEVICE KERNEL DURATION [ns]", fontsize=12)
plt.title("Linear Model: DEVICE KERNEL DURATION vs num_tiles_per_core", fontsize=14)
plt.legend()

# Save the plot
plot_file = os.path.join(output_dir, "linear_model_fit.png")
plt.tight_layout()
plt.savefig(plot_file)
print(f"Plot saved to {plot_file}")

# Show the plot
plt.show()
