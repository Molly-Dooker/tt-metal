import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the predictions file
predictions_file = "tests/sweep_framework/perf_model/binary_add/sss/predictions_output.csv"
df = pd.read_csv(predictions_file)

# Calculate the relative error for each row
df["relative_error"] = np.abs(df["real_time"] - df["estimated_time"]) / df["real_time"]

# Group by num_tiles_per_core and calculate the mean relative error for each group
grouped = df.groupby("num_tiles_per_core")["relative_error"].mean()

# Calculate the RMSRE over the entire dataset (after averaging the errors per num_tiles_per_core)
rmsre = np.sqrt(np.mean(grouped**2))

# Plot RMSRE as a function of num_tiles_per_core
plt.figure(figsize=(10, 6))

# Plot the mean relative error as a function of num_tiles_per_core and connect the spots
plt.plot(grouped.index, grouped, "b-", marker="o", label="Mean Relative Error")

# Adding plot labels and title
plt.title(f"RMSRE vs. num_tiles_per_core (RMSRE = {rmsre:.4f})")
plt.xlabel("num_tiles_per_core")
plt.ylabel("Mean Relative Error")
plt.grid(True)
plt.legend()

# Save the plot as a PNG file
output_image_file = "tests/sweep_framework/perf_model/binary_add/sss/rmsre_vs_num_tiles_per_core.png"
plt.savefig(output_image_file, format="png")

# Close the plot to free memory
plt.close()

print(f"Plot saved as {output_image_file}")
