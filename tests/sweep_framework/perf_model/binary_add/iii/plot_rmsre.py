import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the kernel duration estimations file
estimations_file = "tests/sweep_framework/perf_model/binary_add/iii/kernel_duration_estimations.csv"
df = pd.read_csv(estimations_file)


# Function to compute RMSRE for a given group
def compute_rmsre(group):
    # Calculate the absolute error relative to the real value
    relative_error = np.abs(group["estimated"] - group["real"]) / group["real"]
    # Calculate the RMSRE (Root Mean Square Relative Error)
    rmsre = np.sqrt(np.mean(np.square(relative_error)))
    return rmsre


# Group by memory configurations and num_tiles
grouped = df.groupby(["INPUT_0_MEMORY", "INPUT_1_MEMORY", "OUTPUT_0_MEMORY", "num tiles"])

# Calculate RMSRE for each group
rmsre_values = []
for group_values, group_df in grouped:
    rmsre = compute_rmsre(group_df)
    rmsre_values.append(
        {
            "INPUT_0_MEMORY": group_values[0],
            "INPUT_1_MEMORY": group_values[1],
            "OUTPUT_0_MEMORY": group_values[2],
            "num tiles": group_values[3],
            "RMSRE": rmsre,
        }
    )

# Create a DataFrame with RMSRE values
rmsre_df = pd.DataFrame(rmsre_values)

# Create the plot with subplots (4 per row)
num_plots = len(rmsre_df[["INPUT_0_MEMORY", "INPUT_1_MEMORY", "OUTPUT_0_MEMORY"]].drop_duplicates())
num_rows = (num_plots // 4) + (num_plots % 4 != 0)  # Calculate number of rows (4 per row)

# Create a larger figure with independent axes
fig, axes = plt.subplots(num_rows, 4, figsize=(20, 5 * num_rows))

# Flatten the axes array to iterate easily
axes = axes.flatten()

# Loop through each unique combination of INPUT_0_MEMORY, INPUT_1_MEMORY, OUTPUT_0_MEMORY
unique_combinations = rmsre_df[["INPUT_0_MEMORY", "INPUT_1_MEMORY", "OUTPUT_0_MEMORY"]].drop_duplicates()

for i, (index, row) in enumerate(unique_combinations.iterrows()):
    # Filter the data for this combination of memory configurations
    subset = rmsre_df[
        (rmsre_df["INPUT_0_MEMORY"] == row["INPUT_0_MEMORY"])
        & (rmsre_df["INPUT_1_MEMORY"] == row["INPUT_1_MEMORY"])
        & (rmsre_df["OUTPUT_0_MEMORY"] == row["OUTPUT_0_MEMORY"])
    ]

    # Plot RMSRE vs num_tiles for this combination
    ax = axes[i]
    ax.plot(
        subset["num tiles"],
        subset["RMSRE"],
        marker="o",
        label=f"{row['INPUT_0_MEMORY']}, {row['INPUT_1_MEMORY']}, {row['OUTPUT_0_MEMORY']}",
    )

    # Set the title and labels for the subplot
    ax.set_title(
        f"{row['INPUT_0_MEMORY'].replace('DEV_0_', '')}\n {row['INPUT_1_MEMORY'].replace('DEV_0_', '')}\n {row['OUTPUT_0_MEMORY'].replace('DEV_0_', '')}",
        fontsize=10,
    )
    ax.set_xlabel("Number of Tiles", fontsize=12)
    ax.set_ylabel("RMSRE", fontsize=12)
    ax.grid(True)

# Rotate x-axis labels for better readability
for ax in axes:
    for label in ax.get_xticklabels():
        label.set_rotation(45)

# Hide unused subplots (if any)
for i in range(num_plots, len(axes)):
    axes[i].axis("off")

# Adjust layout to avoid overlap
plt.tight_layout()
plt.subplots_adjust(hspace=0.5, wspace=0.3)

# Save the plot as a PNG image
plt.savefig("tests/sweep_framework/perf_model/binary_add/iii/rmsre_vs_num_tiles.png")

# Show the plot
plt.show()
