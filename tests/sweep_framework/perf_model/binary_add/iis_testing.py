import pandas as pd

# File paths
input_file = "tests/sweep_framework/perf_model/binary_add/iis_extracted.csv"
coefficients_file = "tests/sweep_framework/perf_model/binary_add/iis_coefficients.csv"
output_file = "tests/sweep_framework/perf_model/binary_add/iis_with_estimates.csv"

# Load the extracted performance data
df = pd.read_csv(input_file)

# Load the coefficients from the previously saved CSV
coefficients_df = pd.read_csv(coefficients_file)


# Function to get the coefficients for a given combination of INPUT_0_MEMORY and INPUT_1_MEMORY
def get_coefficients(input_0_memory, input_1_memory):
    # Filter the coefficients DataFrame to find the correct row
    row = coefficients_df[
        (coefficients_df["INPUT_0_MEMORY"] == input_0_memory) & (coefficients_df["INPUT_1_MEMORY"] == input_1_memory)
    ]

    if not row.empty:
        # Extract the coefficients from the row
        intercept = row["intercept"].values[0]
        tiles_per_core_coef = row["TILES_PER_CORE_coef"].values[0]
        core_count_coef = row["CORE_COUNT_coef"].values[0]
        grid_x_coef = row["GRID_X_coef"].values[0]
        grid_y_coef = row["GRID_Y_coef"].values[0]

        return intercept, tiles_per_core_coef, core_count_coef, grid_x_coef, grid_y_coef
    else:
        # If no matching coefficients are found, return None
        return None


# Function to estimate the kernel duration based on the coefficients
def estimate_kernel_duration(row, intercept, tiles_per_core_coef, core_count_coef, grid_x_coef, grid_y_coef):
    X = row[["TILES PER CORE", "CORE COUNT", "GRID_X", "GRID_Y"]]
    estimated_duration = (
        intercept
        + tiles_per_core_coef * X["TILES PER CORE"]
        + core_count_coef * X["CORE COUNT"]
        + grid_x_coef * X["GRID_X"]
        + grid_y_coef * X["GRID_Y"]
    )
    return estimated_duration


# List to store the estimated durations and relative differences
estimated_durations = []
relative_differences = []

# Iterate over each row in the original DataFrame
for _, row in df.iterrows():
    input_0_memory = row["INPUT_0_MEMORY"]
    input_1_memory = row["INPUT_1_MEMORY"]

    # Get the coefficients for the current memory configuration
    coefficients = get_coefficients(input_0_memory, input_1_memory)

    if coefficients:
        # Unpack the coefficients
        intercept, tiles_per_core_coef, core_count_coef, grid_x_coef, grid_y_coef = coefficients

        # Estimate the kernel duration
        estimated_duration = estimate_kernel_duration(
            row, intercept, tiles_per_core_coef, core_count_coef, grid_x_coef, grid_y_coef
        )

        # Calculate the relative difference between the real and estimated kernel durations
        real_duration = row["DEVICE KERNEL DURATION [ns]"]
        relative_diff = abs(estimated_duration - real_duration) / real_duration

        # Append the results to the lists
        estimated_durations.append(estimated_duration)
        relative_differences.append(relative_diff)
    else:
        # If no coefficients were found, append NaN for the estimation and relative difference
        estimated_durations.append(None)
        relative_differences.append(None)

# Add the new columns to the DataFrame
df["Estimated DEVICE KERNEL DURATION"] = estimated_durations
df["Relative Difference"] = relative_differences

# Save the updated DataFrame to a new CSV file
df.to_csv(output_file, index=False)

print(f"Updated dataset with estimates and relative differences has been saved to {output_file}")
