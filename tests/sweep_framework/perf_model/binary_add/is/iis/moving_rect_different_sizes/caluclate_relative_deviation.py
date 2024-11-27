import pandas as pd

# Load the CSV data
csv_path = "tests/sweep_framework/perf_model/binary_add/is/iis/moving_rect_different_sizes/extracted_moving_rectangle_with_sharding.csv"
df = pd.read_csv(csv_path)

# Group by shard_x and shard_y
grouped = df.groupby(["shard_x", "shard_y"])

# Prepare a list to store the results
results = []

# Iterate through each group (grouped by shard_x and shard_y)
for (shard_x, shard_y), group in grouped:
    # Get the maximum DURATION [ns] in the group
    max_duration = group["DEVICE KERNEL DURATION [ns]"].max()

    # Calculate the relative deviation for each element in the group
    group["relative_deviation"] = (group["DEVICE KERNEL DURATION [ns]"] - max_duration).abs() / max_duration

    # Find the row with the max DURATION and extract start_x, start_y
    max_row = group.loc[group["DEVICE KERNEL DURATION [ns]"] == max_duration].iloc[0]
    start_x = max_row["start_x"]
    start_y = max_row["start_y"]

    # Calculate the mean relative deviation
    mean_relative_deviation = group["relative_deviation"].mean()

    # Append the results for the current group
    results.append(
        {
            "shard_x": shard_x,
            "shard_y": shard_y,
            "mean_relative_deviation": mean_relative_deviation,
            "start_x": start_x,
            "start_y": start_y,
        }
    )

# Create a DataFrame from the results
result_df = pd.DataFrame(results)

# Output the result
print(result_df)

# Optionally, save to a new CSV file
result_df.to_csv(
    "tests/sweep_framework/perf_model/binary_add/is/iis/moving_rect_different_sizes/relative_deviation_results.csv",
    index=False,
)
