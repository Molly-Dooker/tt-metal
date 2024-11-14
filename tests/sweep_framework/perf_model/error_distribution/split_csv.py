import csv
import os

# File paths
input_file = "generated/profiler/reports/2024_11_05_10_23_44/ops_perf_results_2024_11_05_10_23_44.csv"
output_folder = "tests/sweep_framework/perf_model/error_distribution/csvs"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Read the data from the input CSV file
with open(input_file, newline="", encoding="utf-8") as infile:
    reader = csv.reader(infile)
    rows = [row for row in reader if not row[0].startswith("(torch)")]  # Skip rows starting with "(torch)"

# Split the data into 10 separate CSVs based on row index
for i in range(10):
    # Create a new list for this CSV file's data (including the header)
    output_data = []

    # Add the header row to each file (assuming the first row is the header)
    output_data.append(rows[0])  # Assuming the first row is the header

    # Add the rows corresponding to (i, i+10, i+20, ...)
    for j in range(i + 1, len(rows), 10):
        output_data.append(rows[j])

    # Write the output data to a new CSV file (1.csv, 2.csv, ..., 10.csv)
    output_file = os.path.join(output_folder, f"{i+1}.csv")
    with open(output_file, mode="w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerows(output_data)

    print(f"File {i+1}.csv written successfully.")

print("All files have been split and saved.")
