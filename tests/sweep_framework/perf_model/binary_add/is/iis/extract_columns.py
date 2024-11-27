import csv
import re  # Regular expressions to extract numbers


def extract_data(file_path):
    # Open the input file
    with open(file_path, "r") as f:
        reader = csv.DictReader(f)

        # List to store extracted data
        extracted_data = []

        # Process each row
        for row in reader:
            # Extract required fields from each row
            device_kernel_duration = row.get("DEVICE KERNEL DURATION [ns]", None)
            input_0_x = int(row.get("INPUT_0_X", 0))  # Ensure input_0_x is an integer
            input_0_y = int(row.get("INPUT_0_Y", 0))  # Ensure input_0_y is an integer
            input_0_memory = row.get("INPUT_0_MEMORY", None).replace("DEV_0_", "")
            input_1_memory = row.get("INPUT_1_MEMORY", None).replace("DEV_0_", "")
            output_0_memory = row.get("OUTPUT_0_MEMORY", None).replace("DEV_0_", "")
            attributes = row.get("ATTRIBUTES", "")

            tiles = (input_0_x * input_0_y) // 1024

            # Collect extracted information
            extracted_data.append(
                {
                    "DEVICE KERNEL DURATION [ns]": device_kernel_duration,
                    "INPUT_0_X": input_0_x,
                    "INPUT_0_Y": input_0_y,
                    "INPUT_0_MEMORY": input_0_memory,
                    "INPUT_1_MEMORY": input_1_memory,
                    "OUTPUT_0_MEMORY": output_0_memory,
                    "TILES": tiles,
                }
            )

    return extracted_data


def read_sharding_configurations(file_path):
    # Open the sharding configuration CSV file (no header)
    with open(file_path, "r") as f:
        reader = csv.reader(f)

        # List to store sharding configuration data
        sharding_data = []

        for row in reader:
            # Assuming the order of columns is shard_x, shard_y, start_x, start_y
            shard_x = int(row[0])  # First column: shard_x
            shard_y = int(row[1])  # Second column: shard_y
            start_x = int(row[2])  # Third column: start_x
            start_y = int(row[3])  # Fourth column: start_y

            sharding_data.append({"shard_x": shard_x, "shard_y": shard_y, "start_x": start_x, "start_y": start_y})

    return sharding_data


def save_to_csv(output_file_path, extracted_data, sharding_data):
    # Define the fieldnames for the CSV (including the new sharding columns)
    fieldnames = [
        "DEVICE KERNEL DURATION [ns]",
        "INPUT_0_X",
        "INPUT_0_Y",
        "INPUT_0_MEMORY",
        "INPUT_1_MEMORY",
        "OUTPUT_0_MEMORY",
        "TILES",
        "shard_x",
        "shard_y",
        "start_x",
        "start_y",
    ]

    # Ensure the lengths of extracted_data and sharding_data match
    if len(extracted_data) != len(sharding_data):
        raise ValueError("The number of rows in the performance data and sharding data do not match!")

    # Combine extracted data with sharding data row by row
    for i, row in enumerate(extracted_data):
        row.update(sharding_data[i])  # Add sharding information to each row

    # Write the combined data to a new CSV file
    with open(output_file_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header
        writer.writeheader()

        # Write the combined data
        writer.writerows(extracted_data)


def main():
    # Input file paths
    input_file_path = "/localdev/skrstic/tt-metal/generated/profiler/reports/2024_11_26_09_05_08/ops_perf_results_2024_11_26_09_05_08.csv"
    sharding_file_path = "tests/sweep_framework/perf_model/binary_add/is/iis/sharding_configurations.csv"

    # Output file path
    output_file_path = "tests/sweep_framework/perf_model/binary_add/is/iis/extracted_moving_rectangle_with_sharding.csv"

    # Extract data from the performance results CSV
    extracted_data = extract_data(input_file_path)

    # Read the sharding configurations
    sharding_data = read_sharding_configurations(sharding_file_path)

    # Save extracted data with sharding information to a new CSV
    save_to_csv(output_file_path, extracted_data, sharding_data)

    print(f"Data has been successfully extracted and saved to {output_file_path}")


if __name__ == "__main__":
    main()
