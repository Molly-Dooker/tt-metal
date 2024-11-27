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
            core_count = int(row.get("CORE COUNT", 1))  # Ensure core_count is treated as an integer
            input_0_x = int(row.get("INPUT_0_X", 0))  # Ensure input_0_x is an integer
            input_0_y = int(row.get("INPUT_0_Y", 0))  # Ensure input_0_y is an integer
            input_0_memory = row.get("INPUT_0_MEMORY", None).replace("DEV_0_", "")
            input_1_memory = row.get("INPUT_1_MEMORY", None).replace("DEV_0_", "")
            output_0_memory = row.get("OUTPUT_0_MEMORY", None).replace("DEV_0_", "")
            attributes = row.get("ATTRIBUTES", "")

            # Extract GRID_X and GRID_Y from the ATTRIBUTES field
            grid_x, grid_y = 0, 0  # Default values
            match = re.search(r"grid=\{\[\(x=(\d+);y=(\d+)\) - \(x=(\d+);y=(\d+)\)\]\}", attributes)
            if match:
                grid_x_start = int(match.group(1))
                grid_y_start = int(match.group(2))
                grid_x_end = int(match.group(3))
                grid_y_end = int(match.group(4))

                # Compute the differences
                grid_x = abs(grid_x_end - grid_x_start + 1)
                grid_y = abs(grid_y_end - grid_y_start + 1)

            # Calculate TILES PER CORE
            if core_count > 0:
                tiles_per_core = (input_0_x * input_0_y) // 1024 // core_count
                tiles = (input_0_x * input_0_y) // 1024
            else:
                tiles_per_core = 0  # If core count is 0, set tiles per core to 0 (avoid division by zero)
                tiles = 0

            # Collect extracted information
            extracted_data.append(
                {
                    "DEVICE KERNEL DURATION [ns]": device_kernel_duration,
                    "CORE COUNT": core_count,
                    "INPUT_0_X": input_0_x,
                    "INPUT_0_Y": input_0_y,
                    "INPUT_0_MEMORY": input_0_memory,
                    "INPUT_1_MEMORY": input_1_memory,
                    "OUTPUT_0_MEMORY": output_0_memory,
                    "TILES PER CORE": tiles_per_core,
                    "TILES": tiles,
                    "GRID_X": grid_x,
                    "GRID_Y": grid_y,
                    "GRID_X_START": grid_x_start,
                    "GRID_Y_START": grid_y_start,
                }
            )

    return extracted_data


def save_to_csv(output_file_path, extracted_data):
    # extracted_data.sort(key=lambda x: (x['INPUT_0_MEMORY'], x['INPUT_1_MEMORY'], x['CORE COUNT'], x['TILES PER CORE'], x['OUTPUT_0_MEMORY'],))
    extracted_data.sort(
        key=lambda x: (
            x["TILES PER CORE"],
            x["DEVICE KERNEL DURATION [ns]"],
        )
    )
    # Define the fieldnames for the CSV
    fieldnames = [
        "DEVICE KERNEL DURATION [ns]",
        "CORE COUNT",
        "INPUT_0_X",
        "INPUT_0_Y",
        "INPUT_0_MEMORY",
        "INPUT_1_MEMORY",
        "OUTPUT_0_MEMORY",
        "TILES PER CORE",
        "TILES",
        "GRID_X",
        "GRID_Y",
        "GRID_X_START",
        "GRID_Y_START",
    ]

    # Write the extracted data to a new CSV file
    with open(output_file_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header
        writer.writeheader()

        # Write the data
        writer.writerows(extracted_data)


def main():
    # Input file path
    input_file_path = "/localdev/skrstic/tt-metal/generated/profiler/reports/2024_11_21_11_30_43/ops_perf_results_2024_11_21_11_30_43.csv"

    # Output file path
    output_file_path = "tests/sweep_framework/perf_model/binary_add/iis_extracted_moving_rectangle_dram_dram.csv"

    # Extract data from the file
    extracted_data = extract_data(input_file_path)

    # Save extracted data to CSV
    save_to_csv(output_file_path, extracted_data)

    print(f"Data has been successfully extracted and saved to {output_file_path}")


if __name__ == "__main__":
    main()
