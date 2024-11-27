import csv
import re
import argparse


def extract_core_count(attributes):
    """Extract core count from the ATTRIBUTES field."""

    # Regex to match and extract grid coordinates from the ATTRIBUTES string
    match = re.search(r"grid=\{\[\(x=(\d+);y=(\d+)\)\s*-\s*\(x=(\d+);y=(\d+)\)\]\}", attributes)

    if match:
        # Extract the x and y grid limits from the match
        x_start = int(match.group(1))  # start x value
        y_start = int(match.group(2))  # start y value
        x_end = int(match.group(3))  # end x value
        y_end = int(match.group(4))  # end y value

        # Calculate the number of cores as (x_end - x_start + 1) * (y_end - y_start + 1)
        core_count = (x_end - x_start + 1) * (y_end - y_start + 1)

        return core_count
    else:
        raise ValueError("Could not extract core count from attributes")


def process_csv(input_csv_path):
    """Process CSV file to update CORE_COUNT column."""
    # Read the CSV file
    with open(input_csv_path, mode="r", newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)

        # Prepare the header for the output CSV
        fieldnames = reader.fieldnames
        rows = []

        # Process each row in the CSV
        for row in reader:
            # Extract core count from the ATTRIBUTES field
            try:
                core_count = extract_core_count(row["ATTRIBUTES"])
                row["CORE COUNT"] = str(core_count)  # Override the CORE_COUNT field
                # print("done")
            except ValueError as e:
                print(f"Error processing row {row['ID']}: {e}")
            rows.append(row)

    # Write the updated data to the same CSV file (overwrite)
    with open(input_csv_path, mode="w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Update CORE_COUNT in CSV file based on ATTRIBUTES.")
    parser.add_argument("--file", required=True, help="Path to the input CSV file")

    args = parser.parse_args()

    # Call the processing function
    process_csv(args.file)

    print(f"Updated CSV file saved: {args.file}")
