import os
import csv
import pandas as pd

# Paths to your directories and files
fitting_data_file = "tests/sweep_framework/perf_model/fitting_data.csv"
coef_results_dir = "tests/sweep_framework/perf_model/error_distribution/coef_results"
output_file = "tests/sweep_framework/perf_model/error_distribution/predicted_kernel_durations_multiple.csv"

# List of num_tiles to predict for
num_tiles_list = [20, 200, 500, 5000]


# Function to read the fitting data coefficients (actual values)
def read_fitting_data():
    fitting_data = {}
    with open(fitting_data_file, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            input_datatype = row["INPUT_0_DATATYPE"]
            input_memory = row["INPUT_0_MEMORY"]
            output_memory = row["OUTPUT_0_MEMORY"]
            coef_a = float(row["COEFFICIENT_A"])
            coef_b = float(row["COEFFICIENT_B"])
            fitting_data[(input_memory, output_memory, input_datatype)] = (coef_a, coef_b)
    return fitting_data


# Function to read coefficient data from multiple CSV files in the directory
def read_coef_results():
    coef_data = {}
    for file_name in os.listdir(coef_results_dir):
        if file_name.endswith(".csv"):
            file_path = os.path.join(coef_results_dir, file_name)
            with open(file_path, mode="r") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    input_memory = row["INPUT_0_MEMORY"]
                    output_memory = row["OUTPUT_0_MEMORY"]
                    input_datatype = row["INPUT_0_DATATYPE"]
                    slope = float(row["SLOPE (A)"])
                    intercept = float(row["INTERCEPT (B)"])
                    coef_data[(input_memory, output_memory, input_datatype)] = (slope, intercept)
    return coef_data


# Function to calculate predicted kernel duration
def calculate_predicted_durations(fitting_data, coef_data, num_tiles_list):
    results = []

    for (input_memory, output_memory, input_datatype), (coef_a, coef_b) in fitting_data.items():
        # Get the slope and intercept from the coef_data
        if (input_memory, output_memory, input_datatype) in coef_data:
            slope, intercept = coef_data[(input_memory, output_memory, input_datatype)]

            # Actual kernel duration based on fitting data coefficients
            actual_duration = (
                coef_a * num_tiles_list[0] + coef_b
            )  # Taking the first num_tiles (20) for actual value calculation

            # Predict kernel duration for each num_tile value and store predictions
            for num_tiles in num_tiles_list:
                predicted_duration = slope * num_tiles + intercept  # Kernel duration prediction formula

                # Store results: Actual duration and 10 different predictions
                results.append(
                    (input_memory, output_memory, input_datatype, num_tiles, actual_duration, predicted_duration)
                )

    # Sort the results by the requested columns, ensuring rows with same INPUT_0_MEMORY, OUTPUT_0_MEMORY, INPUT_0_DATATYPE, NUM_TILES are grouped together
    results.sort(
        key=lambda x: (x[0], x[1], x[2], x[3])
    )  # Sort by INPUT_0_MEMORY, OUTPUT_0_MEMORY, INPUT_0_DATATYPE, NUM_TILES
    return results


# Function to save the results to CSV
def save_results_to_csv(results, output_file):
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow(
            [
                "INPUT_0_MEMORY",
                "OUTPUT_0_MEMORY",
                "INPUT_0_DATATYPE",
                "NUM_TILES",
                "ACTUAL_KERNEL_DURATION_NS",
                "PREDICTED_KERNEL_DURATION_NS",
            ]
        )
        # Write the data rows
        writer.writerows(results)


# Main function to orchestrate everything
def main():
    # Read fitting data and coefficient results
    fitting_data = read_fitting_data()
    coef_data = read_coef_results()

    # Calculate the predicted kernel durations
    results = calculate_predicted_durations(fitting_data, coef_data, num_tiles_list)

    # Save the results to a CSV file
    save_results_to_csv(results, output_file)
    print(f"Predicted kernel durations saved to: {output_file}")


if __name__ == "__main__":
    main()
