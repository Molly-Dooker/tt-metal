import argparse
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pickle


# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train linear regression models for kernel duration predictions.")
    parser.add_argument("--file", type=str, required=True, help="Path to the input CSV file")
    return parser.parse_args()


# Load and process the data from the CSV file
def load_data(file_path):
    df = pd.read_csv(file_path)
    df["NUM_INPUT_TILES"] = (df["INPUT_0_X"] * df["INPUT_0_Y"]) / 1024
    return df


# Function to train the model and save the results
def train_and_save_model(group_df, group_values, output_folder, all_coefficients, ax):
    # Extract relevant columns for modeling
    X = group_df[["NUM_INPUT_TILES"]].values.reshape(-1, 1)
    y = group_df["DEVICE KERNEL DURATION [ns]"].values

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Get the coefficients (intercept and slope)
    intercept = model.intercept_
    slope = model.coef_[0]

    # Save the coefficients in a list
    all_coefficients.append(
        {
            "INPUT_0_MEMORY": group_values[0],
            "INPUT_1_MEMORY": group_values[1],
            "OUTPUT_0_MEMORY": group_values[2],
            "Intercept": intercept,
            "Slope": slope,
        }
    )

    # Plot the results (training data and fit)
    ax.scatter(X, y, color="blue")
    ax.plot(X, model.predict(X), color="red")
    ax.set_xlabel("Number of Input Tiles")
    ax.set_ylabel("Device Kernel Duration [ns]")
    cleaned_group_values = [val.replace("DEV_0_", "") for val in group_values]
    ax.set_title(
        f"{cleaned_group_values[0]}\n {cleaned_group_values[1]}\n {cleaned_group_values[2]}", fontsize=10, pad=15
    )

    return model  # Return the trained model for later use


# Function to estimate kernel duration based on model coefficients
def estimate_kernel_duration(model, num_tiles):
    return model.intercept_ + model.coef_[0] * num_tiles


# Main function
def main():
    # Parse arguments
    args = parse_args()

    # Load the data from the CSV file
    df = load_data(args.file)

    # Group by the specified columns
    group_columns = ["INPUT_0_MEMORY", "INPUT_1_MEMORY", "OUTPUT_0_MEMORY"]
    grouped = df.groupby(group_columns)

    # Create output folder if not exists
    output_folder = "tests/sweep_framework/perf_model/binary_add/iii"
    os.makedirs(output_folder, exist_ok=True)

    # Prepare to save coefficients in a list
    all_coefficients = []

    # Number of groups
    num_groups = len(grouped)

    # Calculate number of rows and columns for the subplots
    ncols = 4
    nrows = (num_groups + ncols - 1) // ncols  # Calculate the required number of rows

    # Create a figure for all subplots (4 per row)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 4 * nrows))  # Increased height per row
    axes = axes.flatten()  # Flatten the 2D array of axes to make indexing easier

    # Train models and save coefficients for each group, while creating subplots
    models = {}  # Dictionary to store models for each group
    for i, (group_values, group_df) in enumerate(grouped):
        ax = axes[i]  # Access the i-th subplot
        model = train_and_save_model(group_df, group_values, output_folder, all_coefficients, ax)
        models[tuple(group_values)] = model  # Store the model for this group

    # Turn off unused subplots if there are any
    for i in range(num_groups, len(axes)):
        axes[i].axis("off")  # Hide the extra axes

    # Adjust the layout to avoid overlapping titles and labels
    plt.subplots_adjust(hspace=0.5, wspace=0.3)  # Adjust horizontal and vertical spacing
    plt.tight_layout()

    # Save the final plot as JPG
    plt.savefig(f"{output_folder}/all_models.jpg")

    # Save all coefficients in a CSV file
    coefficients_df = pd.DataFrame(all_coefficients)
    coefficients_df.to_csv(f"{output_folder}/model_coefficients.csv", index=False)

    print("Models, coefficients, and plots have been saved.")

    # Now, generate the new CSV file with estimations
    estimated_data = []

    # Loop through the data to make predictions using the models
    for group_values, group_df in grouped:
        model = models[tuple(group_values)]  # Get the model for this group
        num_tiles = group_df["NUM_INPUT_TILES"].values
        real_values = group_df["DEVICE KERNEL DURATION [ns]"].values

        # Estimate the kernel duration based on the model
        estimated_values = [estimate_kernel_duration(model, nt) for nt in num_tiles]

        # Add the real and estimated values to the output list
        for i in range(len(group_df)):
            estimated_data.append(
                {
                    "INPUT_0_MEMORY": group_values[0],
                    "INPUT_1_MEMORY": group_values[1],
                    "OUTPUT_0_MEMORY": group_values[2],
                    "num tiles": num_tiles[i],
                    "estimated": estimated_values[i],
                    "real": real_values[i],
                }
            )

    # Create a DataFrame from the estimated data and save it to a CSV
    estimated_df = pd.DataFrame(estimated_data)
    estimated_df.to_csv(f"{output_folder}/kernel_duration_estimations.csv", index=False)

    print("Estimations have been saved.")


# Run the main function
if __name__ == "__main__":
    main()
