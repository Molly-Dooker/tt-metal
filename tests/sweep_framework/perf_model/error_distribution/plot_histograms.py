import matplotlib.pyplot as plt
import csv
import os
from collections import defaultdict

# File path to the RMSRE mean results
rmsre_file = "tests/sweep_framework/perf_model/rmsre_mean_results.csv"
output_image_path = (
    "tests/sweep_framework/perf_model/error_distribution/rmsre_histograms_L1_DRAM.png"  # Path to save the image
)


# Function to read and process the RMSRE mean results from the CSV
def read_rmsre_data(rmsre_file):
    data = defaultdict(
        lambda: defaultdict(list)
    )  # Dictionary to store the data by (Input Memory, Output Memory, Input Datatype, Num Tiles)

    with open(rmsre_file, mode="r") as infile:
        reader = csv.DictReader(infile)

        for row in reader:
            input_memory = row["Input Memory"]
            output_memory = row["Output Memory"]
            input_datatype = row["Input Datatype"]
            num_tiles = int(row["Num Tiles"])
            mean_rmsre = float(row["Mean RMSRE (%)"])

            # Store the RMSRE values by (Input Memory, Output Memory, Input Datatype) -> Num Tiles -> list of RMSRE values
            data[(input_memory, output_memory, input_datatype)][num_tiles].append(mean_rmsre)

    return data


# Function to plot histograms for each (Input Memory, Output Memory, Input Datatype) combination
def plot_rmsre_histograms(data):
    # Define the unique combinations of Input Memory, Output Memory, and Input Datatype
    unique_combinations = sorted(
        set(data.keys())
    )  # Sorted list of (Input Memory, Output Memory, Input Datatype) tuples

    # Filter for combinations where Input Memory = "L1_INTERLEAVED" and Output Memory = "DRAM_INTERLEAVED"
    filtered_combinations = [
        comb for comb in unique_combinations if comb[0] == "L1_INTERLEAVED" and comb[1] == "DRAM_INTERLEAVED"
    ]

    if not filtered_combinations:
        print("No matching combinations found!")
        return

    # Define the unique Num Tiles values
    unique_num_tiles = sorted(
        set(
            num_tiles
            for _, _, _, num_tiles in [
                (input_memory, output_memory, input_datatype, num_tiles)
                for (input_memory, output_memory, input_datatype), tile_data in data.items()
                for num_tiles in tile_data.keys()
            ]
        )
    )

    # Number of subplots (rows x columns)
    num_rows = len(filtered_combinations)
    num_cols = len(unique_num_tiles)

    # Create the subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows), squeeze=False)

    # Flatten axes for easy indexing
    axes = axes.flatten()

    # Iterate over the filtered combinations of (Input Memory, Output Memory, Input Datatype)
    for row_index, (input_memory, output_memory, input_datatype) in enumerate(filtered_combinations):
        # Extract data for the current combination (Input Memory, Output Memory, Input Datatype)
        current_data = data[(input_memory, output_memory, input_datatype)]

        # Iterate over the unique Num Tiles values (columns)
        for col_index, num_tiles in enumerate(unique_num_tiles):
            ax = axes[row_index * num_cols + col_index]

            if num_tiles not in current_data:
                continue

            # Get the RMSRE values for the current combination and num_tiles
            rmsres = current_data[num_tiles]

            # Plot a histogram for RMSRE values
            ax.hist(rmsres, bins=10, alpha=0.7, edgecolor="black", color="lightblue", label=f"Num Tiles: {num_tiles}")
            ax.set_title(f"IM: {input_memory}\nOM: {output_memory}\nDT: {input_datatype}", fontsize=10)
            ax.set_xlabel("RMSRE (%)", fontsize=8)
            ax.set_ylabel("Frequency", fontsize=8)
            ax.legend()
            ax.grid(True)

    # Adjust layout to prevent overlap and make it more readable
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(hspace=1.2, wspace=1.0)  # Increase space between subplots

    # Save the figure as an image
    plt.savefig(output_image_path, bbox_inches="tight")  # Use bbox_inches to ensure tight bounding box
    print(f"Histograms saved to {output_image_path}")

    # Optionally, show the plot (you can disable this to just save the image)
    # plt.show()


# Main function to execute the plotting
def main():
    # Read the RMSRE data
    data = read_rmsre_data(rmsre_file)

    # Plot histograms for the filtered data
    plot_rmsre_histograms(data)


# Run the main function
if __name__ == "__main__":
    main()
