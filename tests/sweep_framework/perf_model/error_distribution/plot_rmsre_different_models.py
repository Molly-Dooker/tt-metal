import matplotlib.pyplot as plt


# Function to plot RMSRE vs num_tiles for each memory combination
def plot_rmsre_vs_num_tiles(aggregated_results, memory_combinations):
    # Create a dictionary to store RMSRE values for each memory combination
    rmsre_data = {mem_comb: [] for mem_comb in memory_combinations}

    # Filter data for each memory combination and prepare data for plotting
    for result in aggregated_results:
        input_memory, output_memory, input_datatype, num_tiles, actual_duration, predicted_duration, rmsre = result
        # Filter results for the specific memory combination
        if (input_memory, output_memory, input_datatype) in memory_combinations:
            rmsre_data[(input_memory, output_memory, input_datatype)].append((num_tiles, rmsre))

    # Plot the RMSRE vs num_tiles for each memory combination
    plt.figure(figsize=(12, 8))

    # Loop through each memory combination to plot the data
    for mem_comb in memory_combinations:
        data = rmsre_data[mem_comb]
        if data:  # Ensure there is data to plot
            num_tiles = [entry[0] for entry in data]
            rmsre = [entry[1] for entry in data]
            plt.plot(num_tiles, rmsre, marker="o", label=f"{mem_comb[0]} -> {mem_comb[1]} -> {mem_comb[2]}")

    # Set plot labels and title
    plt.xlabel("Number of Tiles")
    plt.ylabel("RMSRE")
    plt.title("RMSRE vs Number of Tiles for Different Memory Combinations")
    plt.legend(title="Memory Combinations")
    plt.grid(True)
    plt.xscale("log")  # Log scale for better visibility of the data
    plt.yscale("log")  # Log scale for better visibility of the RMSRE values

    # Show the plot
    plt.tight_layout()
    plt.show()


# Example of how you would call this function in your script
def main():
    # Assuming `aggregated_results` is already computed from the previous part of the code
    memory_combinations = [
        ("L1_INTERLEAVED", "L1_INTERLEAVED", "BFLOAT16"),
        ("L1_INTERLEAVED", "L1_INTERLEAVED", "BFLOAT4_B"),
        ("L1_INTERLEAVED", "L1_INTERLEAVED", "BFLOAT8_B"),
        ("L1_INTERLEAVED", "DRAM_INTERLEAVED", "BFLOAT16"),
        ("L1_INTERLEAVED", "DRAM_INTERLEAVED", "BFLOAT4_B"),
        ("L1_INTERLEAVED", "DRAM_INTERLEAVED", "BFLOAT8_B"),
        ("DRAM_INTERLEAVED", "L1_INTERLEAVED", "BFLOAT16"),
        ("DRAM_INTERLEAVED", "L1_INTERLEAVED", "BFLOAT4_B"),
        ("DRAM_INTERLEAVED", "L1_INTERLEAVED", "BFLOAT8_B"),
        ("DRAM_INTERLEAVED", "DRAM_INTERLEAVED", "BFLOAT16"),
        ("DRAM_INTERLEAVED", "DRAM_INTERLEAVED", "BFLOAT4_B"),
        ("DRAM_INTERLEAVED", "DRAM_INTERLEAVED", "BFLOAT8_B"),
    ]

    # Plot the RMSRE vs num_tiles for the specified memory combinations
    plot_rmsre_vs_num_tiles(aggregated_results, memory_combinations)


if __name__ == "__main__":
    main()
