import pandas as pd
import plotly.graph_objects as go


# Function to read the CSV and return a DataFrame
def read_data(file_path):
    return pd.read_csv(file_path)


# Function to plot the relationship between kernel duration, tiles per core, and core count
def plot_kernel_duration_vs_tiles_3d_interactive(data, input_0_mem, input_1_mem, output_path):
    # Filter data for the specific INPUT_0_MEMORY and INPUT_1_MEMORY combination
    filtered_data = data[(data["INPUT_0_MEMORY"] == input_0_mem) & (data["INPUT_1_MEMORY"] == input_1_mem)]

    # Create the 3D scatter plot
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=filtered_data["TILES"],
                y=filtered_data["CORE COUNT"],
                z=filtered_data["DEVICE KERNEL DURATION [ns]"],
                mode="markers",
                marker=dict(size=5, color="blue"),  # You can change the color if needed
                name=f"{input_0_mem} - {input_1_mem}",
            )
        ]
    )

    # Update layout of the figure
    fig.update_layout(
        title=f"Kernel Duration vs Tiles and Core Count ({input_0_mem} - {input_1_mem})",
        height=600,
        width=800,
        showlegend=False,
        scene=dict(xaxis_title="TILES", yaxis_title="CORE COUNT", zaxis_title="DEVICE KERNEL DURATION [ns]"),
        template="plotly_dark",  # Optional: choose a plotly template
    )

    # Save the interactive plot to an HTML file
    fig.write_html(output_path)


# Main function
def main():
    # Input file path
    input_file_path = "tests/sweep_framework/perf_model/binary_add/iis_extracted.csv"

    # Output directory for saving the plot
    output_directory = "tests/sweep_framework/perf_model/binary_add/images/"
    output_file_path = output_directory + "kernel_duration_vs_tiles_and_cores_L1_interleaved.html"

    # Read the data
    data = read_data(input_file_path)

    # Plot and save the interactive figure for "L1_INTERLEAVED" and "L1_INTERLEAVED"
    plot_kernel_duration_vs_tiles_3d_interactive(data, "L1_INTERLEAVED", "L1_INTERLEAVED", output_file_path)

    print(f"Interactive 3D plot saved as {output_file_path}")


if __name__ == "__main__":
    main()
