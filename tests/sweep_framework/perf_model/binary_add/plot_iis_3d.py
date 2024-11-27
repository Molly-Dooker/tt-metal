import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Function to read the CSV and return a DataFrame
def read_data(file_path):
    return pd.read_csv(file_path)


# Function to plot the relationship between kernel duration, tiles per core, and core count
def plot_kernel_duration_vs_tiles_3d_interactive(data, output_path):
    # Group the data by INPUT_0_MEMORY and INPUT_1_MEMORY
    grouped_data = data.groupby(["INPUT_0_MEMORY", "INPUT_1_MEMORY"])

    # Create a subplot grid for the 3D plots
    n_groups = len(grouped_data)
    rows = (n_groups // 3) + 1  # Number of rows (3 subplots per row)
    cols = 3  # Number of columns per row (adjust this if needed)

    # Create subplots with shared x, y, z axes for consistency
    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=[[{"type": "scatter3d"}] * cols] * rows,
        subplot_titles=[f"{input_0_mem} - {input_1_mem}" for (input_0_mem, input_1_mem) in grouped_data],
    )

    # Iterate over each group and create 3D scatter plots
    for i, ((input_0_mem, input_1_mem), group) in enumerate(grouped_data):
        row = i // cols + 1  # Determine row in subplot grid
        col = i % cols + 1  # Determine column in subplot grid

        # Scatter3d for the group
        scatter = go.Scatter3d(
            x=group["TILES PER CORE"],
            y=group["CORE COUNT"],
            z=group["DEVICE KERNEL DURATION [ns]"],
            mode="markers",
            marker=dict(size=5, color="blue"),  # Set color of points
            name=f"{input_0_mem} - {input_1_mem}",
        )

        # Add scatter plot to the subplot
        fig.add_trace(scatter, row=row, col=col)

    # Update layout of the figure for better aesthetics
    fig.update_layout(
        title="Kernel Duration vs Tiles Per Core and Core Count (Interactive)",
        height=600 * rows,  # Adjust height based on the number of rows
        width=800,
        showlegend=False,
        scene=dict(xaxis_title="TILES PER CORE", yaxis_title="CORE COUNT", zaxis_title="DEVICE KERNEL DURATION [ns]"),
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
    output_file_path = output_directory + "kernel_duration_vs_tiles_and_cores_3d_interactive.html"

    # Read the data
    data = read_data(input_file_path)

    # Plot and save the interactive figure
    plot_kernel_duration_vs_tiles_3d_interactive(data, output_file_path)

    print(f"Interactive 3D plot saved as {output_file_path}")


if __name__ == "__main__":
    main()
