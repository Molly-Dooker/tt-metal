import pandas as pd
from sklearn.linear_model import LinearRegression

# File paths
input_file = "tests/sweep_framework/perf_model/binary_add/iis_extracted.csv"
output_file = "tests/sweep_framework/perf_model/binary_add/iis_coefficients.csv"

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(input_file)

# List of unique combinations of INPUT_0_MEMORY and INPUT_1_MEMORY
input_combinations = df[["INPUT_0_MEMORY", "INPUT_1_MEMORY"]].drop_duplicates().reset_index()

# Prepare a list to store the results
coefficients = []

# Iterate over each unique combination of INPUT_0_MEMORY and INPUT_1_MEMORY
for _, group in input_combinations.iterrows():
    input_0_memory = group["INPUT_0_MEMORY"]
    input_1_memory = group["INPUT_1_MEMORY"]

    # Filter the data for this combination of memory types
    subset = df[(df["INPUT_0_MEMORY"] == input_0_memory) & (df["INPUT_1_MEMORY"] == input_1_memory)]

    print(subset)

    # Define the features (X) and the target (y)
    X = subset[["TILES PER CORE", "CORE COUNT", "GRID_X", "GRID_Y", "TILES"]]
    y = subset["DEVICE KERNEL DURATION [ns]"]

    # Fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Store the coefficients (intercept and slopes) with the input memory types
    coefficients.append(
        {
            "INPUT_0_MEMORY": input_0_memory,
            "INPUT_1_MEMORY": input_1_memory,
            "intercept": model.intercept_,
            "TILES_PER_CORE_coef": model.coef_[0],
            "CORE_COUNT_coef": model.coef_[1],
            "GRID_X_coef": model.coef_[2],
            "GRID_Y_coef": model.coef_[3],
            "TILES": model.coef_[4],
        }
    )

# Convert the results to a DataFrame
coefficients_df = pd.DataFrame(coefficients)

# Save the results to a CSV file
coefficients_df.to_csv(output_file, index=False)

print(f"Coefficients have been saved to {output_file}")
