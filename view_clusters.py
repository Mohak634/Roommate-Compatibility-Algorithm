import pandas as pd

# Load the clustered data
try:
    df = pd.read_csv("clustered_roommates.csv")
    if "Cluster" not in df.columns:
        raise ValueError("Cluster column not found in dataset.")

    # Sort by cluster and save
    df_sorted = df.sort_values(by="Cluster")
    # pandas settings are local to with statement.
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.precision', 3,
                           ):
        print(df_sorted[["Full Name", "Cluster"]])  # Display first 20 records
except FileNotFoundError:
    print("Error: clustered_roommates.csv not found. Run training first.")
except Exception as e:
    print(f"An error occurred: {e}")
