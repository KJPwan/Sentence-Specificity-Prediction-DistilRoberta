import pandas as pd
from sklearn.utils import shuffle

def create_subsets(input_file="train_PBSDS.tsv", output_prefix="train_pbsds"):
    """
    Reads a TSV file, shuffles it, and creates subsets of 10%, 25%, and 50%.
    """
    print(f"Reading data from {input_file}...")
    try:
        df = pd.read_csv(input_file, sep='\t')
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return

    # Shuffle the dataframe
    df = shuffle(df, random_state=42)
    print("Data shuffled.")

    total_rows = len(df)
    subsets = {
        "10": int(total_rows * 0.10),
        "25": int(total_rows * 0.25),
        "50": int(total_rows * 0.50)
    }

    for percent, size in subsets.items():
        subset_df = df.head(size)
        output_filename = f"{output_prefix}_{percent}.tsv"
        print(f"Creating subset: {output_filename} with {size} rows...")
        subset_df.to_csv(output_filename, sep='\t', index=False)

    print("\nData subset creation complete.")

if __name__ == "__main__":
    create_subsets()

