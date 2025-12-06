import pandas as pd

def check_leakage(train_file="train_PBSDS.tsv", valid_file="valid_PBSDS.tsv"):
    """
    Checks for sentence overlap between a training and validation TSV file.
    """
    print(f"Checking for data leakage between '{train_file}' and '{valid_file}'...")

    try:
        train_df = pd.read_csv(train_file, sep='\t')
        valid_df = pd.read_csv(valid_file, sep='\t')
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure both '{train_file}' and '{valid_file}' exist.")
        return

    train_sentences = set(train_df['sentence'].astype(str).tolist())
    valid_sentences = set(valid_df['sentence'].astype(str).tolist())

    overlap = train_sentences.intersection(valid_sentences)

    print(f"\n--- Data Leakage Report ---")
    print(f"Total sentences in training set: {len(train_sentences)}")
    print(f"Total sentences in validation set: {len(valid_sentences)}")
    print(f"Number of overlapping sentences: {len(overlap)}")

    if len(valid_sentences) > 0:
        overlap_percentage = (len(overlap) / len(valid_sentences)) * 100
        print(f"Percentage of validation set that overlaps with training set: {overlap_percentage:.2f}%")
    else:
        print(f"Validation set is empty, cannot calculate overlap percentage.")

    if len(overlap) > 0:
        print("\nWARNING: Data leakage detected! This can lead to overly optimistic performance estimates.")
    else:
        print("\nNo data leakage detected based on exact sentence match.")

if __name__ == "__main__":
    check_leakage()
