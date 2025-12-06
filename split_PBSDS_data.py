import csv
import random

INPUT_PATH = "PBSDS_full.tsv"
TRAIN_PATH = "train_PBSDS.tsv"
VALID_PATH = "valid_PBSDS.tsv"
TEST_PATH = "test_PBSDS.tsv"

# ratios for splits
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1  # whatever is left after train+valid


def main():
    # Read all rows from PBSDS_full.tsv
    with open(INPUT_PATH, "r", encoding="utf8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = []
        for row in reader:
            sent = row.get("sentences")
            score = row.get("score")
            if sent is None or score is None:
                continue  # skip bad/empty lines
            rows.append({"sentences": sent, "score": score})

    if not rows:
        print("No data found in PBSDS_full.tsv. Check the file format.")
        return

    # Shuffle deterministically
    random.seed(42)
    random.shuffle(rows)

    n = len(rows)
    n_train = int(n * TRAIN_RATIO)
    n_valid = int(n * VALID_RATIO)
    # test gets the remainder
    n_test = n - n_train - n_valid

    train_rows = rows[:n_train]
    valid_rows = rows[n_train:n_train + n_valid]
    test_rows = rows[n_train + n_valid:]

    print(f"Total examples: {n}")
    print(f"Train: {len(train_rows)}, Valid: {len(valid_rows)}, Test: {len(test_rows)}")

    def write_tsv(path, subset):
        # Your training code just assumes: first col = sentence, second = score
        with open(path, "w", encoding="utf8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["sentence", "score"])  # header
            for r in subset:
                writer.writerow([r["sentences"], r["score"]])

    write_tsv(TRAIN_PATH, train_rows)
    write_tsv(VALID_PATH, valid_rows)
    write_tsv(TEST_PATH, test_rows)

    print(f"Wrote {TRAIN_PATH}, {VALID_PATH}, {TEST_PATH}")


if __name__ == "__main__":
    main()
