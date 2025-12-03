import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from scipy.stats import pearsonr
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_absolute_error


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------
class SpecificityDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=64):
        self.sentences = []
        self.scores = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(path, "r", encoding="utf8") as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue

                sent = parts[0]
                score = parts[1]

                self.sentences.append(sent)
                self.scores.append(float(score))


    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sent = self.sentences[idx]
        score = self.scores[idx]

        enc = self.tokenizer(
            sent,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(score, dtype=torch.float),
            "sentence": sent
        }


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------
class RobertaRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("distilroberta-base")
        self.hidden = nn.Linear(768, 256)
        self.act = nn.ReLU()
        self.regressor = nn.Linear(256, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        x = self.act(self.hidden(cls))
        out = self.regressor(x)
        return out.squeeze(1)


# ---------------------------------------------------------
# Evaluation function
# ----------------------------------------------------
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_absolute_error

def evaluate(model, loader, device):
    model.eval()
    preds_all, labels_all = [], []
    sents = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].cpu().numpy()

            preds = model(input_ids, mask).cpu().numpy()

            preds_all.extend(preds)
            labels_all.extend(labels)

    preds_arr = np.array(preds_all)
    labels_arr = np.array(labels_all)

    mse = np.mean((preds_arr - labels_arr) ** 2)
    mae = mean_absolute_error(labels_arr, preds_arr)
    pr = pearsonr(preds_arr, labels_arr)[0]
    sr = spearmanr(preds_arr, labels_arr)[0]
    kt = kendalltau(preds_arr, labels_arr)[0]

    return mse, mae, pr, sr, kt, preds_arr, labels_arr, sents



# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str, default="test.tsv")
    parser.add_argument("--model_path", type=str, default="best_model.pt")
    parser.add_argument("--max_len", type=int, default=64)
    args, _ = parser.parse_known_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

    test_data = SpecificityDataset(args.test_path, tokenizer, args.max_len)
    test_loader = DataLoader(test_data, batch_size=16)

    teacher_model = RobertaRegressor().to(device)
    teacher_model.load_state_dict(torch.load(args.model_path, map_location=device))
    teacher_model.eval()

    mse, mae, pr, sr, kt, preds_arr, labels_arr, sents = evaluate(teacher_model, test_loader, device)

    print(f"Test MSE       = {mse:.4f}")
    print(f"Test MAE       = {mae:.4f}")
    print(f"Pearson        = {pr:.4f}")
    print(f"Spearman       = {sr:.4f}")
    print(f"Kendall Tau    = {kt:.4f}")


if __name__ == "__main__":
    main()