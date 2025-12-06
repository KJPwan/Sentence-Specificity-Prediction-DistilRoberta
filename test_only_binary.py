
'''
Modified version of test.py:
modified only for switiching out the roberta regression model
with a binary classification model + sigmoid. 
'''

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import (
    mean_absolute_error,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)


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
class RobertaBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("distilroberta-base")
        self.hidden = nn.Linear(768, 256)
        self.act = nn.ReLU()
        self.regressor = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        x = self.act(self.hidden(cls))
        out = self.regressor(x)
        out = self.sigmoid(out)  # bound to [0, 1]
        return out.squeeze(1)


# ---------------------------------------------------------
# Evaluation function
# ----------------------------------------------------
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_absolute_error

def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    preds_all, labels_all = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].cpu().numpy()

            preds = model(input_ids, mask).cpu().numpy()
            preds_all.extend(preds)
            labels_all.extend(labels)

    preds_all = np.array(preds_all)
    labels_all = np.array(labels_all)

    # Binary classification metrics
    preds_binary = (preds_all > threshold).astype(int)
    labels_binary = labels_all.astype(int)

    # Regression-style metrics (optional but nice to have)
    mse = np.mean((preds_all - labels_all) ** 2)
    mae = mean_absolute_error(labels_all, preds_all)

    # Classification metrics
    accuracy = accuracy_score(labels_binary, preds_binary)
    precision = precision_score(labels_binary, preds_binary, zero_division=0)
    recall = recall_score(labels_binary, preds_binary, zero_division=0)
    f1 = f1_score(labels_binary, preds_binary, zero_division=0)

    return mse, mae, accuracy, precision, recall, f1



# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str, default="test.tsv")
    parser.add_argument("--model_path", type=str, default="best_model.pt")
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for binary classification",
    )
    args, _ = parser.parse_known_args()

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

    # Data
    test_data = SpecificityDataset(args.test_path, tokenizer, args.max_len)
    test_loader = DataLoader(test_data, batch_size=16)

    # Model (load the student model you saved in train_only_binary.py)
    model = RobertaBinaryClassifier().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    mse, mae, acc, prec, rec, f1 = evaluate(
        model,
        test_loader,
        device,
        threshold=args.threshold,
    )

    print(f"Test MSE    = {mse:.4f}")
    print(f"Test MAE    = {mae:.4f}")
    print(f"Test Acc    = {acc:.4f}")
    print(f"Test Prec   = {prec:.4f}")
    print(f"Test Recall = {rec:.4f}")
    print(f"Test F1     = {f1:.4f}")



if __name__ == "__main__":
    main()