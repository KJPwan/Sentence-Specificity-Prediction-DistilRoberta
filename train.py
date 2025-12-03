import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from scipy.stats import pearsonr
from transformers import AutoTokenizer
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
            "label": torch.tensor(score, dtype=torch.float)
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

# function to update the teacher network
def update_teacher(student, teacher, alpha=0.999):
    with torch.no_grad():
        for t_param, s_param in zip(teacher.parameters(), student.parameters()):
            t_param.data.mul_(alpha).add_(s_param.data, alpha=1 - alpha)

# ---------------------------------------------------------
# Training Loop
# ---------------------------------------------------------
def train_epoch(student_model, teacher_model, loader, optimizer, loss_fn, device, consistency_weight=10.0):
    student_model.train()
    teacher_model.eval()  # teacher is not trained directly

    losses = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        # Student forward
        student_preds = student_model(input_ids, mask)

        # Supervised loss
        supervised_loss = loss_fn(student_preds, labels)

        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_preds = teacher_model(input_ids, mask)

        # Consistency loss = MSE(student, teacher)
        consistency_loss = nn.MSELoss()(student_preds, teacher_preds)

        # Combine (paper uses a large weight; you can tune)
        loss = supervised_loss + consistency_weight * consistency_loss

        loss.backward()
        optimizer.step()

        # Update teacher using EMA
        update_teacher(student_model, teacher_model)

        losses.append(loss.item())

    return np.mean(losses)


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------
def evaluate(model, loader, device):
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

    mse = np.mean((np.array(preds_all) - np.array(labels_all)) ** 2)
    mae = mean_absolute_error(labels_all, preds_all)
    pr = pearsonr(preds_all, labels_all)[0]
    sr = spearmanr(preds_all, labels_all)[0]
    kt = kendalltau(preds_all, labels_all)[0]

    return mse, mae, pr, sr, kt


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="train.tsv")
    parser.add_argument("--valid_path", type=str, default="valid.tsv")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--save_path", type=str, default="best_model.pt")
    args, _ = parser.parse_known_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

    train_data = SpecificityDataset(args.train_path, tokenizer, args.max_len)
    valid_data = SpecificityDataset(args.valid_path, tokenizer, args.max_len)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size)
    model = RobertaRegressor().to(device)
    #-------------------------
    # Create teacher model: same architecture
    teacher_model = RobertaRegressor().to(device)
    teacher_model.load_state_dict(model.state_dict())

    # Teacher never gets gradients
    for p in teacher_model.parameters():
        p.requires_grad = False


    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_pr = -1

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, teacher_model, train_loader, optimizer, loss_fn, device)
        mse, mae, pr, sr, kt = evaluate(model, valid_loader, device)

        print(
            f"Epoch {epoch} | "
            f"Train Loss = {train_loss:.4f} | "
            f"MSE = {mse:.4f} | MAE = {mae:.4f} | "
            f"Pearson = {pr:.4f} | Spearman = {sr:.4f} | KendallTau = {kt:.4f}"
        )

        if pr > best_pr:
            best_pr = pr
            torch.save(model.state_dict(), args.save_path)
            print("Saved new best model (validation).")

    print("Training done. Best Pearson:", best_pr)


if __name__ == "__main__":
    main()
