"""
train_binary.py - Training script for BINARY classification tasks

Modified from train.py to use:
- Sigmoid activation (outputs bounded to [0, 1])
- BCELoss (Binary Cross-Entropy) instead of MSELoss
- Optional MixText data augmentation

Use this for datasets with binary labels (0 or 1) like PBSDS.

Usage:
    python train_binary.py --train_path train_PBSDS.tsv --valid_path valid_PBSDS.tsv --epochs 10 --use_mixtext
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_absolute_error, f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm
import matplotlib.pyplot as plt


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
# Model - WITH SIGMOID for binary classification and refactored for MixText
# ---------------------------------------------------------
class RobertaBinaryClassifier(nn.Module):
    def __init__(self, num_frozen_layers=0):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("distilroberta-base")
        self.hidden = nn.Linear(768, 256)
        self.act = nn.ReLU()
        self.regressor = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

        # Freeze embeddings and specified number of layers
        if num_frozen_layers > 0:
            print(f"Freezing embeddings and first {num_frozen_layers} encoder layers.")
            # Freeze embeddings
            for param in self.encoder.embeddings.parameters():
                param.requires_grad = False
            
            # Freeze encoder layers
            for i in range(num_frozen_layers):
                for param in self.encoder.encoder.layer[i].parameters():
                    param.requires_grad = False

    def forward_encoder(self, input_ids, attention_mask):
        """Extracts the [CLS] token's hidden state."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden_state = outputs.last_hidden_state[:, 0, :]
        return cls_hidden_state

    def forward_classifier(self, cls_hidden_state):
        """Takes a hidden state and produces a final prediction."""
        x = self.act(self.hidden(cls_hidden_state))
        out = self.regressor(x)
        out = self.sigmoid(out)
        return out.squeeze(1)

    def forward(self, input_ids, attention_mask):
        """Standard forward pass."""
        cls_hidden_state = self.forward_encoder(input_ids, attention_mask)
        return self.forward_classifier(cls_hidden_state)


# function to update the teacher network
def update_teacher(student, teacher, alpha=0.999):
    with torch.no_grad():
        for t_param, s_param in zip(teacher.parameters(), student.parameters()):
            t_param.data.mul_(alpha).add_(s_param.data, alpha=1 - alpha)

def set_dropout_to_train_mode(model):
    """Recursively sets dropout layers to train mode within a model."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

# ---------------------------------------------------------
# Training Loop
# ---------------------------------------------------------
def train_epoch(student_model, teacher_model, loader, optimizer, loss_fn, device, args):
    student_model.train()
    teacher_model.eval()
    set_dropout_to_train_mode(teacher_model) # Ensure dropout is active for teacher's forward pass

    losses = []

    for batch in tqdm(loader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        # Student forward for supervised and consistency loss
        student_preds = student_model(input_ids, mask)
        supervised_loss = loss_fn(student_preds, labels)

        # Teacher forward (no grad) for consistency loss
        with torch.no_grad():
            teacher_preds = teacher_model(input_ids, mask)
        consistency_loss = nn.MSELoss()(student_preds, teacher_preds)

        # Base loss
        loss = supervised_loss + args.consistency_weight * consistency_loss

        # --- MixText Augmentation ---
        if args.use_mixtext:
            # Get hidden states from the student model
            hidden_1 = student_model.forward_encoder(input_ids, mask)

            # Create a shuffled version of the batch to mix with
            batch_size = input_ids.size(0)
            shuffle_indices = torch.randperm(batch_size)
            input_ids_2 = input_ids[shuffle_indices]
            mask_2 = mask[shuffle_indices]
            labels_2 = labels[shuffle_indices]

            hidden_2 = student_model.forward_encoder(input_ids_2, mask_2)

            # Generate mixing lambda and mix hidden states and labels
            lam = np.random.beta(args.mix_alpha, args.mix_alpha)
            mixed_hidden = lam * hidden_1 + (1 - lam) * hidden_2
            mixed_labels = lam * labels + (1 - lam) * labels_2

            # Get predictions on the mixed hidden state
            mixed_preds = student_model.forward_classifier(mixed_hidden)
            
            # Calculate MixText loss and add it to the total loss
            mix_loss = loss_fn(mixed_preds, mixed_labels)
            loss = loss + mix_loss

        loss.backward()
        optimizer.step()

        # Update teacher using EMA
        update_teacher(student_model, teacher_model)

        losses.append(loss.item())

    return np.mean(losses)


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------
def evaluate(model, loader, loss_fn, device, threshold=0.5):
    model.eval()
    preds_all, labels_all = [], []
    val_losses = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            preds = model(input_ids, mask)
            
            # Calculate validation loss
            loss = loss_fn(preds, labels)
            val_losses.append(loss.item())
            
            # Store preds and labels for metrics
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    preds_all = np.array(preds_all)
    labels_all = np.array(labels_all)
    
    # Calculate average validation loss
    val_loss = np.mean(val_losses)
    
    # Binary classification metrics
    preds_binary = (preds_all > threshold).astype(int)
    labels_binary = labels_all.astype(int)
    
    # Regression metrics (for comparison)
    mse = np.mean((preds_all - labels_all) ** 2)
    mae = mean_absolute_error(labels_all, preds_all)
    
    # Classification metrics
    accuracy = accuracy_score(labels_binary, preds_binary)
    precision = precision_score(labels_binary, preds_binary, zero_division=0)
    recall = recall_score(labels_binary, preds_binary, zero_division=0)
    f1 = f1_score(labels_binary, preds_binary, zero_division=0)

    return val_loss, mse, mae, accuracy, precision, recall, f1


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="train.tsv")
    parser.add_argument("--valid_path", type=str, default="valid.tsv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--consistency_weight", type=float, default=10.0)
    parser.add_argument("--save_path", type=str, default="best_model.pt")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for binary classification")
    # MixText arguments
    parser.add_argument("--use_mixtext", action="store_true", help="Enable MixText data augmentation")
    parser.add_argument("--mix_alpha", type=float, default=0.2, help="Alpha parameter for the Beta distribution in MixText")
    parser.add_argument("--save_plot", action="store_true", help="Save a plot of training and validation loss")
    parser.add_argument("--num_frozen_layers", type=int, default=0, help="Number of initial RoBERTa layers to freeze")

    args, _ = parser.parse_known_args()

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    # elif torch.backends.mps.is_available():
    #     device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

    # Load data
    print("\nLoading data...")
    train_data = SpecificityDataset(args.train_path, tokenizer, args.max_len)
    valid_data = SpecificityDataset(args.valid_path, tokenizer, args.max_len)
    print(f"  Train: {len(train_data)} samples")
    print(f"  Valid: {len(valid_data)} samples")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size)
    
    # Create models
    model = RobertaBinaryClassifier(num_frozen_layers=args.num_frozen_layers).to(device)
    teacher_model = RobertaBinaryClassifier(num_frozen_layers=args.num_frozen_layers).to(device)
    teacher_model.load_state_dict(model.state_dict())

    for p in teacher_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.BCELoss()  # Binary Cross-Entropy Loss

    best_f1 = -1
    train_loss_history = []
    val_loss_history = []

    print(f"\nTraining for {args.epochs} epochs...")
    print(f"Using BCELoss + Sigmoid activation")
    if args.use_mixtext:
        print(f"Using MixText augmentation with alpha = {args.mix_alpha}")
    print(f"Threshold: {args.threshold}")
    print("-" * 80)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, teacher_model, train_loader, optimizer, loss_fn, device, args
        )
        val_loss, mse, mae, accuracy, precision, recall, f1 = evaluate(
            model, valid_loader, loss_fn, device, threshold=args.threshold
        )
        
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        print(
            f"Epoch {epoch:2d} | "
            f"Train Loss = {train_loss:.4f} | "
            f"Val Loss = {val_loss:.4f} | "
            f"Acc = {accuracy:.4f} | "
            f"P = {precision:.4f} | "
            f"R = {recall:.4f} | "
            f"F1 = {f1:.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), args.save_path)
            print(f"  -> Saved new best model (F1: {f1:.4f})")

    print("-" * 80)
    print(f"Training complete! Best F1: {best_f1:.4f}")
    print(f"Model saved to: {args.save_path}")

    # Plot and save the loss curve if requested
    if args.save_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, args.epochs + 1), train_loss_history, label="Training Loss")
        plt.plot(range(1, args.epochs + 1), val_loss_history, label="Validation Loss")
        plt.title("Training and Validation Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        # Create a dynamic filename
        mix_str = f"_mixalpha_{args.mix_alpha}" if args.use_mixtext else ""
        plot_filename = f"loss_curve_lr_{args.lr}_cw_{args.consistency_weight}{mix_str}.png"
        
        plt.savefig(plot_filename)
        print(f"Loss curve saved to: {plot_filename}")




if __name__ == "__main__":
    main()
