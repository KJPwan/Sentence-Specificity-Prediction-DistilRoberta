"""
test_metrics.py - Evaluate trained models with classification metrics

For binary classification tasks (like PBSDS: 0 = poor quality, 1 = good quality)

Outputs: F1, Precision, Recall, Accuracy

Usage:
    python test_metrics.py --model_path best_model.pt --test_path test_PBSDS.tsv
    
    # With custom threshold
    python test_metrics.py --model_path best_model.pt --test_path test_PBSDS.tsv --threshold 0.5
    
    # Test multiple models at once
    python test_metrics.py --model_path best_model_baseline.pt best_model_bt_es.pt --test_path test_PBSDS.tsv
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import (
    f1_score, 
    precision_score, 
    recall_score, 
    accuracy_score,
    classification_report,
    confusion_matrix
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
            "label": torch.tensor(score, dtype=torch.float)
        }


# ---------------------------------------------------------
# Model (must match training architecture)
# ---------------------------------------------------------
class RobertaRegressor(nn.Module):
    """Original regressor model (no sigmoid)."""
    def __init__(self, freeze_layers=0, freeze_embeddings=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("distilroberta-base")
        self.hidden = nn.Linear(768, 256)
        self.act = nn.ReLU()
        self.regressor = nn.Linear(256, 1)
        
        if freeze_layers > 0 or freeze_embeddings:
            self._freeze_layers(freeze_layers, freeze_embeddings)
    
    def _freeze_layers(self, freeze_layers, freeze_embeddings):
        if freeze_embeddings or freeze_layers > 0:
            for param in self.encoder.embeddings.parameters():
                param.requires_grad = False
        
        if freeze_layers > 0:
            layers_to_freeze = min(freeze_layers, 6)
            for i in range(layers_to_freeze):
                for param in self.encoder.encoder.layer[i].parameters():
                    param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        x = self.act(self.hidden(cls))
        out = self.regressor(x)
        return out.squeeze(1)


class RobertaBinaryClassifier(nn.Module):
    """Binary classifier model (with sigmoid)."""
    def __init__(self, freeze_layers=0):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("distilroberta-base")
        self.hidden = nn.Linear(768, 256)
        self.act = nn.ReLU()
        self.regressor = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
        
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)
    
    def _freeze_layers(self, freeze_layers):
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False
        for i in range(min(freeze_layers, 6)):
            for param in self.encoder.encoder.layer[i].parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        x = self.act(self.hidden(cls))
        out = self.regressor(x)
        out = self.sigmoid(out)
        return out.squeeze(1)


# ---------------------------------------------------------
# Evaluation Functions
# ---------------------------------------------------------
def get_predictions(model, loader, device):
    """Get raw predictions and labels from the model."""
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

    return np.array(preds_all), np.array(labels_all)


def calculate_metrics(preds_raw, labels, threshold=0.5):
    """
    Calculate classification metrics.
    
    Args:
        preds_raw: Raw model predictions (continuous values)
        labels: True labels (0 or 1)
        threshold: Threshold for converting predictions to binary
    
    Returns:
        Dictionary of metrics
    """
    # Convert predictions to binary
    preds_binary = (preds_raw > threshold).astype(int)
    labels_binary = labels.astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(labels_binary, preds_binary),
        'precision': precision_score(labels_binary, preds_binary, zero_division=0),
        'recall': recall_score(labels_binary, preds_binary, zero_division=0),
        'f1': f1_score(labels_binary, preds_binary, zero_division=0)
    }
    
    # Confusion matrix
    cm = confusion_matrix(labels_binary, preds_binary)
    
    return metrics, preds_binary, cm


def find_best_threshold(preds_raw, labels, thresholds=None):
    """Find the threshold that maximizes F1 score."""
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.05)
    
    best_f1 = 0
    best_threshold = 0.5
    
    for thresh in thresholds:
        preds_binary = (preds_raw > thresh).astype(int)
        labels_binary = labels.astype(int)
        f1 = f1_score(labels_binary, preds_binary, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    return best_threshold, best_f1


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, nargs='+', required=True,
                        help="Path(s) to trained model file(s)")
    parser.add_argument("--test_path", type=str, required=True,
                        help="Path to test TSV file")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for binary classification (default: 0.5)")
    parser.add_argument("--find_best_threshold", action="store_true",
                        help="Find the threshold that maximizes F1")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--freeze_layers", type=int, default=0,
                        help="Number of frozen layers (must match training)")
    parser.add_argument("--binary", action="store_true",
                        help="Use binary classifier model (with sigmoid). Use this for models trained with train_binary_bt.py")
    
    args = parser.parse_args()

    # Device setup
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

    # Load test data
    print(f"\nLoading test data from {args.test_path}...")
    test_data = SpecificityDataset(args.test_path, tokenizer, args.max_len)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)
    print(f"  Test samples: {len(test_data)}")

    # Print header
    print("\n" + "=" * 80)
    print("CLASSIFICATION METRICS")
    print("=" * 80)
    
    results = []
    
    # Evaluate each model
    for model_path in args.model_path:
        print(f"\n{'─' * 80}")
        print(f"Model: {model_path}")
        print(f"{'─' * 80}")
        
        # Load model (choose based on --binary flag)
        if args.binary:
            model = RobertaBinaryClassifier(freeze_layers=args.freeze_layers).to(device)
        else:
            model = RobertaRegressor(freeze_layers=args.freeze_layers).to(device)
        
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            print(f"  Error loading model: {e}")
            continue
        
        # Get predictions
        preds_raw, labels = get_predictions(model, test_loader, device)
        
        # Show prediction distribution
        print(f"\n  Raw prediction stats:")
        print(f"    Min: {preds_raw.min():.4f}")
        print(f"    Max: {preds_raw.max():.4f}")
        print(f"    Mean: {preds_raw.mean():.4f}")
        print(f"    Std: {preds_raw.std():.4f}")
        
        print(f"\n  Label distribution:")
        unique, counts = np.unique(labels.astype(int), return_counts=True)
        for u, c in zip(unique, counts):
            print(f"    Class {int(u)}: {c} ({100*c/len(labels):.1f}%)")
        
        # Find best threshold if requested
        if args.find_best_threshold:
            best_thresh, best_f1 = find_best_threshold(preds_raw, labels)
            print(f"\n  Best threshold: {best_thresh:.2f} (F1: {best_f1:.4f})")
            threshold = best_thresh
        else:
            threshold = args.threshold
        
        # Calculate metrics
        metrics, preds_binary, cm = calculate_metrics(preds_raw, labels, threshold)
        
        print(f"\n  Threshold: {threshold}")
        print(f"\n  Results:")
        print(f"    Accuracy:  {metrics['accuracy']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")
        print(f"    F1 Score:  {metrics['f1']:.4f}")
        
        print(f"\n  Confusion Matrix:")
        print(f"                 Predicted")
        print(f"                 0      1")
        print(f"    Actual 0  [{cm[0,0]:5d}  {cm[0,1]:5d}]")
        print(f"    Actual 1  [{cm[1,0]:5d}  {cm[1,1]:5d}]")
        
        # Store results
        results.append({
            'model': model_path,
            'threshold': threshold,
            **metrics
        })
    
    # Summary table if multiple models
    if len(results) > 1:
        print("\n" + "=" * 80)
        print("SUMMARY COMPARISON")
        print("=" * 80)
        print(f"\n{'Model':<50} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}")
        print("-" * 82)
        for r in results:
            model_name = r['model'][:48]
            print(f"{model_name:<50} {r['accuracy']:>8.4f} {r['precision']:>8.4f} {r['recall']:>8.4f} {r['f1']:>8.4f}")
    
    print("\n" + "=" * 80)
    print("Done!")


if __name__ == "__main__":
    main()
