"""
train_binary_bt.py - Binary classification with backtranslation support

For datasets with binary labels (0 or 1) like PBSDS.

Features:
- Sigmoid activation (outputs bounded to [0, 1])
- BCELoss (Binary Cross-Entropy)
- Optional backtranslation augmentation
- Optional layer freezing

Usage:
    # Baseline (no backtranslation)
    python train_binary_bt.py --train_path train_PBSDS.tsv --valid_path valid_PBSDS.tsv

    # With backtranslation
    python train_binary_bt.py --train_path train_PBSDS.tsv --valid_path valid_PBSDS.tsv \
        --use_backtranslation --bt_lang es --bt_mode combine

    # With layer freezing
    python train_binary_bt.py --train_path train_PBSDS.tsv --valid_path valid_PBSDS.tsv \
        --freeze_layers 5

    # Both backtranslation + freezing
    python train_binary_bt.py --train_path train_PBSDS.tsv --valid_path valid_PBSDS.tsv \
        --use_backtranslation --bt_lang es --bt_mode combine --freeze_layers 5
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm


# ---------------------------------------------------------
# Backtranslation Module
# ---------------------------------------------------------
class BackTranslator:
    """Backtranslation for data augmentation."""
    
    def __init__(self, intermediate_lang='es', device=None):
        from transformers import MarianMTModel, MarianTokenizer
        
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
            
        print(f"BackTranslator using device: {self.device}")
        print(f"Loading backtranslation models (en <-> {intermediate_lang})...")
        
        self.tokenizer_forward = MarianTokenizer.from_pretrained(
            f'Helsinki-NLP/opus-mt-en-{intermediate_lang}'
        )
        self.model_forward = MarianMTModel.from_pretrained(
            f'Helsinki-NLP/opus-mt-en-{intermediate_lang}'
        ).to(self.device)
        
        self.tokenizer_backward = MarianTokenizer.from_pretrained(
            f'Helsinki-NLP/opus-mt-{intermediate_lang}-en'
        )
        self.model_backward = MarianMTModel.from_pretrained(
            f'Helsinki-NLP/opus-mt-{intermediate_lang}-en'
        ).to(self.device)
        
        self.model_forward.eval()
        self.model_backward.eval()
        print("Backtranslation models loaded!")
    
    def backtranslate(self, text):
        if not text or not text.strip():
            return text
            
        with torch.no_grad():
            inputs = self.tokenizer_forward(
                text, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(self.device)
            translated = self.model_forward.generate(**inputs, max_length=512)
            intermediate = self.tokenizer_forward.decode(translated[0], skip_special_tokens=True)
            
            inputs = self.tokenizer_backward(
                intermediate, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(self.device)
            back_translated = self.model_backward.generate(**inputs, max_length=512)
            result = self.tokenizer_backward.decode(back_translated[0], skip_special_tokens=True)
            
        return result
    
    def backtranslate_batch(self, texts, show_progress=True):
        results = []
        iterator = tqdm(texts, desc="Backtranslating") if show_progress else texts
        for text in iterator:
            try:
                results.append(self.backtranslate(text))
            except Exception as e:
                print(f"Error: {e}")
                results.append(text)
        return results


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------
class BinaryDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=64):
        self.sentences = []
        self.labels = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(path, "r", encoding="utf8") as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                self.sentences.append(parts[0])
                self.labels.append(float(parts[1]))
        
        print(f"Loaded {len(self.sentences)} samples from {path}")

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.sentences[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.float)
        }


class AugmentedBinaryDataset(Dataset):
    """Dataset with backtranslation augmentation support."""
    
    def __init__(self, path, tokenizer, max_len=64, 
                 backtranslator=None, bt_mode='none', cache_path=None):
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Load original data
        original_sentences = []
        original_labels = []
        
        with open(path, "r", encoding="utf8") as f:
            next(f)
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                original_sentences.append(parts[0])
                original_labels.append(float(parts[1]))
        
        print(f"Loaded {len(original_sentences)} sentences from {path}")
        
        # Apply backtranslation if requested
        if bt_mode != 'none' and backtranslator is not None:
            if cache_path and os.path.exists(cache_path):
                print(f"Loading cached backtranslations from {cache_path}...")
                with open(cache_path, 'r', encoding='utf8') as f:
                    bt_sentences = [line.strip() for line in f]
            else:
                print(f"Backtranslating {len(original_sentences)} sentences...")
                bt_sentences = backtranslator.backtranslate_batch(original_sentences)
                
                if cache_path:
                    print(f"Saving cache to {cache_path}...")
                    with open(cache_path, 'w', encoding='utf8') as f:
                        for sent in bt_sentences:
                            f.write(sent + '\n')
            
            if bt_mode == 'replace':
                self.sentences = bt_sentences
                self.labels = original_labels
                print(f"Mode 'replace': Using {len(self.sentences)} backtranslated sentences")
            elif bt_mode == 'combine':
                self.sentences = original_sentences + bt_sentences
                self.labels = original_labels + original_labels
                print(f"Mode 'combine': Using {len(self.sentences)} sentences")
            else:
                self.sentences = original_sentences
                self.labels = original_labels
        else:
            self.sentences = original_sentences
            self.labels = original_labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.sentences[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.float)
        }


# ---------------------------------------------------------
# Model - WITH SIGMOID for binary classification
# ---------------------------------------------------------
class RobertaBinaryClassifier(nn.Module):
    def __init__(self, freeze_layers=0):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("distilroberta-base")
        self.hidden = nn.Linear(768, 256)
        self.act = nn.ReLU()
        self.regressor = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()  # Bound output to [0, 1]
        
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)
    
    def _freeze_layers(self, freeze_layers):
        # Freeze embeddings
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False
        
        # Freeze transformer layers
        for i in range(min(freeze_layers, 6)):
            for param in self.encoder.encoder.layer[i].parameters():
                param.requires_grad = False
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Frozen {freeze_layers} layers. Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        x = self.act(self.hidden(cls))
        out = self.regressor(x)
        out = self.sigmoid(out)  # Apply sigmoid
        return out.squeeze(1)


def update_teacher(student, teacher, alpha=0.999):
    with torch.no_grad():
        for t_param, s_param in zip(teacher.parameters(), student.parameters()):
            t_param.data.mul_(alpha).add_(s_param.data, alpha=1 - alpha)


# ---------------------------------------------------------
# Training Loop
# ---------------------------------------------------------
def train_epoch(student_model, teacher_model, loader, optimizer, loss_fn, device, consistency_weight=10.0):
    student_model.train()
    teacher_model.eval()
    losses = []

    for batch in tqdm(loader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        student_preds = student_model(input_ids, mask)
        supervised_loss = loss_fn(student_preds, labels)

        with torch.no_grad():
            teacher_preds = teacher_model(input_ids, mask)
        consistency_loss = nn.MSELoss()(student_preds, teacher_preds)

        loss = supervised_loss + consistency_weight * consistency_loss
        loss.backward()
        optimizer.step()

        update_teacher(student_model, teacher_model)
        losses.append(loss.item())

    return np.mean(losses)


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------
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
    
    # Convert to binary
    preds_binary = (preds_all > threshold).astype(int)
    labels_binary = labels_all.astype(int)
    
    metrics = {
        'accuracy': accuracy_score(labels_binary, preds_binary),
        'precision': precision_score(labels_binary, preds_binary, zero_division=0),
        'recall': recall_score(labels_binary, preds_binary, zero_division=0),
        'f1': f1_score(labels_binary, preds_binary, zero_division=0),
        'pred_mean': preds_all.mean(),
        'pred_std': preds_all.std()
    }
    
    return metrics


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument("--train_path", type=str, default="train.tsv")
    parser.add_argument("--valid_path", type=str, default="valid.tsv")
    
    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--consistency_weight", type=float, default=10.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    
    # Freezing
    parser.add_argument("--freeze_layers", type=int, default=0)
    
    # Backtranslation
    parser.add_argument("--use_backtranslation", action="store_true")
    parser.add_argument("--bt_lang", type=str, default="es", choices=["es", "de", "fr"])
    parser.add_argument("--bt_mode", type=str, default="combine", choices=["replace", "combine"])
    parser.add_argument("--bt_cache_dir", type=str, default="./bt_cache")
    
    # Output
    parser.add_argument("--save_path", type=str, default="best_model.pt")
    
    args, _ = parser.parse_known_args()

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

    # Backtranslation setup
    backtranslator = None
    if args.use_backtranslation:
        print(f"\nInitializing backtranslation (en <-> {args.bt_lang})...")
        backtranslator = BackTranslator(intermediate_lang=args.bt_lang, device=device)
        os.makedirs(args.bt_cache_dir, exist_ok=True)
    
    # Cache path
    train_cache_path = None
    if args.use_backtranslation:
        train_basename = os.path.basename(args.train_path).replace('.tsv', '')
        train_cache_path = os.path.join(args.bt_cache_dir, f"{train_basename}_bt_{args.bt_lang}.txt")

    # Load data
    print("\nLoading training data...")
    train_data = AugmentedBinaryDataset(
        args.train_path, tokenizer, args.max_len,
        backtranslator=backtranslator,
        bt_mode=args.bt_mode if args.use_backtranslation else 'none',
        cache_path=train_cache_path
    )
    
    print("\nLoading validation data...")
    valid_data = BinaryDataset(args.valid_path, tokenizer, args.max_len)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size)

    # Create models
    print(f"\nCreating model (freeze_layers={args.freeze_layers})...")
    model = RobertaBinaryClassifier(freeze_layers=args.freeze_layers).to(device)
    teacher_model = RobertaBinaryClassifier(freeze_layers=args.freeze_layers).to(device)
    teacher_model.load_state_dict(model.state_dict())
    
    for p in teacher_model.parameters():
        p.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    loss_fn = nn.BCELoss()  # Binary Cross-Entropy

    # Generate save path
    save_parts = [args.save_path.replace('.pt', '')]
    if args.freeze_layers > 0:
        save_parts.append(f"freeze{args.freeze_layers}")
    if args.use_backtranslation:
        save_parts.append(f"bt_{args.bt_lang}_{args.bt_mode}")
    save_path = "_".join(save_parts) + ".pt"

    best_f1 = -1

    print(f"\nModel will be saved to: {save_path}")
    print(f"Training for {args.epochs} epochs...")
    print(f"Loss: BCELoss | Activation: Sigmoid | Threshold: {args.threshold}")
    print("-" * 90)
    print(f"{'Epoch':>5} | {'Loss':>8} | {'Acc':>7} | {'Prec':>7} | {'Recall':>7} | {'F1':>7} | {'PredMean':>8}")
    print("-" * 90)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, teacher_model, train_loader, optimizer, loss_fn, device,
            consistency_weight=args.consistency_weight
        )
        metrics = evaluate(model, valid_loader, device, threshold=args.threshold)

        print(
            f"{epoch:>5} | "
            f"{train_loss:>8.4f} | "
            f"{metrics['accuracy']:>7.4f} | "
            f"{metrics['precision']:>7.4f} | "
            f"{metrics['recall']:>7.4f} | "
            f"{metrics['f1']:>7.4f} | "
            f"{metrics['pred_mean']:>8.4f}"
        )

        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model (F1: {metrics['f1']:.4f})")

    print("-" * 90)
    print(f"Training complete! Best F1: {best_f1:.4f}")
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()
