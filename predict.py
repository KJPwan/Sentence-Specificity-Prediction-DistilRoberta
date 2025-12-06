
import argparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np

class RobertaBinaryClassifier(nn.Module):
    def __init__(self, num_frozen_layers=0):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("distilroberta-base")
        self.hidden = nn.Linear(768, 256)
        self.act = nn.ReLU()
        self.regressor = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

        if num_frozen_layers > 0:
            print(f"Freezing embeddings and first {num_frozen_layers} encoder layers.")
            for param in self.encoder.embeddings.parameters():
                param.requires_grad = False
            
            for i in range(num_frozen_layers):
                for param in self.encoder.encoder.layer[i].parameters():
                    param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden_state = outputs.last_hidden_state[:, 0, :]
        x = self.act(self.hidden(cls_hidden_state))
        out = self.regressor(x)
        out = self.sigmoid(out)
        return out.squeeze(1)

def predict(model, sentences, tokenizer, device, max_len=64):
    model.eval()
    predictions = []
    with torch.no_grad():
        for sent in sentences:
            enc = tokenizer(
                sent,
                truncation=True,
                padding="max_length",
                max_length=max_len,
                return_tensors="pt"
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            pred = model(input_ids, attention_mask)
            predictions.append(pred.item())
    return predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=64)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
    model = RobertaBinaryClassifier().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    with open(args.input_file, "r", encoding="utf8") as f:
        # Read sentences, assuming one sentence per line, and skip header if it exists
        lines = f.readlines()
        if lines[0].strip().lower().startswith("sentence"):
             lines = lines[1:]
        sentences = [line.strip().split('\t')[0] for line in lines]


    predictions = predict(model, sentences, tokenizer, device, args.max_len)

    with open(args.output_file, "w", encoding="utf8") as f:
        f.write("sentence\tprediction\n")
        for sent, pred in zip(sentences, predictions):
            f.write(f"{sent}\t{pred}\n")

if __name__ == "__main__":
    main()
