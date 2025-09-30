#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------- Preprocessing ----------
def clean_text(text: str):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = [w for w in text.split() if len(w) >= 2]
    return words


def vocabulary_building(tokenized_documents, max_size=5000):
    counter = Counter()
    for toks in tokenized_documents:
        counter.update(toks)
    most_common = counter.most_common(max_size)
    word_to_index = {w: i for i, (w, _) in enumerate(most_common)}
    return word_to_index


def docs_to_seq_and_bow(docs, word_to_idx, max_len=200):
    V = len(word_to_idx)
    N = len(docs)

    seq_tensor = torch.zeros((N, max_len), dtype=torch.long)
    bow_tensor = torch.zeros((N, V), dtype=torch.float32)

    for i, toks in enumerate(docs):
        idxs = [word_to_idx[w] for w in toks if w in word_to_idx]
        idxs = idxs[:max_len]
        seq_tensor[i, :len(idxs)] = torch.tensor(idxs, dtype=torch.long)

        for w in toks:
            if w in word_to_idx:
                bow_tensor[i, word_to_idx[w]] = 1.0

    return seq_tensor, bow_tensor


# ---------- Model ----------
class TextAutoencoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, embedding_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# ---------- Training ----------
def train_autoencoder(model, X, epochs=50, batch_size=32):
    dataset = TensorDataset(X)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    record = []
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for (Xb,) in dl:
            recon, _ = model(Xb)
            loss = criterion(recon, Xb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * Xb.size(0)
        avg_loss = total_loss / len(X)
        print(f"[Epoch {ep}] Loss = {avg_loss:.6f}")
        record.append({"epoch": ep, "loss": avg_loss})
    return record


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_papers", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    input_papers = args.input_papers
    output_dir = args.output_dir
    epochs = args.epochs
    batch_size = args.batch_size

    with open(input_papers, "r") as f:
        papers = json.load(f)

    abstracts = [p["abstract"] for p in papers if "abstract" in p and p["abstract"].strip()]
    tokenized = [clean_text(abs_) for abs_ in abstracts]

    vocab = vocabulary_building(tokenized, max_size=5000)
    _, bow_tensor = docs_to_seq_and_bow(tokenized, vocab)

    model = TextAutoencoder(vocab_size=len(vocab))
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params}")
    if n_params > 2_000_000:
        raise RuntimeError("Parameter budget exceeded")

    start_time = time.time()
    history = train_autoencoder(model, bow_tensor, epochs=epochs, batch_size=batch_size)
    end_time = time.time()

    model.eval()
    with torch.no_grad():
        recon, embeddings = model(bow_tensor)
        recon_error = torch.mean((recon - bow_tensor) ** 2, dim=1).tolist()

    os.makedirs(output_dir, exist_ok=True)

    torch.save({
        "state_dict": model.state_dict(),
        "vocab": vocab,
        "config": {"hidden_dim": 256, "embedding_dim": 64}
    }, os.path.join(output_dir, "model.pth"))

    emb_out = []
    for i, p in enumerate(papers[:len(embeddings)]):
        emb_out.append({
            "arxiv_id": p.get("arxiv_id", str(i)),
            "embedding": embeddings[i].tolist(),
            "reconstruction_error": recon_error[i]
        })
    with open(os.path.join(output_dir, "embeddings.json"), "w") as f:
        json.dump(emb_out, f, indent=2)

    with open(os.path.join(output_dir, "vocabulary.json"), "w") as f:
        json.dump(vocab, f, indent=2)

    log = {
        "train_time_sec": end_time - start_time,
        "num_papers": len(papers),
        "history": history,
        "n_params": n_params,
        "epochs": epochs,
        "batch_size": batch_size
    }
    with open(os.path.join(output_dir, "training_log.json"), "w") as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    main()
