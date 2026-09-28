# Experiment 2: RoBERTa vs BERT on SMS spam
# Both models are fully fine-tuned with identical settings, only the model changes.

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

here = Path(__file__).parent
data_file = here.parent / "spamdata_v2.csv"
results_dir = here / "results"

models = {
    "bert": "bert-base-uncased",   # lowercases everything
    "roberta": "roberta-base",     # keeps capitals, so "FREE" and "free" look different
}
seeds = [42, 43, 44]

# same settings as the fine-tuned setup in experiment 1
lr = 2e-5
batch_size = 32
epochs = 5
max_len = 64
split_seed = 2018

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    df = pd.read_csv(data_file)
    x_train, x_temp, y_train, y_temp = train_test_split(
        df["text"], df["label"], test_size=0.3, random_state=split_seed, stratify=df["label"])
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=split_seed, stratify=y_temp)
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def make_loader(tokenizer, texts, labels, shuffle=False):
    enc = tokenizer(texts.tolist(), max_length=max_len, padding="max_length", truncation=True)
    data = TensorDataset(torch.tensor(enc["input_ids"]),
                         torch.tensor(enc["attention_mask"]),
                         torch.tensor(labels.tolist()))
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)


def show_tokens():
    # handy for a slide: the two models split the same message differently
    msg = "WINNER!! Claim your FREE prize now, txt 80082"
    print(f"\nmessage: {msg}")
    for key, name in models.items():
        tokens = AutoTokenizer.from_pretrained(name).tokenize(msg)
        print(f"{key:8s} ({len(tokens)} tokens): {tokens}")


class SpamClassifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        hidden = self.encoder.config.hidden_size  # 768 for both base models

        self.head = nn.Sequential(
            nn.Linear(hidden, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 2),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, ids, mask):
        out = self.encoder(ids, attention_mask=mask)
        first_token = out.last_hidden_state[:, 0]  # [CLS] for BERT, <s> for RoBERTa
        return self.head(first_token)


def run_epoch(model, loader, loss_fn, optimizer=None, scheduler=None):
    """Trains if an optimizer is passed in, otherwise just evaluates."""
    train = optimizer is not None
    model.train(train)
    total_loss = 0
    preds, labels = [], []

    for ids, mask, y in loader:
        ids, mask, y = ids.to(device), mask.to(device), y.to(device)

        with torch.set_grad_enabled(train):
            out = model(ids, mask)
            loss = loss_fn(out, y)

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        total_loss += loss.item()
        preds.append(out.argmax(1).cpu())
        labels.append(y.cpu())

    return total_loss / len(loader), torch.cat(preds).numpy(), torch.cat(labels).numpy()


def run(key, seed):
    model_name = models[key]
    print(f"\n{key}, seed {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_loader = make_loader(tokenizer, x_train, y_train, shuffle=True)
    val_loader = make_loader(tokenizer, x_val, y_val)
    test_loader = make_loader(tokenizer, x_test, y_test)

    model = SpamClassifier(model_name).to(device)

    weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    loss_fn = nn.NLLLoss(weight=torch.tensor(weights, dtype=torch.float, device=device))

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=lr)
    steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    history = []
    best_loss, best_epoch, best_state = float("inf"), 0, None

    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss, _, _ = run_epoch(model, train_loader, loss_fn, optimizer, scheduler)
        epoch_time = time.time() - start
        val_loss, _, _ = run_epoch(model, val_loader, loss_fn)

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "epoch_time_s": epoch_time})
        print(f"  epoch {epoch}: train {train_loss:.4f}, val {val_loss:.4f} ({epoch_time:.1f}s)")

        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    _, preds, labels = run_epoch(model, test_loader, loss_fn)
    print(f"  best epoch: {best_epoch}")
    print(classification_report(labels, preds, target_names=["ham", "spam"], zero_division=0))

    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    results = {
        "setup": key,
        "model": model_name,
        "seed": seed,
        "settings": {"lr": lr, "epochs": epochs, "max_len": max_len},
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "trainable_params": sum(p.numel() for p in params),
        "best_epoch": best_epoch,
        "mean_epoch_time_s": np.mean([h["epoch_time_s"] for h in history]),
        "peak_gpu_mem_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None,
        "history": history,
        "test": {
            "accuracy": report["accuracy"],
            "spam_precision": report["1"]["precision"],
            "spam_recall": report["1"]["recall"],
            "spam_f1": report["1"]["f1-score"],
            "ham_precision": report["0"]["precision"],
            "ham_recall": report["0"]["recall"],
            "ham_f1": report["0"]["f1-score"],
            "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        },
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"{key}_seed{seed}.json", "w") as f:
        json.dump(results, f, indent=2)

    del model, best_state
    torch.cuda.empty_cache()


if __name__ == "__main__":
    show_tokens()
    for key in models:
        for seed in seeds:
            run(key, seed)