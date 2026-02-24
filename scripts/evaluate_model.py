"""
Model Evaluation Script
========================
Loads the best saved checkpoint and evaluates it on a held-out test split,
producing a full evaluation report.

Usage:
    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --checkpoint checkpoints/best_model.pt
    python scripts/evaluate_model.py --test-ratio 0.15 --batch-size 32

Output:
    logs/evaluation_report.json  — full metrics in JSON format
    Terminal                     — formatted human-readable summary

Metrics Computed:
    ├── Sentiment (3-class)
    │   ├── Per-class: Precision / Recall / F1 / Support
    │   ├── Macro F1, Weighted F1, Accuracy
    │   └── Confusion matrix (3x3)
    └── Sarcasm (binary)
        ├── Precision / Recall / F1 at default threshold (0.5)
        ├── Optimal threshold (swept 0.30–0.70 by best F1)
        ├── Metrics at optimal threshold
        └── Confusion matrix (2x2)
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("ERROR: PyTorch is required. Run: pip install torch")
    sys.exit(1)

try:
    from sklearn.metrics import (
        f1_score, accuracy_score, precision_score, recall_score,
        classification_report, confusion_matrix
    )
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

from utils.logger import get_logger
from utils.config_loader import load_config
from training.train_utils import find_optimal_threshold

logger = get_logger("evaluate_model")

SENTIMENT_NAMES = ["Negative", "Neutral", "Positive"]
SARCASM_NAMES   = ["Not Sarcastic", "Sarcastic"]


# ─────────────────────────────────────────────────────────────────────────────
# Inline dataset (same as resume_training.py — kept self-contained)
# ─────────────────────────────────────────────────────────────────────────────

class SimpleTokenizer:
    def __init__(self, max_vocab=20000):
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.max_vocab = max_vocab

    def build_vocab(self, texts):
        from collections import Counter
        counts = Counter()
        for t in texts:
            counts.update(t.lower().split())
        for word, _ in counts.most_common(self.max_vocab - 2):
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)

    def encode(self, text, max_length=128):
        tokens = text.lower().split()[:max_length]
        ids  = [self.vocab.get(w, 1) for w in tokens]
        ids  += [0] * (max_length - len(ids))
        mask = [1 if i > 0 else 0 for i in ids]
        return ids, mask


class EvalDataset(Dataset):
    SENTIMENT_MAP = {"negative": 0, "neutral": 1, "positive": 2, "0": 0, "1": 1, "2": 2}

    def __init__(self, rows, tokenizer, max_length=128):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        text = str(row.get("text", ""))
        sent_raw = str(row.get("sentiment_label", "1")).strip().lower()
        sarc_raw = str(row.get("sarcasm_label",  "0")).strip()
        sent_label = self.SENTIMENT_MAP.get(sent_raw, 1)
        sarc_label = int(sarc_raw) if sarc_raw in ("0", "1") else 0
        ids, mask = self.tokenizer.encode(text, self.max_length)
        return {
            "input_ids":       torch.tensor(ids,  dtype=torch.long),
            "attention_mask":  torch.tensor(mask, dtype=torch.long),
            "lid_token_ids":   torch.zeros(self.max_length, dtype=torch.long),
            "sentiment_label": torch.tensor(sent_label, dtype=torch.long),
            "sarcasm_label":   torch.tensor(sarc_label,  dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# BiLSTM model (must mirror the one used during training)
# ─────────────────────────────────────────────────────────────────────────────

class OfflineSentimentModel(torch.nn.Module):
    """
    BiLSTM model — matches exactly the architecture saved in checkpoints/best_model.pt.
    Layer names: embed, lid_embed, lstm, sentiment_head, sarcasm_head
    """
    def __init__(self, vocab_size=30004, embed_dim=128, hidden_dim=256,
                 num_lid_langs=10, num_sentiment_classes=3):
        super().__init__()
        self.embed     = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lid_embed = torch.nn.Embedding(num_lid_langs + 1, 16, padding_idx=0)
        self.lstm      = torch.nn.LSTM(
            embed_dim + 16, hidden_dim, num_layers=2,
            batch_first=True, dropout=0.3, bidirectional=True
        )
        self.dropout   = torch.nn.Dropout(0.4)
        feat_dim       = hidden_dim * 2  # 512
        self.sentiment_head = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, 128), torch.nn.Tanh(),
            torch.nn.Dropout(0.2), torch.nn.Linear(128, num_sentiment_classes),
        )
        self.sarcasm_head = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, 64), torch.nn.ReLU(),
            torch.nn.Dropout(0.2), torch.nn.Linear(64, 1),
        )

    def forward(self, input_ids, attention_mask=None, lid_token_ids=None):
        if lid_token_ids is None:
            lid_token_ids = torch.zeros_like(input_ids)
        lid_token_ids = lid_token_ids.clamp(0, 10)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        x = torch.cat([self.embed(input_ids), self.lid_embed(lid_token_ids)], dim=-1)
        out, _ = self.lstm(x)
        mask   = attention_mask.unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        pooled = self.dropout(pooled)
        return {
            "sentiment_logits": self.sentiment_head(pooled),
            "sarcasm_logit":    self.sarcasm_head(pooled),
        }



# ─────────────────────────────────────────────────────────────────────────────
# Evaluation logic
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(model, loader, device):
    """Run full inference pass, collect predictions and probabilities."""
    model.eval()
    sent_preds, sent_labels   = [], []
    sarc_probs, sarc_labels   = [], []

    with torch.no_grad():
        for batch in loader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            lid   = batch["lid_token_ids"].to(device)

            out = model(ids, mask, lid)
            s_pred = out["sentiment_logits"].argmax(dim=-1)
            s_prob = torch.sigmoid(out["sarcasm_logit"].squeeze(-1))

            sent_preds.extend(s_pred.cpu().tolist())
            sent_labels.extend(batch["sentiment_label"].tolist())
            sarc_probs.extend(s_prob.cpu().tolist())
            sarc_labels.extend(batch["sarcasm_label"].tolist())

    return sent_preds, sent_labels, sarc_probs, sarc_labels


def build_sentiment_report(preds, labels):
    report = {}
    for i, name in enumerate(SENTIMENT_NAMES):
        mask_pred  = [1 if p == i else 0 for p in preds]
        mask_label = [1 if l == i else 0 for l in labels]
        report[name] = {
            "precision": round(precision_score(mask_label, mask_pred, zero_division=0), 4),
            "recall":    round(recall_score(   mask_label, mask_pred, zero_division=0), 4),
            "f1":        round(f1_score(       mask_label, mask_pred, zero_division=0), 4),
            "support":   int(sum(mask_label)),
        }
    report["macro_f1"]    = round(f1_score(labels, preds, average="macro",    zero_division=0), 4)
    report["weighted_f1"] = round(f1_score(labels, preds, average="weighted", zero_division=0), 4)
    report["accuracy"]    = round(accuracy_score(labels, preds), 4)
    report["confusion_matrix"] = confusion_matrix(labels, preds).tolist()
    return report


def build_sarcasm_report(probs, labels, optimal_thresh):
    preds_05  = [1 if p >= 0.5          else 0 for p in probs]
    preds_opt = [1 if p >= optimal_thresh else 0 for p in probs]
    return {
        "default_threshold_0.5": {
            "precision":  round(precision_score(labels, preds_05, zero_division=0), 4),
            "recall":     round(recall_score(   labels, preds_05, zero_division=0), 4),
            "f1":         round(f1_score(       labels, preds_05, zero_division=0), 4),
            "accuracy":   round(accuracy_score( labels, preds_05), 4),
        },
        "optimal_threshold": optimal_thresh,
        "optimal_threshold_metrics": {
            "precision": round(precision_score(labels, preds_opt, zero_division=0), 4),
            "recall":    round(recall_score(   labels, preds_opt, zero_division=0), 4),
            "f1":        round(f1_score(       labels, preds_opt, zero_division=0), 4),
            "accuracy":  round(accuracy_score( labels, preds_opt), 4),
        },
        "class_distribution": {
            "sarcastic":     int(sum(labels)),
            "not_sarcastic": int(len(labels) - sum(labels)),
            "sarcasm_rate":  round(sum(labels) / len(labels), 4) if labels else 0.0,
        },
        "confusion_matrix_at_optimal": confusion_matrix(labels, preds_opt).tolist(),
    }


def print_report(report: dict):
    print("\n" + "═" * 65)
    print("  MULTILINGUAL SENTIMENT MODEL — EVALUATION REPORT")
    print("═" * 65)
    s = report["sentiment"]
    print(f"\n{'SENTIMENT TASK':}")
    print(f"  Accuracy:    {s['accuracy']:.4f}")
    print(f"  Macro F1:    {s['macro_f1']:.4f}")
    print(f"  Weighted F1: {s['weighted_f1']:.4f}")
    print(f"\n  {'Class':<15} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>8}")
    print("  " + "─" * 48)
    for name in SENTIMENT_NAMES:
        m = s[name]
        print(f"  {name:<15} {m['precision']:>10.4f} {m['recall']:>8.4f} "
              f"{m['f1']:>8.4f} {m['support']:>8}")
    print(f"\n  Confusion Matrix (rows=true, cols=pred):")
    cm = s["confusion_matrix"]
    header = f"         {'Neg':>7} {'Neu':>7} {'Pos':>7}"
    print(f"  {header}")
    for row_name, row in zip(["Neg", "Neu", "Pos"], cm):
        print(f"  {row_name:>5}   " + " ".join(f"{v:>7}" for v in row))

    r = report["sarcasm"]
    d = r["default_threshold_0.5"]
    o = r["optimal_threshold_metrics"]
    dist = r["class_distribution"]
    print(f"\n{'SARCASM TASK':}")
    print(f"  Class distribution: {dist['sarcastic']} sarcastic / "
          f"{dist['not_sarcastic']} non-sarcastic "
          f"({dist['sarcasm_rate']:.1%} sarcasm rate)")
    print(f"\n  At threshold 0.50:")
    print(f"    Precision={d['precision']:.4f}  Recall={d['recall']:.4f}  "
          f"F1={d['f1']:.4f}  Acc={d['accuracy']:.4f}")
    print(f"\n  At optimal threshold {r['optimal_threshold']:.2f}:")
    print(f"    Precision={o['precision']:.4f}  Recall={o['recall']:.4f}  "
          f"F1={o['f1']:.4f}  Acc={o['accuracy']:.4f}")

    print(f"\n  Recommended inference threshold: {r['optimal_threshold']:.2f}")
    print("═" * 65)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate best model checkpoint")
    p.add_argument("--checkpoint",  default=str(ROOT / "checkpoints" / "best_model.pt"))
    p.add_argument("--config-dir",  default=str(ROOT / "config"))
    p.add_argument("--test-ratio",  type=float, default=0.15)
    p.add_argument("--batch-size",  type=int,   default=32)
    p.add_argument("--output",      default=str(ROOT / "logs" / "evaluation_report.json"))
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()

    if not SKLEARN_OK:
        print("ERROR: scikit-learn required. Run: pip install scikit-learn")
        sys.exit(1)

    random.seed(args.seed)
    np.random.seed(args.seed)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # ── Load CSV data ──────────────────────────────────────────────────────
    csv_path = ROOT / "data" / "raw" / "combined_full.csv"
    if not csv_path.exists():
        csv_path = ROOT / "data" / "raw" / "sarcasm_news.csv"

    rows = []
    with open(csv_path, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            if row.get("text", "").strip():
                rows.append(row)

    logger.info(f"Loaded {len(rows):,} rows from {csv_path.name}")

    # Use fixed-seed split for reproducible test set
    random.shuffle(rows)
    split = int((1.0 - args.test_ratio) * len(rows))
    test_rows = rows[split:]
    logger.info(f"Test set: {len(test_rows):,} samples ({args.test_ratio:.0%})")

    # Build tokenizer on ALL data (same vocab as training)
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab([r["text"] for r in rows])

    test_ds = EvalDataset(test_rows, tokenizer)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # ── Load model ─────────────────────────────────────────────────────────
    logger.info(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_vocab = checkpoint.get("vocab_size", 30004)
    model = OfflineSentimentModel(
        vocab_size=saved_vocab, embed_dim=128, hidden_dim=256,
        num_lid_langs=10, num_sentiment_classes=3,
    )
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    prev_f1 = checkpoint.get("combined_f1", 0.0)
    logger.info(f"Model loaded | device={device} | prev F1={prev_f1:.4f}")

    # ── Run inference ──────────────────────────────────────────────────────
    logger.info("Running inference on test set...")
    sent_preds, sent_labels, sarc_probs, sarc_labels = run_inference(model, test_loader, device)

    # ── Compute metrics ────────────────────────────────────────────────────
    opt_thresh = find_optimal_threshold(sarc_probs, sarc_labels)

    report = {
        "metadata": {
            "checkpoint": str(ckpt_path),
            "test_samples": len(test_rows),
            "test_ratio": args.test_ratio,
            "evaluated_at": datetime.now().isoformat(),
            "device": str(device),
            "optimal_sarcasm_threshold": opt_thresh,
        },
        "sentiment": build_sentiment_report(sent_preds, sent_labels),
        "sarcasm":   build_sarcasm_report(sarc_probs, sarc_labels, opt_thresh),
    }

    # ── Save report ────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Evaluation report saved → {out_path}")

    # ── Print summary ──────────────────────────────────────────────────────
    print_report(report)
    print(f"\nFull report: {out_path}")


if __name__ == "__main__":
    main()
