"""
Resume Training Script
=====================
Loads the best saved model checkpoint and continues training for additional epochs.

Usage:
    python scripts/resume_training.py
    python scripts/resume_training.py --extra-epochs 5
    python scripts/resume_training.py --checkpoint checkpoints/best_model.pt --extra-epochs 10
    python scripts/resume_training.py --extra-epochs 5 --batch-size 8

Features:
    - Resumes from best_model.pt (restores model + optimizer + scheduler state)
    - Rebuilds offline data pipeline from raw CSV (no internet required)
    - Applies updated sarcasm pos_weight from current model_config.yaml
    - Saves new checkpoints continuing from best previous F1
    - Prints per-epoch metrics table to terminal
"""

import argparse
import csv
import os
import random
import sys
from pathlib import Path

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

from utils.logger import get_logger
from utils.config_loader import load_config
from models.multitask_transformer import MultiTaskSentimentModel
from training.focal_loss import CompositeLoss, FocalLoss, BinaryFocalLoss
from training.trainer import Trainer, set_seed
from training.train_utils import find_optimal_threshold

logger = get_logger("resume_training")


# ─────────────────────────────────────────────────────────────────────────────
# Minimal offline Dataset (mirrors download_datasets.py offline pipeline)
# ─────────────────────────────────────────────────────────────────────────────

class SimpleTokenizer:
    """Word-level tokenizer for offline mode (no HuggingFace needed)."""

    def __init__(self, vocab=None, max_vocab=20000):
        self.vocab = vocab or {"<PAD>": 0, "<UNK>": 1}
        self.max_vocab = max_vocab

    def build_vocab(self, texts):
        from collections import Counter
        counts = Counter()
        for t in texts:
            counts.update(t.lower().split())
        for word, _ in counts.most_common(self.max_vocab - 2):
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
        logger.info(f"Vocab built: {len(self.vocab):,} tokens")

    def encode(self, text, max_length=128):
        tokens = text.lower().split()[:max_length]
        ids = [self.vocab.get(w, 1) for w in tokens]
        # Pad or truncate
        if len(ids) < max_length:
            ids += [0] * (max_length - len(ids))
        mask = [1 if i > 0 else 0 for i in ids]
        return ids, mask


class OfflineSentimentDataset(Dataset):
    """
    Reads combined_full.csv and returns tokenized samples.
    Expected CSV columns: text, sentiment_label, sarcasm_label
    """

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
        sarc_raw = str(row.get("sarcasm_label", "0")).strip()

        sent_label = self.SENTIMENT_MAP.get(sent_raw, 1)
        sarc_label = int(sarc_raw) if sarc_raw in ("0", "1") else 0

        ids, mask = self.tokenizer.encode(text, self.max_length)

        return {
            "input_ids":      torch.tensor(ids,  dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "lid_token_ids":  torch.zeros(self.max_length, dtype=torch.long),
            "sentiment_label": torch.tensor(sent_label, dtype=torch.long),
            "sarcasm_label":   torch.tensor(sarc_label,  dtype=torch.long),
        }


def load_csv_data(csv_path: Path):
    rows = []
    with open(csv_path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("text", "").strip():
                rows.append(row)
    logger.info(f"Loaded {len(rows):,} rows from {csv_path.name}")
    return rows


def build_offline_loaders(config: dict, batch_size: int):
    """Build train/val DataLoaders from the raw CSV file offline."""
    csv_path = ROOT / "data" / "raw" / "combined_full.csv"
    if not csv_path.exists():
        # Try sarcasm_news.csv as fallback
        csv_path = ROOT / "data" / "raw" / "sarcasm_news.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No CSV found in {ROOT / 'data' / 'raw'}")

    rows = load_csv_data(csv_path)

    # Build vocabulary
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab([r["text"] for r in rows])

    # Split: 85% train, 15% val
    random.shuffle(rows)
    split = int(0.85 * len(rows))
    train_rows, val_rows = rows[:split], rows[split:]

    train_ds = OfflineSentimentDataset(train_rows, tokenizer, max_length=128)
    val_ds   = OfflineSentimentDataset(val_rows,   tokenizer, max_length=128)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)

    logger.info(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | Batch: {batch_size}")
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# Offline BiLSTM model (mirrors the one used in initial training)
# ─────────────────────────────────────────────────────────────────────────────

class OfflineSentimentModel(torch.nn.Module):
    """
    Matches exactly the architecture saved in checkpoints/best_model.pt.
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

    def forward(self, input_ids, attention_mask=None, lid_token_ids=None, **kwargs):
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

    def get_optimizer_param_groups(self, encoder_lr=2e-5, head_lr=1e-4, weight_decay=0.01):
        head_params = list(self.sentiment_head.parameters()) + list(self.sarcasm_head.parameters())
        enc_params  = [p for p in self.parameters() if not any(p is h for h in head_params)]
        return [
            {"params": enc_params,  "lr": encoder_lr, "weight_decay": weight_decay},
            {"params": head_params, "lr": head_lr,    "weight_decay": weight_decay},
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Resume training from a checkpoint")
    p.add_argument("--checkpoint",   default=str(ROOT / "checkpoints" / "best_model.pt"),
                   help="Path to checkpoint .pt file")
    p.add_argument("--extra-epochs", type=int, default=5,
                   help="Number of additional epochs to train (default: 5)")
    p.add_argument("--batch-size",   type=int, default=16,
                   help="Batch size (default: 16)")
    p.add_argument("--lr",           type=float, default=1e-4,
                   help="Learning rate for resumed training (default: 1e-4)")
    p.add_argument("--config-dir",   default=str(ROOT / "config"),
                   help="Path to config directory")
    p.add_argument("--checkpoint-dir", default=str(ROOT / "checkpoints"),
                   help="Directory to save new checkpoints")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(42)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    logger.info(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    prev_f1 = checkpoint.get("combined_f1", 0.0)
    prev_epoch = checkpoint.get("epoch", 0)
    logger.info(f"Resuming from epoch {prev_epoch} | previous best F1: {prev_f1:.4f}")

    # Build offline data loaders
    model_config   = load_config(str(Path(args.config_dir) / "model_config.yaml"))
    training_config = load_config(str(Path(args.config_dir) / "training_config.yaml"))

    train_loader, val_loader = build_offline_loaders(training_config, args.batch_size)

    # Rebuild the model (exact architecture matching saved checkpoint)
    saved_vocab = checkpoint.get("vocab_size", 30004)
    model = OfflineSentimentModel(
        vocab_size=saved_vocab, embed_dim=128, hidden_dim=256,
        num_lid_langs=10, num_sentiment_classes=3,
    )
    model.load_state_dict(checkpoint["model_state"])
    logger.info("Model weights restored from checkpoint ✓")

    # Build loss with updated sarcasm weights from config
    mt = model_config["multitask"]
    sarc_cfg = mt["sarcasm_loss"]
    sent_cfg = mt["sentiment_loss"]

    focal_sent = FocalLoss(
        alpha=sent_cfg.get("focal_alpha", [0.25, 0.25, 0.5]),
        gamma=sent_cfg.get("focal_gamma", 2.0),
        num_classes=3,
    )
    focal_sarc = BinaryFocalLoss(
        alpha=sarc_cfg.get("focal_alpha", 0.85),
        gamma=sarc_cfg.get("focal_gamma", 2.5),
        pos_weight=sarc_cfg.get("pos_weight", 4.0),
    )
    loss_fn = CompositeLoss(
        focal_sent, focal_sarc,
        sentiment_weight=mt.get("sentiment_weight", 0.6),
        sarcasm_weight=mt.get("sarcasm_weight", 0.4),
    )
    logger.info(
        f"Loss: sarcasm focal_alpha={sarc_cfg.get('focal_alpha','?')} "
        f"pos_weight={sarc_cfg.get('pos_weight','?')}"
    )

    # Patch training_config for the resumed run
    training_config["training"]["num_epochs"] = args.extra_epochs
    training_config["training"]["batch_size"] = args.batch_size
    training_config["training"]["optimizer"]["learning_rate"] = args.lr

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        training_config=training_config,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Override best F1 so trainer saves improved checkpoints from the right baseline
    trainer.best_combined_f1 = prev_f1

    logger.info(
        f"Starting resumed training: {args.extra_epochs} more epochs "
        f"(after epoch {prev_epoch}) | LR={args.lr}"
    )

    history = trainer.train()

    # Print final summary table
    print("\n" + "=" * 60)
    print("RESUMED TRAINING SUMMARY")
    print("=" * 60)
    headers = ["Epoch", "Train Loss", "Val Loss", "Sent F1", "Sarc F1", "Combined F1"]
    print(f"{'Epoch':>6} {'Train Loss':>11} {'Val Loss':>9} {'Sent F1':>8} {'Sarc F1':>8} {'Combined F1':>12}")
    print("-" * 60)
    for i, (tl, vl, sf, rcf, cf) in enumerate(zip(
        history["train_loss"], history["val_loss"],
        history["val_sentiment_f1"], history["val_sarcasm_f1"], history["val_combined_f1"]
    ), start=prev_epoch + 1):
        print(f"{i:>6} {tl:>11.4f} {vl:>9.4f} {sf:>8.4f} {rcf:>8.4f} {cf:>12.4f}")
    print("=" * 60)
    print(f"\nBest Combined F1: {trainer.best_combined_f1:.4f}")
    print(f"Best model saved to: {args.checkpoint_dir}/best_model.pt")


if __name__ == "__main__":
    main()
