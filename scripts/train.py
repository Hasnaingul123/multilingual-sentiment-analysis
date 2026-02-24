"""
Training Entry Point

Wires CSV data → SentimentDatasetBuilder → Trainer for end-to-end training.

Usage:
    python scripts/train.py
    python scripts/train.py --data data/raw/combined_full.csv --epochs 3
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ─────────────────────────────────────────────
# Minimal tokenizer fallback (no HuggingFace needed)
# ─────────────────────────────────────────────

class SimpleTokenizer:
    """
    Word-level tokenizer fallback (used when transformers not available).
    Matches the interface expected by SentimentDatasetBuilder.
    """
    def __init__(self, vocab_size=30000, max_length=128):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.word2id = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
        self.next_id = 4

    def _get_id(self, word):
        if word not in self.word2id:
            if len(self.word2id) < self.vocab_size:
                self.word2id[word] = self.next_id
                self.next_id += 1
            else:
                return self.word2id["[UNK]"]
        return self.word2id[word]

    def __call__(self, text, max_length=128, padding="max_length",
                 truncation=True, return_tensors=None,
                 return_attention_mask=True, return_token_type_ids=False, **kwargs):
        tokens = str(text).lower().split()[:max_length - 2]
        ids = [2] + [self._get_id(t) for t in tokens] + [3]
        mask = [1] * len(ids)

        # Pad
        pad_len = max_length - len(ids)
        ids += [0] * pad_len
        mask += [0] * pad_len

        result = type("Encoding", (), {
            "input_ids": ids[:max_length],
            "attention_mask": mask[:max_length],
        })()

        # Provide word_ids() method
        n_real = len(tokens) + 2
        def word_ids():
            wids = [None]  # CLS
            for i in range(len(tokens)):
                wids.append(i)
            wids.append(None)  # SEP
            wids += [None] * pad_len
            return wids[:max_length]

        result.word_ids = word_ids
        return result


# ─────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────

def make_torch_dataset(samples):
    """Convert processed samples to a PyTorch Dataset."""
    try:
        import torch
        from torch.utils.data import Dataset

        class SentimentTorchDataset(Dataset):
            def __init__(self, samples):
                self.samples = samples

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                s = self.samples[idx]
                return {
                    "input_ids":      torch.tensor(s["input_ids"],       dtype=torch.long),
                    "attention_mask": torch.tensor(s["attention_mask"],  dtype=torch.long),
                    "lid_token_ids":  torch.tensor(s["lid_token_ids"],   dtype=torch.long),
                    "sentiment_label":torch.tensor(s["sentiment_label"], dtype=torch.long),
                    "sarcasm_label":  torch.tensor(s["sarcasm_label"],   dtype=torch.long),
                }

        return SentimentTorchDataset(samples)

    except ImportError:
        raise RuntimeError("PyTorch is required for training. pip install torch")



# ─────────────────────────────────────────────
# Offline BiLSTM fallback model (no HuggingFace needed)
# ─────────────────────────────────────────────

def _build_offline_model(vocab_size=30000, embed_dim=128, hidden_dim=256,
                          max_length=64, num_lid_langs=10):
    """
    Pure-PyTorch BiLSTM model for multi-task sentiment + sarcasm.
    No internet required — trains from scratch.
    Matches the MultiTaskSentimentModel forward() interface:
        forward(batch) → {"sentiment_logits": ..., "sarcasm_logits": ...}
    """
    import torch
    import torch.nn as nn

    class OfflineSentimentModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed       = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.lid_embed   = nn.Embedding(num_lid_langs + 1, 16, padding_idx=0)
            self.lstm        = nn.LSTM(embed_dim + 16, hidden_dim, num_layers=2,
                                       batch_first=True, dropout=0.3,
                                       bidirectional=True)
            self.dropout     = nn.Dropout(0.4)
            feat_dim         = hidden_dim * 2  # bidirectional

            # Task heads
            self.sentiment_head = nn.Sequential(
                nn.Linear(feat_dim, 128), nn.Tanh(), nn.Dropout(0.2),
                nn.Linear(128, 3),
            )
            self.sarcasm_head = nn.Sequential(
                nn.Linear(feat_dim, 64), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, 1),   # binary → BinaryFocalLoss expects (B,) or (B,1)
            )

        def forward(self, input_ids, attention_mask=None, lid_token_ids=None):
            # Clamp lid ids to valid range; default to zeros if not provided
            if lid_token_ids is None:
                lid_token_ids = torch.zeros_like(input_ids)
            lid_token_ids = lid_token_ids.clamp(0, 10)
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)

            x    = torch.cat([self.embed(input_ids), self.lid_embed(lid_token_ids)], dim=-1)
            out, _ = self.lstm(x)              # (B, L, 2H)
            # Mean pool over non-padding tokens
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
            pooled = self.dropout(pooled)

            return {
                "sentiment_logits": self.sentiment_head(pooled),
                "sarcasm_logit":    self.sarcasm_head(pooled),
            }

    return OfflineSentimentModel()


# ─────────────────────────────────────────────
# Main Training Script
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train multilingual sentiment model")
    parser.add_argument("--data",       default="data/raw/combined_full.csv")
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--max-length", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap dataset size for quick runs")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("Multilingual Sentiment Model Training")
    print("="*60)

    # ── 1. Load CSV ──────────────────────────────────────
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Dataset not found at {data_path}")
        print("Run: python scripts/download_datasets.py")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"\nLoaded: {len(df):,} samples from {data_path}")

    # Keep only needed cols; fill missing sarcasm with 0
    if "sarcasm" not in df.columns:
        df["sarcasm"] = "0"
    df = df[["text", "sentiment", "sarcasm"]].dropna()

    if args.max_samples:
        df = df.sample(min(args.max_samples, len(df)), random_state=42)
        print(f"Capped to {len(df):,} samples")

    # ── 2. Tokenizer + direct offline processing ─────────
    SENTIMENT_MAP = {"negative": 0, "neutral": 1, "positive": 2}
    MAX_LEN = args.max_length
    tokenizer = SimpleTokenizer(vocab_size=30000, max_length=MAX_LEN)

    print(f"\nProcessing {len(df):,} samples offline (no HuggingFace needed)...")

    def row_to_sample(row):
        text    = str(row["text"])
        sent    = SENTIMENT_MAP.get(str(row.get("sentiment", "neutral")).lower(), 1)
        sarc    = int(str(row.get("sarcasm", "0")).strip()) if str(row.get("sarcasm","0")).strip().isdigit() else 0
        enc     = tokenizer(text, max_length=MAX_LEN)
        lid_ids = [0] * MAX_LEN  # heuristic LID = all "unknown"
        return {
            "input_ids":       enc.input_ids[:MAX_LEN],
            "attention_mask":  enc.attention_mask[:MAX_LEN],
            "lid_token_ids":   lid_ids,
            "sentiment_label": sent,
            "sarcasm_label":   sarc,
        }

    samples = [row_to_sample(row) for _, row in df.iterrows()]
    print(f"✓ {len(samples):,} samples processed")

    # Train / val / test split (75/15/10)
    import random
    random.seed(42)
    random.shuffle(samples)
    n      = len(samples)
    n_test = max(1, int(n * 0.10))
    n_val  = max(1, int(n * 0.15))
    splits = {
        "train":      samples[n_test + n_val:],
        "validation": samples[n_test:n_test + n_val],
        "test":       samples[:n_test],
    }

    print(f"\nSplit sizes:")
    for split, data in splits.items():
        print(f"  {split:12s}: {len(data):,} samples")

    # ── 4. PyTorch DataLoaders ────────────────────────────
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        print("\nError: PyTorch not installed.")
        print("Run: pip install torch")
        sys.exit(1)

    train_ds   = make_torch_dataset(splits["train"])
    val_ds     = make_torch_dataset(splits["validation"])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"\nDataLoaders ready:")
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")

    # ── 5. Model ─────────────────────────────────────────
    print("\nBuilding model...")
    model = None
    try:
        from models.multitask_transformer import MultiTaskSentimentModel
        from utils.config_loader import load_config as _lc
        model_config = _lc("config/model_config.yaml")
        model = MultiTaskSentimentModel.from_config(model_config)
        print("✓ MultiTaskSentimentModel loaded")
    except Exception as e:
        print(f"⚠ Transformer model unavailable: {e}")
        print("  → Falling back to offline BiLSTM model (trains from scratch)")
        model = _build_offline_model(
            vocab_size=getattr(tokenizer, "next_id", 30000),
            embed_dim=128,
            hidden_dim=256,
            max_length=args.max_length,
            num_lid_langs=10,
        )
        print("✓ Offline BiLSTM model ready")

    # ── 6. Loss & Trainer ────────────────────────────────
    from training.focal_loss import CompositeLoss, FocalLoss, BinaryFocalLoss
    from training.trainer import build_trainer
    try:
        from utils.config_loader import load_config
        model_config = load_config("config/model_config.yaml")
        mt_cfg = model_config["multitask"]
        sentiment_weight = mt_cfg["sentiment_weight"]
        sarcasm_weight   = mt_cfg["sarcasm_weight"]
    except Exception:
        sentiment_weight, sarcasm_weight = 0.6, 0.4

    loss_fn = CompositeLoss(
        sentiment_loss_fn=FocalLoss(gamma=2.0),
        sarcasm_loss_fn=BinaryFocalLoss(alpha=0.75, gamma=2.0, pos_weight=2.0),
        sentiment_weight=sentiment_weight,
        sarcasm_weight=sarcasm_weight,
    )

    lr = args.lr if model.__class__.__name__ == "OfflineSentimentModel" else 2e-5
    training_config = {
        "training": {
            "num_epochs":                  args.epochs,
            "batch_size":                  args.batch_size,
            "learning_rate":               lr,
            "encoder_lr":                  lr,
            "head_lr":                     lr * 10,
            "weight_decay":                0.01,
            "gradient_accumulation_steps": 1,
            "max_grad_norm":               1.0,
            "log_every_n_steps":           50,
            "seed":                        42,
            "scheduler":                   {"num_warmup_steps": 100},
            "early_stopping":              {"enabled": True, "patience": 3, "min_delta": 1e-3},
        }
    }

    trainer = build_trainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        training_config=training_config,
        checkpoint_dir=args.checkpoint_dir,
    )

    # ── 7. Train ─────────────────────────────────────────
    print(f"\nStarting training: {args.epochs} epochs | batch={args.batch_size} | lr={args.lr}")
    print("="*60 + "\n")

    history = trainer.train()

    # ── 8. Summary ───────────────────────────────────────
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)

    try:
        if isinstance(history, list) and history and isinstance(history[0], dict):
            best_f1 = max((e.get("combined_f1", 0) for e in history), default=0)
        else:
            best_f1 = 0.0
    except Exception:
        best_f1 = 0.0

    print(f"Best Combined F1 : {best_f1:.4f}")
    print(f"Checkpoints saved: {args.checkpoint_dir}/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
