"""
PyTorch Dataset and DataLoader Module

Wraps processed samples into torch.utils.data.Dataset
and provides factory functions for DataLoaders with:
    - Stratified shuffling
    - Dynamic padding (optional)
    - Class-weight computation for imbalanced labels
    - LID-feature tensor construction
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import get_logger

logger = get_logger("data_loader")

# Lazy torch import — allows importing module metadata without torch installed
try:
    import torch
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Provide stub so type annotations don't break
    class Dataset:  # type: ignore
        pass


# ─────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────

class MultiTaskSentimentDataset(Dataset):
    """
    PyTorch Dataset for multi-task sentiment + sarcasm classification.

    Each item returns a dict of tensors ready for the model forward pass:
        input_ids        : (max_length,)  LongTensor
        attention_mask   : (max_length,)  LongTensor
        lid_token_ids    : (max_length,)  LongTensor   — language ID per subword
        sentiment_label  : ()             LongTensor
        sarcasm_label    : ()             LongTensor

    Attributes:
        samples (List[Dict]):   Pre-processed sample dicts from DatasetBuilder
        has_labels (bool):      False for inference-only datasets
    """

    REQUIRED_KEYS = {"input_ids", "attention_mask", "lid_token_ids"}

    def __init__(
        self,
        samples: List[Dict],
        has_labels: bool = True,
    ):
        """
        Args:
            samples:    List of processed sample dicts
            has_labels: Whether label fields are expected in samples
        """
        if not samples:
            raise ValueError("Cannot create dataset from empty sample list")

        self._validate_samples(samples, has_labels)
        self.samples = samples
        self.has_labels = has_labels

        logger.info(
            f"MultiTaskSentimentDataset: {len(samples)} samples | "
            f"labels={'yes' if has_labels else 'no'}"
        )

    def _validate_samples(self, samples: List[Dict], has_labels: bool) -> None:
        """Validate the first sample for required keys."""
        first = samples[0]
        missing = self.REQUIRED_KEYS - set(first.keys())
        if missing:
            raise KeyError(
                f"Sample missing required keys: {missing}. "
                f"Run samples through SentimentDatasetBuilder first."
            )
        if has_labels:
            label_keys = {"sentiment_label", "sarcasm_label"}
            missing_labels = label_keys - set(first.keys())
            if missing_labels:
                raise KeyError(
                    f"Sample missing label keys: {missing_labels}"
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        item = {
            "input_ids":     torch.tensor(sample["input_ids"],     dtype=torch.long),
            "attention_mask":torch.tensor(sample["attention_mask"], dtype=torch.long),
            "lid_token_ids": torch.tensor(sample["lid_token_ids"], dtype=torch.long),
        }

        if self.has_labels:
            item["sentiment_label"] = torch.tensor(
                sample["sentiment_label"], dtype=torch.long
            )
            item["sarcasm_label"] = torch.tensor(
                sample["sarcasm_label"], dtype=torch.long
            )

        return item

    # ── Utility Methods ─────────────────────────────────────────────────────

    def get_sentiment_labels(self) -> List[int]:
        """Return all sentiment labels as a flat list."""
        if not self.has_labels:
            raise RuntimeError("Dataset has no labels (inference mode)")
        return [s["sentiment_label"] for s in self.samples]

    def get_sarcasm_labels(self) -> List[int]:
        """Return all sarcasm labels as a flat list."""
        if not self.has_labels:
            raise RuntimeError("Dataset has no labels (inference mode)")
        return [s["sarcasm_label"] for s in self.samples]

    def class_counts(self) -> Dict[str, Dict[int, int]]:
        """Return class distribution for sentiment and sarcasm."""
        from collections import Counter
        return {
            "sentiment": dict(Counter(self.get_sentiment_labels())),
            "sarcasm":   dict(Counter(self.get_sarcasm_labels())),
        }

    def language_distribution(self) -> Dict[str, int]:
        """Return distribution of dominant languages."""
        from collections import Counter
        langs = [s.get("dominant_language", "und") for s in self.samples]
        return dict(Counter(langs))

    def code_switched_fraction(self) -> float:
        """Fraction of samples that are code-switched."""
        cs = sum(1 for s in self.samples if s.get("is_code_switched", False))
        return cs / len(self.samples)


# ─────────────────────────────────────────────
# Class Weight Computation
# ─────────────────────────────────────────────

def compute_class_weights(
    labels: List[int],
    num_classes: int,
    method: str = "inverse_freq",
) -> torch.Tensor:
    """
    Compute per-class weights to counteract label imbalance.

    Args:
        labels:      Flat list of integer labels
        num_classes: Total number of classes
        method:      'inverse_freq' (1/count) or 'balanced' (sklearn-style)

    Returns:
        Float tensor of shape (num_classes,) — one weight per class
    """
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.where(counts == 0, 1e-6, counts)   # avoid division by zero

    if method == "inverse_freq":
        weights = 1.0 / counts
    elif method == "balanced":
        # sklearn balanced: n_samples / (n_classes * count)
        weights = len(labels) / (num_classes * counts)
    else:
        raise ValueError(f"Unknown weight method: {method}")

    # Normalise so weights sum to num_classes
    weights = weights / weights.sum() * num_classes

    logger.debug(
        f"Class weights ({method}): "
        + ", ".join(f"cls{i}={w:.3f}" for i, w in enumerate(weights))
    )

    return torch.tensor(weights, dtype=torch.float)


# ─────────────────────────────────────────────
# Weighted Sampler (for training)
# ─────────────────────────────────────────────

def build_weighted_sampler(
    dataset: MultiTaskSentimentDataset,
    primary_task: str = "sentiment",
) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler to oversample minority classes during training.

    Weights each sample by the inverse frequency of its primary task label.

    Args:
        dataset:      Training dataset instance
        primary_task: 'sentiment' or 'sarcasm' — which task drives sampling

    Returns:
        WeightedRandomSampler instance
    """
    if primary_task == "sentiment":
        labels = dataset.get_sentiment_labels()
        num_classes = 3
    else:
        labels = dataset.get_sarcasm_labels()
        num_classes = 2

    class_weights = compute_class_weights(labels, num_classes)

    # Weight per sample = class weight of its label
    sample_weights = torch.tensor(
        [float(class_weights[lbl]) for lbl in labels],
        dtype=torch.double
    )

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ─────────────────────────────────────────────
# DataLoader Factory
# ─────────────────────────────────────────────

def build_dataloaders(
    train_samples:  List[Dict],
    val_samples:    List[Dict],
    test_samples:   List[Dict],
    batch_size:     int = 16,
    eval_batch_size: int = 32,
    num_workers:    int = 0,
    pin_memory:     bool = False,
    use_weighted_sampler: bool = True,
    sampler_task:   str = "sentiment",
    seed:           int = 42,
) -> Dict[str, DataLoader]:
    """
    Build train / validation / test DataLoaders.

    Args:
        train_samples:         Processed train samples
        val_samples:           Processed validation samples
        test_samples:          Processed test samples
        batch_size:            Training batch size
        eval_batch_size:       Eval/test batch size
        num_workers:           DataLoader worker processes
        pin_memory:            Pin tensors in memory (GPU speedup)
        use_weighted_sampler:  Oversample minority classes in training
        sampler_task:          Which task drives sampling weights
        seed:                  RNG seed for reproducibility

    Returns:
        Dict with 'train', 'validation', 'test' DataLoaders
    """

    def _seed_worker(worker_id: int) -> None:
        """Ensure reproducibility in DataLoader workers."""
        np.random.seed(seed + worker_id)
        torch.manual_seed(seed + worker_id)

    generator = torch.Generator().manual_seed(seed)

    # ── Training DataLoader ──────────────────────────────────────────
    train_dataset = MultiTaskSentimentDataset(train_samples, has_labels=True)

    if use_weighted_sampler:
        sampler = build_weighted_sampler(train_dataset, sampler_task)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=_seed_worker,
            generator=generator,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=_seed_worker,
            generator=generator,
        )

    # ── Validation DataLoader ─────────────────────────────────────────
    val_dataset = MultiTaskSentimentDataset(val_samples, has_labels=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # ── Test DataLoader ───────────────────────────────────────────────
    test_dataset = MultiTaskSentimentDataset(test_samples, has_labels=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    logger.info(
        f"DataLoaders ready | "
        f"train={len(train_dataset)} (bs={batch_size}, "
        f"weighted={'yes' if use_weighted_sampler else 'no'}) | "
        f"val={len(val_dataset)} | test={len(test_dataset)}"
    )

    return {
        "train":      train_loader,
        "validation": val_loader,
        "test":       test_loader,
    }


def build_inference_loader(
    texts: List[str],
    processor,
    batch_size: int = 32,
    num_workers: int = 0,
) -> DataLoader:
    """
    Build a DataLoader for inference (no labels).

    Args:
        texts:       Raw text strings to predict
        processor:   SampleProcessor instance
        batch_size:  Inference batch size
        num_workers: DataLoader workers

    Returns:
        DataLoader over inference samples (no label tensors)
    """
    samples = []
    for text in texts:
        try:
            sample = processor.process(text)
            samples.append(sample)
        except Exception as e:
            logger.warning(f"Skipping text due to processing error: {e}")

    if not samples:
        raise ValueError("No valid samples after processing")

    dataset = MultiTaskSentimentDataset(samples, has_labels=False)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


# ─────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing DataLoader components...\n")

    # Minimal fake samples to test tensor shapes
    fake_samples = [
        {
            "input_ids":       [1] + [101, 202, 303] + [0] * 124,
            "attention_mask":  [1] * 4 + [0] * 124,
            "lid_token_ids":   [10] + [0, 1, 0] + [10] * 124,
            "sentiment_label": i % 3,
            "sarcasm_label":   i % 2,
            "is_code_switched": i % 2 == 0,
            "dominant_language": "en" if i % 2 == 0 else "hi",
        }
        for i in range(60)
    ]

    # Test Dataset
    dataset = MultiTaskSentimentDataset(fake_samples)
    sample_item = dataset[0]
    print("Dataset item keys:", list(sample_item.keys()))
    print("input_ids shape:", sample_item["input_ids"].shape)
    print("Class distribution:", dataset.class_counts())
    print("Language distribution:", dataset.language_distribution())
    print(f"Code-switched fraction: {dataset.code_switched_fraction():.2%}")

    # Split into train/val/test
    train_s  = fake_samples[:40]
    val_s    = fake_samples[40:50]
    test_s   = fake_samples[50:]

    # Build DataLoaders
    loaders = build_dataloaders(
        train_s, val_s, test_s,
        batch_size=8, eval_batch_size=16,
        use_weighted_sampler=True,
    )

    # Inspect one batch
    batch = next(iter(loaders["train"]))
    print("\nTrain batch keys:", list(batch.keys()))
    for k, v in batch.items():
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

    # Test class weights
    labels = [s["sentiment_label"] for s in fake_samples]
    weights = compute_class_weights(labels, num_classes=3)
    print("\nSentiment class weights:", weights)

    print("\nDataLoader smoke-test passed ✓")
