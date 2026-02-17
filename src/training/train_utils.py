"""
Training Utilities

Helper functions and classes for the training pipeline:
- EarlyStopping: monitors validation metric and stops training
- Metrics computation: F1, accuracy, precision, recall
- Combined metric calculation for multi-task learning
"""

from typing import Dict, List
import numpy as np

try:
    from sklearn.metrics import (
        f1_score,
        accuracy_score,
        precision_score,
        recall_score,
        classification_report,
    )
except ImportError:
    # Graceful degradation if sklearn not installed
    f1_score = None
    accuracy_score = None
    precision_score = None
    recall_score = None
    classification_report = None


class EarlyStopping:
    """
    Early stopping to halt training when validation metric stops improving.

    Args:
        patience:   Number of epochs to wait for improvement
        min_delta:  Minimum change to qualify as improvement
        mode:       'min' (loss) or 'max' (F1, accuracy)
    """

    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 1e-3,
        mode: str = "max",
    ):
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")

        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.counter    = 0
        self.best_score = None
        self.should_stop = False

    def step(self, metric: float) -> bool:
        """
        Update early stopping state with new metric value.

        Args:
            metric: Current epoch's validation metric

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = metric
            return False

        # Check if improved
        if self.mode == "max":
            improved = metric > (self.best_score + self.min_delta)
        else:
            improved = metric < (self.best_score - self.min_delta)

        if improved:
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True

        return False

    def reset(self) -> None:
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.should_stop = False


def compute_epoch_metrics(
    sentiment_preds: List[int],
    sentiment_labels: List[int],
    sarcasm_preds: List[int],
    sarcasm_labels: List[int],
    sentiment_weight: float = 0.6,
    sarcasm_weight: float = 0.4,
) -> Dict[str, float]:
    """
    Compute multi-task evaluation metrics.

    Args:
        sentiment_preds:  Predicted sentiment classes (0/1/2)
        sentiment_labels: True sentiment classes
        sarcasm_preds:    Predicted sarcasm labels (0/1)
        sarcasm_labels:   True sarcasm labels
        sentiment_weight: Weight for sentiment in combined metric
        sarcasm_weight:   Weight for sarcasm in combined metric

    Returns:
        Dict with:
            sentiment_f1, sentiment_acc, sentiment_precision, sentiment_recall,
            sarcasm_f1, sarcasm_acc, sarcasm_precision, sarcasm_recall,
            combined_f1
    """
    if f1_score is None:
        raise ImportError(
            "sklearn required for metrics computation. "
            "Run: pip install scikit-learn"
        )

    # Sentiment metrics (macro-averaged for 3 classes)
    sent_f1   = f1_score(sentiment_labels, sentiment_preds, average="macro")
    sent_acc  = accuracy_score(sentiment_labels, sentiment_preds)
    sent_prec = precision_score(sentiment_labels, sentiment_preds, average="macro", zero_division=0)
    sent_rec  = recall_score(sentiment_labels, sentiment_preds, average="macro", zero_division=0)

    # Sarcasm metrics (binary)
    sarc_f1   = f1_score(sarcasm_labels, sarcasm_preds, average="binary", zero_division=0)
    sarc_acc  = accuracy_score(sarcasm_labels, sarcasm_preds)
    sarc_prec = precision_score(sarcasm_labels, sarcasm_preds, average="binary", zero_division=0)
    sarc_rec  = recall_score(sarcasm_labels, sarcasm_preds, average="binary", zero_division=0)

    # Combined metric (mirrors loss weights)
    combined_f1 = sentiment_weight * sent_f1 + sarcasm_weight * sarc_f1

    return {
        "sentiment_f1":        sent_f1,
        "sentiment_acc":       sent_acc,
        "sentiment_precision": sent_prec,
        "sentiment_recall":    sent_rec,
        "sarcasm_f1":          sarc_f1,
        "sarcasm_acc":         sarc_acc,
        "sarcasm_precision":   sarc_prec,
        "sarcasm_recall":      sarc_rec,
        "combined_f1":         combined_f1,
    }


def print_classification_report(
    sentiment_preds: List[int],
    sentiment_labels: List[int],
    sarcasm_preds: List[int],
    sarcasm_labels: List[int],
) -> None:
    """Print detailed classification reports for both tasks."""
    if classification_report is None:
        print("sklearn not available for detailed reports")
        return

    print("\n" + "="*60)
    print("SENTIMENT CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(
        sentiment_labels,
        sentiment_preds,
        target_names=["Negative", "Neutral", "Positive"],
        digits=4,
    ))

    print("\n" + "="*60)
    print("SARCASM DETECTION REPORT")
    print("="*60)
    print(classification_report(
        sarcasm_labels,
        sarcasm_preds,
        target_names=["Not Sarcastic", "Sarcastic"],
        digits=4,
    ))
