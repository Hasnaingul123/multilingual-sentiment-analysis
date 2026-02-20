"""
Evaluation Metrics Module

Comprehensive evaluation suite for multi-task sentiment analysis.

Metrics Categories:
    1. Classification Metrics: Precision, Recall, F1, Accuracy
    2. Per-Language Breakdown: Performance by dominant language
    3. Calibration Metrics: ECE (Expected Calibration Error), reliability diagrams
    4. Confusion Matrices: Per-task error analysis
    5. Aggregate Metrics: Combined F1, weighted scores

Rationale:
    - Standard metrics alone insufficient for multilingual, multi-task models
    - Need to diagnose: Which languages are hard? Is model overconfident?
    - Calibration critical for production deployment (trust in probabilities)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import get_logger

logger = get_logger("metrics")


# ═══════════════════════════════════════════════════════════
# Classification Metrics
# ═══════════════════════════════════════════════════════════

class ClassificationMetrics:
    """
    Compute precision, recall, F1, and accuracy for classification tasks.
    
    Supports both binary (sarcasm) and multi-class (sentiment) problems.
    """
    
    @staticmethod
    def compute(
        predictions: np.ndarray,
        labels: np.ndarray,
        num_classes: int,
        average: str = "macro",
        zero_division: float = 0.0,
    ) -> Dict[str, float]:
        """
        Compute classification metrics.
        
        Args:
            predictions: Predicted class indices (N,)
            labels:      True class indices (N,)
            num_classes: Total number of classes
            average:     'macro' | 'micro' | 'weighted' | 'binary'
            zero_division: Value to use when denominator is zero
            
        Returns:
            Dict with precision, recall, f1, accuracy
        """
        from sklearn.metrics import (
            precision_score, recall_score, f1_score, accuracy_score
        )
        
        metrics = {
            "precision": precision_score(
                labels, predictions, average=average,
                zero_division=zero_division, labels=range(num_classes)
            ),
            "recall": recall_score(
                labels, predictions, average=average,
                zero_division=zero_division, labels=range(num_classes)
            ),
            "f1": f1_score(
                labels, predictions, average=average,
                zero_division=zero_division, labels=range(num_classes)
            ),
            "accuracy": accuracy_score(labels, predictions),
        }
        
        return {k: float(v) for k, v in metrics.items()}
    
    @staticmethod
    def per_class_metrics(
        predictions: np.ndarray,
        labels: np.ndarray,
        num_classes: int,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics for each class individually.
        
        Returns:
            Dict[class_name, {precision, recall, f1, support}]
        """
        from sklearn.metrics import precision_recall_fscore_support
        
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, labels=range(num_classes), zero_division=0
        )
        
        if class_names is None:
            class_names = [f"class_{i}" for i in range(num_classes)]
        
        per_class = {}
        for i, name in enumerate(class_names):
            per_class[name] = {
                "precision": float(precision[i]),
                "recall":    float(recall[i]),
                "f1":        float(f1[i]),
                "support":   int(support[i]),
            }
        
        return per_class


# ═══════════════════════════════════════════════════════════
# Calibration Metrics
# ═══════════════════════════════════════════════════════════

class CalibrationMetrics:
    """
    Evaluate model calibration: Are predicted probabilities reliable?
    
    Well-calibrated model: When it predicts 70% confidence, it's correct 70% of the time.
    
    Metrics:
        - ECE (Expected Calibration Error): Average gap between confidence and accuracy
        - MCE (Maximum Calibration Error): Worst-case bin
        - Reliability diagram data: (confidence, accuracy) per bin
    """
    
    @staticmethod
    def expected_calibration_error(
        probabilities: np.ndarray,  # (N, num_classes) or (N,) for binary
        labels: np.ndarray,         # (N,)
        num_bins: int = 15,
    ) -> Dict[str, float]:
        """
        Compute Expected Calibration Error (ECE).
        
        ECE = Σ_b (n_b / N) * |acc_b - conf_b|
        
        Where:
            n_b    = number of samples in bin b
            acc_b  = accuracy in bin b
            conf_b = average confidence in bin b
        
        Args:
            probabilities: Predicted probabilities
            labels:        True labels
            num_bins:      Number of bins for histogram
            
        Returns:
            Dict with ece, mce, and bin statistics
        """
        # Handle binary case (N,) → (N, 2)
        if probabilities.ndim == 1:
            probabilities = np.stack([1 - probabilities, probabilities], axis=-1)
        
        # Get predicted class and confidence
        confidences = np.max(probabilities, axis=-1)  # (N,)
        predictions = np.argmax(probabilities, axis=-1)  # (N,)
        accuracies = (predictions == labels).astype(float)
        
        # Bin samples by confidence
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        mce = 0.0
        bin_data = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Samples in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                # Calibration error for this bin
                bin_error = abs(avg_confidence_in_bin - accuracy_in_bin)
                ece += prop_in_bin * bin_error
                mce = max(mce, bin_error)
                
                bin_data.append({
                    "bin_lower":    float(bin_lower),
                    "bin_upper":    float(bin_upper),
                    "confidence":   float(avg_confidence_in_bin),
                    "accuracy":     float(accuracy_in_bin),
                    "count":        int(in_bin.sum()),
                    "bin_error":    float(bin_error),
                })
        
        return {
            "ece": float(ece),
            "mce": float(mce),
            "num_bins": num_bins,
            "bins": bin_data,
        }
    
    @staticmethod
    def reliability_diagram_data(
        probabilities: np.ndarray,
        labels: np.ndarray,
        num_bins: int = 15,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate data for plotting reliability diagram.
        
        Returns:
            (bin_centers, accuracies, counts)
        """
        ece_result = CalibrationMetrics.expected_calibration_error(
            probabilities, labels, num_bins
        )
        
        bins = ece_result["bins"]
        bin_centers = np.array([
            (b["bin_lower"] + b["bin_upper"]) / 2 for b in bins
        ])
        accuracies = np.array([b["accuracy"] for b in bins])
        counts = np.array([b["count"] for b in bins])
        
        return bin_centers, accuracies, counts


# ═══════════════════════════════════════════════════════════
# Per-Language Evaluation
# ═══════════════════════════════════════════════════════════

class PerLanguageMetrics:
    """
    Breakdown of performance by dominant language.
    
    Critical for multilingual models: Are we overfitting to English?
    Do code-switched examples perform worse?
    """
    
    @staticmethod
    def compute(
        predictions: np.ndarray,
        labels: np.ndarray,
        languages: List[str],
        num_classes: int,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics separately for each language.
        
        Args:
            predictions: Predicted class indices
            labels:      True class indices
            languages:   List of language codes per sample
            num_classes: Total number of classes
            
        Returns:
            Dict[language_code, {f1, accuracy, count}]
        """
        from sklearn.metrics import f1_score, accuracy_score
        
        # Group samples by language
        lang_to_indices = defaultdict(list)
        for idx, lang in enumerate(languages):
            lang_to_indices[lang].append(idx)
        
        per_lang_metrics = {}
        for lang, indices in lang_to_indices.items():
            if len(indices) == 0:
                continue
            
            lang_preds = predictions[indices]
            lang_labels = labels[indices]
            
            per_lang_metrics[lang] = {
                "f1": float(f1_score(
                    lang_labels, lang_preds,
                    average="macro" if num_classes > 2 else "binary",
                    zero_division=0,
                )),
                "accuracy": float(accuracy_score(lang_labels, lang_preds)),
                "count": len(indices),
            }
        
        return per_lang_metrics


# ═══════════════════════════════════════════════════════════
# Confusion Matrix
# ═══════════════════════════════════════════════════════════

class ConfusionMatrixMetrics:
    """
    Compute and analyze confusion matrices.
    
    Identifies systematic errors: e.g., does model confuse neutral with positive?
    """
    
    @staticmethod
    def compute(
        predictions: np.ndarray,
        labels: np.ndarray,
        num_classes: int,
        normalize: Optional[str] = None,  # None | 'true' | 'pred' | 'all'
    ) -> np.ndarray:
        """
        Compute confusion matrix.
        
        Args:
            predictions: Predicted class indices
            labels:      True class indices
            num_classes: Total classes
            normalize:   Normalization mode
            
        Returns:
            Confusion matrix (num_classes, num_classes)
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(
            labels, predictions, labels=range(num_classes)
        )
        
        if normalize == "true":
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        elif normalize == "pred":
            cm = cm.astype(float) / cm.sum(axis=0, keepdims=True)
        elif normalize == "all":
            cm = cm.astype(float) / cm.sum()
        
        return cm
    
    @staticmethod
    def most_confused_pairs(
        cm: np.ndarray,
        class_names: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> List[Tuple[str, str, float]]:
        """
        Identify most commonly confused class pairs.
        
        Returns:
            List of (true_class, pred_class, count) sorted by count
        """
        if class_names is None:
            class_names = [f"class_{i}" for i in range(cm.shape[0])]
        
        # Extract off-diagonal elements (errors)
        confused_pairs = []
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if i != j and cm[i, j] > 0:
                    confused_pairs.append((
                        class_names[i],
                        class_names[j],
                        float(cm[i, j])
                    ))
        
        # Sort by count descending
        confused_pairs.sort(key=lambda x: x[2], reverse=True)
        return confused_pairs[:top_k]


# ═══════════════════════════════════════════════════════════
# Multi-Task Evaluator (convenience wrapper)
# ═══════════════════════════════════════════════════════════

class MultiTaskEvaluator:
    """
    Complete evaluation suite for sentiment + sarcasm multi-task model.
    
    Usage:
        evaluator = MultiTaskEvaluator()
        results = evaluator.evaluate(
            sentiment_preds, sentiment_labels, sentiment_probs,
            sarcasm_preds, sarcasm_labels, sarcasm_probs,
            languages
        )
    """
    
    def __init__(
        self,
        sentiment_class_names: List[str] = ["negative", "neutral", "positive"],
        sarcasm_class_names: List[str] = ["literal", "sarcastic"],
    ):
        self.sentiment_class_names = sentiment_class_names
        self.sarcasm_class_names = sarcasm_class_names
        logger.info("MultiTaskEvaluator initialized")
    
    def evaluate(
        self,
        sentiment_preds:  np.ndarray,
        sentiment_labels: np.ndarray,
        sentiment_probs:  np.ndarray,
        sarcasm_preds:    np.ndarray,
        sarcasm_labels:   np.ndarray,
        sarcasm_probs:    np.ndarray,
        languages:        Optional[List[str]] = None,
    ) -> Dict:
        """
        Run full evaluation suite.
        
        Returns:
            Comprehensive results dict with all metrics
        """
        results = {}
        
        # ── Sentiment Metrics ────────────────────────────────────────────
        results["sentiment"] = {
            "overall": ClassificationMetrics.compute(
                sentiment_preds, sentiment_labels, num_classes=3, average="macro"
            ),
            "per_class": ClassificationMetrics.per_class_metrics(
                sentiment_preds, sentiment_labels, num_classes=3,
                class_names=self.sentiment_class_names
            ),
            "confusion_matrix": ConfusionMatrixMetrics.compute(
                sentiment_preds, sentiment_labels, num_classes=3
            ).tolist(),
            "calibration": CalibrationMetrics.expected_calibration_error(
                sentiment_probs, sentiment_labels, num_bins=15
            ),
        }
        
        # ── Sarcasm Metrics ──────────────────────────────────────────────
        results["sarcasm"] = {
            "overall": ClassificationMetrics.compute(
                sarcasm_preds, sarcasm_labels, num_classes=2, average="binary"
            ),
            "per_class": ClassificationMetrics.per_class_metrics(
                sarcasm_preds, sarcasm_labels, num_classes=2,
                class_names=self.sarcasm_class_names
            ),
            "confusion_matrix": ConfusionMatrixMetrics.compute(
                sarcasm_preds, sarcasm_labels, num_classes=2
            ).tolist(),
            "calibration": CalibrationMetrics.expected_calibration_error(
                sarcasm_probs, sarcasm_labels, num_bins=15
            ),
        }
        
        # ── Per-Language Breakdown ───────────────────────────────────────
        if languages is not None:
            results["per_language"] = {
                "sentiment": PerLanguageMetrics.compute(
                    sentiment_preds, sentiment_labels, languages, num_classes=3
                ),
                "sarcasm": PerLanguageMetrics.compute(
                    sarcasm_preds, sarcasm_labels, languages, num_classes=2
                ),
            }
        
        # ── Combined Metrics ─────────────────────────────────────────────
        results["combined"] = {
            "combined_f1": (
                0.6 * results["sentiment"]["overall"]["f1"] +
                0.4 * results["sarcasm"]["overall"]["f1"]
            ),
            "avg_ece": (
                results["sentiment"]["calibration"]["ece"] +
                results["sarcasm"]["calibration"]["ece"]
            ) / 2,
        }
        
        logger.info(
            f"Evaluation complete | "
            f"Sentiment F1: {results['sentiment']['overall']['f1']:.4f} | "
            f"Sarcasm F1: {results['sarcasm']['overall']['f1']:.4f} | "
            f"Combined F1: {results['combined']['combined_f1']:.4f}"
        )
        
        return results


# ═══════════════════════════════════════════════════════════
# Smoke-test
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing evaluation metrics...\n")
    
    N = 100
    np.random.seed(42)
    
    # ── Classification Metrics ───────────────────────────────────────────
    print("[1] ClassificationMetrics")
    preds = np.random.randint(0, 3, N)
    labels = np.random.randint(0, 3, N)
    metrics = ClassificationMetrics.compute(preds, labels, num_classes=3)
    print(f"  F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
    
    per_class = ClassificationMetrics.per_class_metrics(preds, labels, num_classes=3)
    print(f"  Per-class: {list(per_class.keys())}")
    
    # ── Calibration Metrics ──────────────────────────────────────────────
    print("\n[2] CalibrationMetrics")
    probs = np.random.dirichlet([1, 1, 1], N)
    labels = np.random.randint(0, 3, N)
    calib = CalibrationMetrics.expected_calibration_error(probs, labels)
    print(f"  ECE: {calib['ece']:.4f}, MCE: {calib['mce']:.4f}")
    
    # ── Per-Language Metrics ─────────────────────────────────────────────
    print("\n[3] PerLanguageMetrics")
    langs = np.random.choice(["en", "hi", "es"], N).tolist()
    per_lang = PerLanguageMetrics.compute(preds, labels, langs, num_classes=3)
    print(f"  Languages: {list(per_lang.keys())}")
    for lang, m in per_lang.items():
        print(f"    {lang}: F1={m['f1']:.3f}, n={m['count']}")
    
    # ── Confusion Matrix ─────────────────────────────────────────────────
    print("\n[4] ConfusionMatrix")
    cm = ConfusionMatrixMetrics.compute(preds, labels, num_classes=3)
    print(f"  Shape: {cm.shape}")
    confused = ConfusionMatrixMetrics.most_confused_pairs(
        cm, class_names=["neg", "neu", "pos"], top_k=3
    )
    print(f"  Most confused pairs: {confused[:2]}")
    
    # ── MultiTaskEvaluator ───────────────────────────────────────────────
    print("\n[5] MultiTaskEvaluator")
    evaluator = MultiTaskEvaluator()
    
    sent_preds = np.random.randint(0, 3, N)
    sent_labels = np.random.randint(0, 3, N)
    sent_probs = np.random.dirichlet([1, 1, 1], N)
    
    sarc_preds = np.random.randint(0, 2, N)
    sarc_labels = np.random.randint(0, 2, N)
    sarc_probs = np.random.rand(N)
    
    results = evaluator.evaluate(
        sent_preds, sent_labels, sent_probs,
        sarc_preds, sarc_labels, sarc_probs,
        languages=langs
    )
    
    print(f"  Combined F1: {results['combined']['combined_f1']:.4f}")
    print(f"  Avg ECE: {results['combined']['avg_ece']:.4f}")
    
    print("\n✓ All evaluation metrics tests passed!")
