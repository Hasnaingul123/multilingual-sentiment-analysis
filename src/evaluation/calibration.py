"""
Model Calibration Module

Implements post-hoc calibration techniques to improve probability reliability.

Temperature Scaling (Guo et al., 2017):
    The simplest and most effective calibration method.
    
    Original: p(y|x) = softmax(z/T)
    Where: z = logits, T = temperature (learned on validation set)
    
    T > 1 → softer probabilities (less confident)
    T < 1 → sharper probabilities (more confident)
    T = 1 → no change

Rationale:
    - Neural networks are often miscalibrated (overconfident)
    - Temperature scaling fixes this with a single scalar parameter
    - No retraining needed — just optimize T on validation set

Binary calibration (for sarcasm):
    Similar to temperature scaling but for binary logits
"""

from pathlib import Path
from typing import Optional

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.nn import functional as F
except ImportError:
    torch = None
    nn = None
    F = None

from utils.logger import get_logger

logger = get_logger("calibration")


# ═══════════════════════════════════════════════════════════
# Temperature Scaling (Multi-class)
# ═══════════════════════════════════════════════════════════

class TemperatureScaling:
    """
    Temperature scaling for multi-class classification (sentiment).
    
    Learns a single temperature parameter T to calibrate probabilities:
        p_calibrated = softmax(logits / T)
    
    T is optimized to minimize NLL on a validation set.
    
    Usage:
        calibrator = TemperatureScaling()
        calibrator.fit(val_logits, val_labels)
        calibrated_probs = calibrator.predict(test_logits)
    """
    
    def __init__(self):
        self.temperature = 1.0  # Default: no scaling
        self.is_fitted = False
        logger.debug("TemperatureScaling initialized")
    
    def fit(
        self,
        logits: "torch.Tensor",  # (N, num_classes)
        labels: "torch.Tensor",  # (N,)
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> float:
        """
        Learn optimal temperature on validation set.
        
        Args:
            logits: Raw model outputs (before softmax)
            labels: Ground truth class indices
            lr:     Learning rate for temperature optimization
            max_iter: Maximum optimization iterations
            
        Returns:
            Final NLL loss (calibration quality)
        """
        if torch is None or nn is None:
            raise RuntimeError("torch not available")
        
        # Temperature parameter (initialized to 1.0)
        temperature = torch.nn.Parameter(torch.ones(1) * 1.5)
        optimizer = optim.LBFGS([temperature], lr=lr, max_iter=max_iter)
        
        logits = logits.detach()
        labels = labels.detach()
        
        def eval_loss():
            optimizer.zero_grad()
            loss = F.cross_entropy(logits / temperature, labels)
            loss.backward()
            return loss
        
        # Optimize temperature
        optimizer.step(eval_loss)
        
        self.temperature = temperature.item()
        self.is_fitted = True
        
        final_loss = eval_loss().item()
        logger.info(
            f"Temperature scaling fitted: T={self.temperature:.4f}, "
            f"NLL={final_loss:.4f}"
        )
        
        return final_loss
    
    def predict(self, logits: "torch.Tensor") -> "torch.Tensor":
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: Raw model outputs (N, num_classes)
            
        Returns:
            Calibrated probabilities (N, num_classes)
        """
        if torch is None or F is None:
            raise RuntimeError("torch not available")
        
        if not self.is_fitted:
            logger.warning("Temperature not fitted — using T=1.0")
        
        return F.softmax(logits / self.temperature, dim=-1)
    
    def get_temperature(self) -> float:
        """Return learned temperature."""
        return self.temperature


# ═══════════════════════════════════════════════════════════
# Binary Temperature Scaling (for sarcasm)
# ═══════════════════════════════════════════════════════════

class BinaryTemperatureScaling:
    """
    Temperature scaling for binary classification (sarcasm).
    
    Learns temperature T for sigmoid calibration:
        p_calibrated = sigmoid(logit / T)
    """
    
    def __init__(self):
        self.temperature = 1.0
        self.is_fitted = False
        logger.debug("BinaryTemperatureScaling initialized")
    
    def fit(
        self,
        logits: "torch.Tensor",  # (N,) or (N, 1)
        labels: "torch.Tensor",  # (N,)
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> float:
        """
        Learn optimal temperature for binary classification.
        
        Args:
            logits: Raw model outputs (before sigmoid)
            labels: Ground truth binary labels (0 or 1)
            lr:     Learning rate
            max_iter: Maximum iterations
            
        Returns:
            Final BCE loss
        """
        if torch is None or nn is None or F is None:
            raise RuntimeError("torch not available")
        
        logits = logits.squeeze().detach()
        labels = labels.float().detach()
        
        temperature = torch.nn.Parameter(torch.ones(1) * 1.5)
        optimizer = optim.LBFGS([temperature], lr=lr, max_iter=max_iter)
        
        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = logits / temperature
            loss = F.binary_cross_entropy_with_logits(scaled_logits, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        
        self.temperature = temperature.item()
        self.is_fitted = True
        
        final_loss = eval_loss().item()
        logger.info(
            f"Binary temperature scaling fitted: T={self.temperature:.4f}, "
            f"BCE={final_loss:.4f}"
        )
        
        return final_loss
    
    def predict(self, logits: "torch.Tensor") -> "torch.Tensor":
        """
        Apply temperature scaling to binary logits.
        
        Args:
            logits: Raw model outputs (N,) or (N, 1)
            
        Returns:
            Calibrated probabilities (N,)
        """
        if torch is None:
            raise RuntimeError("torch not available")
        
        if not self.is_fitted:
            logger.warning("Temperature not fitted — using T=1.0")
        
        logits = logits.squeeze()
        return torch.sigmoid(logits / self.temperature)
    
    def get_temperature(self) -> float:
        return self.temperature


# ═══════════════════════════════════════════════════════════
# Multi-Task Calibrator (convenience wrapper)
# ═══════════════════════════════════════════════════════════

class MultiTaskCalibrator:
    """
    Calibrate both sentiment and sarcasm heads jointly.
    
    Usage:
        calibrator = MultiTaskCalibrator()
        calibrator.fit(
            val_sentiment_logits, val_sentiment_labels,
            val_sarcasm_logits, val_sarcasm_labels
        )
        
        # In inference:
        sent_probs = calibrator.calibrate_sentiment(sent_logits)
        sarc_prob = calibrator.calibrate_sarcasm(sarc_logits)
    """
    
    def __init__(self):
        self.sentiment_calibrator = TemperatureScaling()
        self.sarcasm_calibrator = BinaryTemperatureScaling()
        logger.info("MultiTaskCalibrator initialized")
    
    def fit(
        self,
        sentiment_logits: "torch.Tensor",
        sentiment_labels: "torch.Tensor",
        sarcasm_logits: "torch.Tensor",
        sarcasm_labels: "torch.Tensor",
    ) -> Dict[str, float]:
        """
        Fit both calibrators on validation data.
        
        Returns:
            Dict with temperatures and losses
        """
        sent_loss = self.sentiment_calibrator.fit(sentiment_logits, sentiment_labels)
        sarc_loss = self.sarcasm_calibrator.fit(sarcasm_logits, sarcasm_labels)
        
        results = {
            "sentiment_temperature": self.sentiment_calibrator.get_temperature(),
            "sarcasm_temperature":   self.sarcasm_calibrator.get_temperature(),
            "sentiment_nll":         sent_loss,
            "sarcasm_bce":           sarc_loss,
        }
        
        logger.info(
            f"Calibration complete | "
            f"T_sentiment={results['sentiment_temperature']:.3f} | "
            f"T_sarcasm={results['sarcasm_temperature']:.3f}"
        )
        
        return results
    
    def calibrate_sentiment(
        self, logits: "torch.Tensor"
    ) -> "torch.Tensor":
        """Apply sentiment calibration."""
        return self.sentiment_calibrator.predict(logits)
    
    def calibrate_sarcasm(
        self, logits: "torch.Tensor"
    ) -> "torch.Tensor":
        """Apply sarcasm calibration."""
        return self.sarcasm_calibrator.predict(logits)


# ═══════════════════════════════════════════════════════════
# Smoke-test
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    if torch is None:
        print("⚠ torch not available — skipping calibration tests")
    else:
        print("Testing calibration modules...\n")
        torch.manual_seed(42)
        
        N = 200
        
        # ── Multi-class (sentiment) ──────────────────────────────────────
        print("[1] TemperatureScaling (multi-class)")
        val_logits = torch.randn(N, 3)
        val_labels = torch.randint(0, 3, (N,))
        
        calibrator = TemperatureScaling()
        nll = calibrator.fit(val_logits, val_labels)
        print(f"  Learned T: {calibrator.get_temperature():.4f}")
        print(f"  Final NLL: {nll:.4f}")
        
        # Test prediction
        test_logits = torch.randn(50, 3)
        calibrated_probs = calibrator.predict(test_logits)
        print(f"  Calibrated probs shape: {calibrated_probs.shape}")
        assert calibrated_probs.shape == (50, 3)
        assert torch.allclose(calibrated_probs.sum(dim=-1), torch.ones(50), atol=1e-5)
        
        # ── Binary (sarcasm) ─────────────────────────────────────────────
        print("\n[2] BinaryTemperatureScaling")
        val_logits_bin = torch.randn(N)
        val_labels_bin = torch.randint(0, 2, (N,))
        
        bin_calibrator = BinaryTemperatureScaling()
        bce = bin_calibrator.fit(val_logits_bin, val_labels_bin)
        print(f"  Learned T: {bin_calibrator.get_temperature():.4f}")
        print(f"  Final BCE: {bce:.4f}")
        
        test_logits_bin = torch.randn(50)
        calibrated_probs_bin = bin_calibrator.predict(test_logits_bin)
        print(f"  Calibrated probs shape: {calibrated_probs_bin.shape}")
        assert calibrated_probs_bin.shape == (50,)
        assert (calibrated_probs_bin >= 0).all() and (calibrated_probs_bin <= 1).all()
        
        # ── Multi-task ───────────────────────────────────────────────────
        print("\n[3] MultiTaskCalibrator")
        mt_calibrator = MultiTaskCalibrator()
        results = mt_calibrator.fit(
            val_logits, val_labels,
            val_logits_bin, val_labels_bin
        )
        print(f"  Sentiment T: {results['sentiment_temperature']:.3f}")
        print(f"  Sarcasm T: {results['sarcasm_temperature']:.3f}")
        
        # Test inference
        sent_probs = mt_calibrator.calibrate_sentiment(test_logits)
        sarc_probs = mt_calibrator.calibrate_sarcasm(test_logits_bin)
        assert sent_probs.shape == (50, 3)
        assert sarc_probs.shape == (50,)
        
        print("\n✓ All calibration tests passed!")
