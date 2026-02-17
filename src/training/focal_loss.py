"""
Focal Loss Module

Implements Focal Loss (Lin et al., 2017) for both multi-class and binary
classification, designed to down-weight easy examples and focus training
on hard, misclassified samples.

Standard Cross-Entropy:
    CE(p_t) = -log(p_t)

Focal Loss:
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

Where:
    p_t   = model probability for the correct class
    α_t   = class-balancing weight (inverse frequency)
    γ     = focusing parameter (γ=0 reduces to weighted CE; γ=2 recommended)

Rationale for this project:
    - Sentiment data is often class-imbalanced (more negative than neutral)
    - Sarcasm examples are inherently rare (typically <20% of samples)
    - Hard examples (subtle sarcasm, ambiguous sentiment) should dominate
      gradient updates, not easy positive/negative cases
"""

from pathlib import Path
from typing import List, Optional, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore
    class _NNStub:
        class Module: pass
    nn = _NNStub()  # type: ignore

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import get_logger

logger = get_logger("focal_loss")


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss for sentiment classification (3 classes).

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Args:
        alpha:     Per-class weights as list or tensor of shape (num_classes,).
                   If None, uniform weighting is used.
        gamma:     Focusing parameter. γ=0 → standard CE; γ=2 recommended.
        reduction: 'mean' | 'sum' | 'none'
        num_classes: Number of output classes

    Example:
        >>> criterion = FocalLoss(alpha=[0.25, 0.25, 0.50], gamma=2.0)
        >>> loss = criterion(logits, targets)
    """

    def __init__(
        self,
        alpha: Optional[Union[List[float], torch.Tensor]] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        num_classes: int = 3,
    ):
        super().__init__()

        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean'|'sum'|'none', got {reduction}")

        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes

        # Register alpha as a buffer (moves to correct device automatically)
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            if alpha.shape[0] != num_classes:
                raise ValueError(
                    f"alpha length {alpha.shape[0]} != num_classes {num_classes}"
                )
            self.register_buffer("alpha", alpha)
        else:
            # Uniform weights
            self.register_buffer(
                "alpha", torch.ones(num_classes, dtype=torch.float32)
            )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits:  Raw model output, shape (batch, num_classes)
            targets: Ground truth class indices, shape (batch,)

        Returns:
            Scalar loss (if reduction='mean'/'sum') or per-sample losses
        """
        # Validate inputs
        if logits.dim() != 2:
            raise ValueError(f"logits must be 2D (batch, classes), got {logits.dim()}D")
        if targets.dim() != 1:
            raise ValueError(f"targets must be 1D (batch,), got {targets.dim()}D")
        if logits.shape[0] != targets.shape[0]:
            raise ValueError(
                f"Batch size mismatch: logits={logits.shape[0]}, targets={targets.shape[0]}"
            )

        # Compute log-softmax for numerical stability
        log_probs = F.log_softmax(logits, dim=-1)               # (batch, classes)
        probs     = torch.exp(log_probs)                         # (batch, classes)

        # Gather p_t: probability of the correct class for each sample
        targets_expanded = targets.unsqueeze(1)                  # (batch, 1)
        log_pt = log_probs.gather(1, targets_expanded).squeeze(1)# (batch,)
        pt     = probs.gather(1, targets_expanded).squeeze(1)    # (batch,)

        # Gather α_t: class weight for each sample's true class
        alpha_t = self.alpha.gather(0, targets)                  # (batch,)

        # Focal weight: (1 - p_t)^γ
        focal_weight = (1.0 - pt).pow(self.gamma)

        # Focal loss per sample
        loss = -alpha_t * focal_weight * log_pt                  # (batch,)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss  # 'none'

    def extra_repr(self) -> str:
        return (
            f"gamma={self.gamma}, "
            f"reduction={self.reduction}, "
            f"num_classes={self.num_classes}"
        )


class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss for sarcasm detection.

    Handles class imbalance (sarcastic examples typically rare) by:
      1. Focal weighting: focuses on hard negatives / positives
      2. pos_weight: BCEWithLogitsLoss-style positive class upweighting

    FL_binary(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Where:
        p_t = sigmoid(logit)  for positive class
        p_t = 1 - sigmoid(logit) for negative class

    Args:
        alpha:     Weight for positive class [0,1]. (1-alpha) for negative.
        gamma:     Focusing parameter.
        pos_weight: Additional multiplier on positive class loss
                    (equivalent to oversampling positives).
        reduction: 'mean' | 'sum' | 'none'

    Example:
        >>> criterion = BinaryFocalLoss(alpha=0.75, gamma=2.0, pos_weight=2.0)
        >>> loss = criterion(logits.squeeze(), targets.float())
    """

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        pos_weight: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__()

        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0,1], got {alpha}")
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer(
            "pos_weight", torch.tensor(pos_weight, dtype=torch.float32)
        )

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute binary focal loss.

        Args:
            logits:  Raw model output, shape (batch,) or (batch, 1)
            targets: Binary ground truth, shape (batch,) — float 0.0 or 1.0

        Returns:
            Scalar loss or per-sample tensor
        """
        logits  = logits.squeeze(-1)
        targets = targets.float().squeeze(-1)

        # Numerically stable sigmoid + log
        probs     = torch.sigmoid(logits)
        log_pos   = F.logsigmoid(logits)               # log(p)
        log_neg   = F.logsigmoid(-logits)              # log(1-p)

        # Per-sample p_t and log(p_t)
        pt      = torch.where(targets == 1.0, probs,     1.0 - probs)
        log_pt  = torch.where(targets == 1.0, log_pos,   log_neg)

        # Per-sample α_t
        alpha_t = torch.where(
            targets == 1.0,
            torch.full_like(targets, self.alpha),
            torch.full_like(targets, 1.0 - self.alpha),
        )

        # Focal weight
        focal_weight = (1.0 - pt).pow(self.gamma)

        # Apply pos_weight to positive examples
        pos_weight_t = torch.where(
            targets == 1.0,
            self.pos_weight.expand_as(targets),
            torch.ones_like(targets),
        )

        loss = -alpha_t * focal_weight * log_pt * pos_weight_t

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    def extra_repr(self) -> str:
        return (
            f"alpha={self.alpha}, gamma={self.gamma}, "
            f"reduction={self.reduction}"
        )


class CompositeLoss(nn.Module):
    """
    Composite multi-task loss:

        L_total = λ₁ * L_sentiment + λ₂ * L_sarcasm

    Supports:
        - Static λ weights (set at init)
        - Optional uncertainty-based weighting (Kendall et al., 2018):
          L_total = (1/σ₁²) * L_sentiment + (1/σ₂²) * L_sarcasm + log(σ₁ * σ₂)
          where σ₁, σ₂ are learned log-variance parameters.

    Design rationale:
        - Static weighting (default): simple, stable, interpretable
        - Uncertainty weighting: lets model balance tasks dynamically,
          prevents one task dominating. Useful when tasks conflict.

    Args:
        sentiment_loss_fn: Loss function for sentiment head
        sarcasm_loss_fn:   Loss function for sarcasm head
        sentiment_weight:  λ₁ (used when uncertainty_weighting=False)
        sarcasm_weight:    λ₂
        uncertainty_weighting: Enable learned task weighting

    Example:
        >>> loss_fn = CompositeLoss(
        ...     FocalLoss(alpha=[.25,.25,.5], gamma=2),
        ...     BinaryFocalLoss(alpha=.75, gamma=2),
        ...     sentiment_weight=0.6,
        ...     sarcasm_weight=0.4,
        ... )
        >>> total, components = loss_fn(sent_logits, sarc_logits, sent_labels, sarc_labels)
    """

    def __init__(
        self,
        sentiment_loss_fn: nn.Module,
        sarcasm_loss_fn: nn.Module,
        sentiment_weight: float = 0.6,
        sarcasm_weight: float = 0.4,
        uncertainty_weighting: bool = False,
    ):
        super().__init__()

        self.sentiment_loss_fn = sentiment_loss_fn
        self.sarcasm_loss_fn   = sarcasm_loss_fn
        self.uncertainty_weighting = uncertainty_weighting

        if uncertainty_weighting:
            # Learnable log-variance parameters (σ_i = exp(log_var_i / 2))
            # Initialised near 0 → initial σ ≈ 1 → initial weight ≈ 1
            self.log_var_sentiment = nn.Parameter(torch.zeros(1))
            self.log_var_sarcasm   = nn.Parameter(torch.zeros(1))
            logger.info("CompositeLoss: uncertainty weighting ENABLED")
        else:
            # Static weights — not trainable
            if not (0.0 < sentiment_weight <= 1.0 and 0.0 < sarcasm_weight <= 1.0):
                raise ValueError("Loss weights must be in (0, 1]")
            self.register_buffer(
                "sentiment_weight", torch.tensor(sentiment_weight)
            )
            self.register_buffer(
                "sarcasm_weight", torch.tensor(sarcasm_weight)
            )
            logger.info(
                f"CompositeLoss: static weights "
                f"λ₁={sentiment_weight}, λ₂={sarcasm_weight}"
            )

    def forward(
        self,
        sentiment_logits: torch.Tensor,
        sarcasm_logits:   torch.Tensor,
        sentiment_labels: torch.Tensor,
        sarcasm_labels:   torch.Tensor,
    ) -> tuple:
        """
        Compute composite loss.

        Args:
            sentiment_logits: Shape (batch, 3)
            sarcasm_logits:   Shape (batch, 1) or (batch,)
            sentiment_labels: Shape (batch,) — long
            sarcasm_labels:   Shape (batch,) — long

        Returns:
            Tuple of (total_loss, {component losses dict})
        """
        l_sentiment = self.sentiment_loss_fn(sentiment_logits, sentiment_labels)
        l_sarcasm   = self.sarcasm_loss_fn(
            sarcasm_logits.squeeze(-1), sarcasm_labels.float()
        )

        if self.uncertainty_weighting:
            # Uncertainty weighting (Kendall et al., 2018)
            # L = (1/σ₁²)·L₁ + (1/σ₂²)·L₂ + log(σ₁·σ₂)
            precision_sent = torch.exp(-self.log_var_sentiment)
            precision_sarc = torch.exp(-self.log_var_sarcasm)

            total = (
                precision_sent * l_sentiment + self.log_var_sentiment * 0.5 +
                precision_sarc * l_sarcasm   + self.log_var_sarcasm   * 0.5
            )
            total = total.squeeze()

            eff_w_sent = precision_sent.item()
            eff_w_sarc = precision_sarc.item()
        else:
            total = self.sentiment_weight * l_sentiment + \
                    self.sarcasm_weight   * l_sarcasm
            eff_w_sent = self.sentiment_weight.item()
            eff_w_sarc = self.sarcasm_weight.item()

        components = {
            "loss_total":            total.item(),
            "loss_sentiment":        l_sentiment.item(),
            "loss_sarcasm":          l_sarcasm.item(),
            "effective_weight_sent": eff_w_sent,
            "effective_weight_sarc": eff_w_sarc,
        }

        return total, components

    @classmethod
    def from_config(cls, model_config: dict) -> "CompositeLoss":
        """
        Build CompositeLoss from model_config.yaml dict.

        Args:
            model_config: Loaded model_config.yaml content

        Returns:
            Configured CompositeLoss instance
        """
        mt = model_config["multitask"]

        # Sentiment loss
        sent_cfg = mt["sentiment_loss"]
        if sent_cfg["type"] == "focal":
            sentiment_fn = FocalLoss(
                alpha=sent_cfg.get("focal_alpha"),
                gamma=sent_cfg.get("focal_gamma", 2.0),
                num_classes=model_config["model"]["sentiment"]["num_classes"],
            )
        else:
            sentiment_fn = nn.CrossEntropyLoss()

        # Sarcasm loss
        sarc_cfg = mt["sarcasm_loss"]
        if sarc_cfg["type"] == "focal":
            sarcasm_fn = BinaryFocalLoss(
                alpha=sarc_cfg.get("focal_alpha", 0.75),
                gamma=sarc_cfg.get("focal_gamma", 2.0),
                pos_weight=sarc_cfg.get("pos_weight", 2.0),
            )
        else:
            sarcasm_fn = nn.BCEWithLogitsLoss()

        # Loss balancing
        balancing = mt.get("loss_balancing", {})
        uncertainty = balancing.get("method") == "uncertainty_weighting"

        return cls(
            sentiment_loss_fn=sentiment_fn,
            sarcasm_loss_fn=sarcasm_fn,
            sentiment_weight=mt.get("sentiment_weight", 0.6),
            sarcasm_weight=mt.get("sarcasm_weight", 0.4),
            uncertainty_weighting=uncertainty,
        )


# ─────────────────────────────────────────────
# Smoke-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    B = 8  # batch size

    # FocalLoss
    fl = FocalLoss(alpha=[0.25, 0.25, 0.50], gamma=2.0, num_classes=3)
    logits  = torch.randn(B, 3)
    targets = torch.randint(0, 3, (B,))
    loss = fl(logits, targets)
    print(f"FocalLoss:       {loss.item():.4f}")
    assert loss.item() > 0

    # BinaryFocalLoss
    bfl     = BinaryFocalLoss(alpha=0.75, gamma=2.0, pos_weight=2.0)
    b_logits = torch.randn(B)
    b_labels = torch.randint(0, 2, (B,)).float()
    b_loss   = bfl(b_logits, b_labels)
    print(f"BinaryFocalLoss: {b_loss.item():.4f}")
    assert b_loss.item() > 0

    # CompositeLoss (static)
    comp = CompositeLoss(fl, bfl, sentiment_weight=0.6, sarcasm_weight=0.4)
    total, comps = comp(logits, b_logits, targets, b_labels.long())
    print(f"CompositeLoss (static):      total={comps['loss_total']:.4f}")
    print(f"  L_sentiment={comps['loss_sentiment']:.4f}, "
          f"L_sarcasm={comps['loss_sarcasm']:.4f}")

    # CompositeLoss (uncertainty weighting)
    comp_unc = CompositeLoss(fl, bfl, uncertainty_weighting=True)
    total_u, comps_u = comp_unc(logits, b_logits, targets, b_labels.long())
    print(f"CompositeLoss (uncertainty): total={comps_u['loss_total']:.4f}")

    # Gradient flow check (use total_u which has grad_fn)
    total_u.backward()
    print("\nAll focal loss tests passed!")
