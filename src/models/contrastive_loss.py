"""
Contrastive Learning for Sarcasm Detection

Implements supervised contrastive loss to create distinct clusters for
sarcastic vs. non-sarcastic samples in the embedding space.

Motivation:
    Sarcasm is context-dependent and semantically similar to literal language
    at the surface level. E.g.:
        "Great job!"           (literal positive)
        "Great job! 🙄"        (sarcastic negative)

    These have near-identical token sequences but opposite intents.
    Contrastive learning forces the model to separate them in latent space.

Contrastive Loss (SupCon, Khosla et al., 2020):
    L_contrast = Σ_i [ -log( Σ_pos exp(z_i·z_pos/τ) / Σ_all exp(z_i·z_all/τ) ) ]

    Where:
        z_i     = normalized embedding for sample i
        z_pos   = embeddings of same-class samples (sarcastic/non-sarcastic)
        τ       = temperature (controls concentration)

Architecture Integration:
    - Applied to [CLS] embeddings from the encoder
    - Added as an auxiliary loss term: L_total = L_main + λ_contrast · L_contrast
    - Pulls sarcastic samples together, pushes them away from non-sarcastic

This complements the focal loss by shaping the representation space directly.
"""

from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None

from utils.logger import get_logger

logger = get_logger("contrastive_loss")


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., 2020)

    Encourages:
        - High similarity between same-class samples
        - Low similarity between different-class samples

    L = Σ_i [ -log( Σ_{p∈P(i)} exp(z_i·z_p/τ) / Σ_{a∈A(i)} exp(z_i·z_a/τ) ) ]

    Where:
        P(i) = positive set (same class as i, excluding i)
        A(i) = all samples in batch except i
        τ    = temperature parameter

    Args:
        temperature: Softmax temperature (lower = sharper clustering)
        base_temperature: Base temperature for scaling (default 0.07)
    """

    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
    ):
        super().__init__()
        if torch is None:
            return
        self.temperature = temperature
        self.base_temperature = base_temperature
        logger.debug(f"SupervisedContrastiveLoss: temperature={temperature}")

    def forward(
        self,
        embeddings: "torch.Tensor",  # (B, D) — normalized embeddings
        labels:     "torch.Tensor",  # (B,)   — class labels
    ) -> "torch.Tensor":
        """
        Compute supervised contrastive loss.

        Args:
            embeddings: L2-normalized feature embeddings, shape (batch, dim)
            labels:     Class labels (0 or 1 for sarcasm), shape (batch,)

        Returns:
            Scalar contrastive loss
        """
        if torch is None or F is None:
            raise RuntimeError("torch not available")

        device = embeddings.device
        batch_size = embeddings.shape[0]

        # Normalize embeddings (SupCon uses cosine similarity)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute similarity matrix: (B, B)
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Create mask for positive pairs (same label, different sample)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)   # (B, B)

        # Remove diagonal (self-similarity)
        logits_mask = torch.ones_like(mask).fill_diagonal_(0)
        mask = mask * logits_mask

        # Compute log-sum-exp for numerical stability
        # exp_logits = exp(sim / τ)
        exp_logits = torch.exp(similarity_matrix) * logits_mask

        # For each sample i:
        #   log_prob = log( Σ_pos exp(sim) / Σ_all exp(sim) )
        log_prob = similarity_matrix - torch.log(exp_logits.sum(dim=1, keepdim=True))

        # Mean over positive pairs
        # Σ_pos log_prob[i,p] / |P(i)|
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

        # Final loss (negative mean)
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss


class TripletContrastiveLoss(nn.Module):
    """
    Triplet Loss for sarcasm embeddings.

    Simpler alternative to SupCon: for each anchor (sarcastic sample),
    pull a positive (another sarcastic) closer and push a negative
    (non-sarcastic) farther.

    L = max(0, ||anchor - positive||² - ||anchor - negative||² + margin)

    Args:
        margin: Minimum distance gap between pos and neg pairs
        p:      Norm order (2 for Euclidean distance)
    """

    def __init__(self, margin: float = 1.0, p: int = 2):
        super().__init__()
        if torch is None:
            return
        self.margin = margin
        self.p = p
        logger.debug(f"TripletContrastiveLoss: margin={margin}, p={p}")

    def forward(
        self,
        embeddings: "torch.Tensor",  # (B, D)
        labels:     "torch.Tensor",  # (B,) — 0/1 labels
    ) -> "torch.Tensor":
        """
        Compute triplet loss using online hard triplet mining.

        For each anchor:
            - hardest positive = same-class sample farthest from anchor
            - hardest negative = different-class sample closest to anchor

        This is more sample-efficient than random triplet selection.
        """
        if torch is None or F is None:
            raise RuntimeError("torch not available")

        device = embeddings.device
        batch_size = embeddings.shape[0]

        # Pairwise distances: (B, B)
        distances = torch.cdist(embeddings, embeddings, p=self.p)

        # Masks for positive and negative pairs
        labels = labels.unsqueeze(1)  # (B, 1)
        pos_mask = (labels == labels.T).float()  # same class
        neg_mask = (labels != labels.T).float()  # different class

        # Remove self-pairs (diagonal)
        pos_mask.fill_diagonal_(0)

        # Hard positive: max distance among same-class pairs
        pos_distances = distances * pos_mask
        pos_distances[pos_mask == 0] = -float('inf')
        hardest_positive = pos_distances.max(dim=1)[0]  # (B,)

        # Hard negative: min distance among different-class pairs
        neg_distances = distances + (1 - neg_mask) * 1e9  # mask out same-class
        hardest_negative = neg_distances.min(dim=1)[0]  # (B,)

        # Triplet loss
        loss = F.relu(hardest_positive - hardest_negative + self.margin)

        # Only compute loss for samples that have both pos and neg pairs
        valid_mask = (pos_mask.sum(dim=1) > 0) & (neg_mask.sum(dim=1) > 0)
        if valid_mask.sum() > 0:
            loss = loss[valid_mask].mean()
        else:
            loss = torch.tensor(0.0, device=device)

        return loss


class ContrastiveLossWrapper:
    """
    Wrapper to integrate contrastive loss into training loop.

    Usage:
        contrast_loss = ContrastiveLossWrapper(method='supcon', weight=0.1)
        ...
        # In training loop:
        total_loss, components = main_loss(...)
        contrast_loss_val = contrast_loss(cls_embeddings, sarcasm_labels)
        total_loss = total_loss + contrast_loss_val
    """

    def __init__(
        self,
        method: str = "supcon",  # 'supcon' | 'triplet'
        weight: float = 0.1,
        temperature: float = 0.07,
        margin: float = 1.0,
    ):
        if torch is None:
            self.loss_fn = None
            return

        if method == "supcon":
            self.loss_fn = SupervisedContrastiveLoss(temperature=temperature)
        elif method == "triplet":
            self.loss_fn = TripletContrastiveLoss(margin=margin)
        else:
            raise ValueError(f"Unknown contrastive method: {method}")

        self.weight = weight
        logger.info(
            f"ContrastiveLossWrapper: method={method}, weight={weight}"
        )

    def __call__(
        self,
        embeddings: "torch.Tensor",
        labels: "torch.Tensor",
    ) -> "torch.Tensor":
        """Compute weighted contrastive loss."""
        if self.loss_fn is None or torch is None:
            return torch.tensor(0.0)
        return self.weight * self.loss_fn(embeddings, labels)


# ═══════════════════════════════════════════════════════════
# Smoke-test
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    if torch is None:
        print("⚠ torch not available — skipping contrastive loss tests")
    else:
        print("Testing contrastive losses...\n")
        torch.manual_seed(42)

        B, D = 16, 128
        embeddings = torch.randn(B, D)
        labels = torch.randint(0, 2, (B,))  # binary sarcasm labels

        # SupCon
        print("[1] SupervisedContrastiveLoss")
        supcon = SupervisedContrastiveLoss(temperature=0.07)
        loss_sc = supcon(embeddings, labels)
        print(f"  Loss: {loss_sc.item():.4f}")
        assert loss_sc.item() > 0

        # Triplet
        print("\n[2] TripletContrastiveLoss")
        triplet = TripletContrastiveLoss(margin=1.0)
        loss_tr = triplet(embeddings, labels)
        print(f"  Loss: {loss_tr.item():.4f}")
        assert loss_tr.item() >= 0

        # Wrapper
        print("\n[3] ContrastiveLossWrapper")
        wrapper = ContrastiveLossWrapper(method="supcon", weight=0.1)
        loss_w = wrapper(embeddings, labels)
        print(f"  Weighted loss: {loss_w.item():.4f}")
        assert abs(loss_w.item() - 0.1 * loss_sc.item()) < 1e-5

        # Gradient flow
        print("\n[4] Gradient flow test")
        embeddings.requires_grad_(True)
        loss = supcon(embeddings, labels)
        loss.backward()
        assert embeddings.grad is not None
        assert not torch.isnan(embeddings.grad).any()

        print("\n✓ All contrastive loss tests passed!")
