"""
Task Head Modules

Implements classification heads for sentiment and sarcasm tasks.
Both heads attach to the shared XLM-RoBERTa encoder's [CLS] representation.

Architecture per head:
    [CLS] embedding (768-dim)
        ↓ LayerNorm
        ↓ Dropout
        ↓ Linear(768 → hidden_dim)
        ↓ Activation (tanh for sentiment, relu for sarcasm)
        ↓ Dropout
        ↓ Linear(hidden_dim → num_classes)
        ↓ raw logits (loss functions handle softmax/sigmoid)

Design rationale:
    - Two Dropout layers: regularise both the encoder output AND intermediate
    - LayerNorm before projection: stabilises training with pretrained embeddings
    - No final activation: let loss function decide (focal, CE, BCE)
    - Separate hidden_dim per task: sarcasm head is smaller (binary) by default

LID Feature Fusion:
    When LID embeddings are enabled, language IDs per subword token are
    embedded and averaged across non-padding positions, then concatenated
    to [CLS] before the projection layers.
"""

from pathlib import Path
from typing import Optional

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None  # type: ignore
    class _NNStub:
        class Module: pass
    nn = _NNStub()  # type: ignore

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import get_logger

logger = get_logger("task_heads")


class LIDFusionLayer(nn.Module):
    """
    Language ID Fusion Layer.

    Embeds per-subword language IDs and produces a sentence-level
    language representation by mean-pooling over non-padding positions.
    The result is projected and concatenated to [CLS] before classification.

    Args:
        num_languages:   Total number of language ID classes (incl. und, emoji)
        lid_embed_dim:   Embedding dimension for language IDs
        output_dim:      Target dimension after projection (matches encoder hidden size)
        dropout:         Dropout applied after projection
    """

    def __init__(
        self,
        num_languages: int = 12,
        lid_embed_dim: int = 32,
        output_dim:    int = 64,
        dropout:       float = 0.1,
    ):
        super().__init__()
        self.embedding  = nn.Embedding(num_languages, lid_embed_dim, padding_idx=10)
        self.projection = nn.Linear(lid_embed_dim, output_dim)
        self.dropout    = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        lid_token_ids:  torch.Tensor,   # (batch, seq_len)
        attention_mask: torch.Tensor,   # (batch, seq_len)
    ) -> torch.Tensor:
        """
        Args:
            lid_token_ids:  Language ID per subword token
            attention_mask: 1 for real tokens, 0 for padding

        Returns:
            LID representation, shape (batch, output_dim)
        """
        # Embed language IDs: (batch, seq_len, lid_embed_dim)
        embedded = self.embedding(lid_token_ids)

        # Mean-pool over non-padding tokens
        mask_expanded = attention_mask.unsqueeze(-1).float()        # (B, L, 1)
        summed   = (embedded * mask_expanded).sum(dim=1)            # (B, lid_embed_dim)
        count    = mask_expanded.sum(dim=1).clamp(min=1e-9)         # (B, 1)
        mean_lid = summed / count                                    # (B, lid_embed_dim)

        # Project and normalise
        out = self.projection(mean_lid)
        out = self.layer_norm(out)
        out = self.dropout(out)

        return out


class SentimentHead(nn.Module):
    """
    3-class sentiment classification head (negative / neutral / positive).

    Input:  [CLS] embedding, optionally concatenated with LID features
    Output: logits of shape (batch, 3)

    Args:
        input_dim:   Dimension of input (encoder hidden size + LID dim if fused)
        hidden_dim:  Intermediate projection size
        num_classes: Number of sentiment classes (default 3)
        dropout:     Dropout probability
        activation:  'tanh' | 'relu' | 'gelu'
    """

    def __init__(
        self,
        input_dim:   int = 768,
        hidden_dim:  int = 256,
        num_classes: int = 3,
        dropout:     float = 0.2,
        activation:  str = "tanh",
    ):
        super().__init__()

        act_map = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU}
        if activation not in act_map:
            raise ValueError(f"activation must be one of {list(act_map)}")

        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout1   = nn.Dropout(dropout)
        self.dense      = nn.Linear(input_dim, hidden_dim)
        self.activation = act_map[activation]()
        self.dropout2   = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

        self._init_weights()
        logger.debug(
            f"SentimentHead: {input_dim}→{hidden_dim}→{num_classes} | "
            f"act={activation} | dropout={dropout}"
        )

    def _init_weights(self) -> None:
        """Xavier uniform initialisation for linear layers."""
        nn.init.xavier_uniform_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, cls_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cls_embedding: (batch, input_dim)

        Returns:
            logits: (batch, num_classes)
        """
        x = self.layer_norm(cls_embedding)
        x = self.dropout1(x)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout2(x)
        return self.classifier(x)


class SarcasmHead(nn.Module):
    """
    Binary sarcasm detection head.

    Returns a single logit per sample (no sigmoid — BCE/focal handles it).

    Args:
        input_dim:  Dimension of input
        hidden_dim: Intermediate projection size
        dropout:    Dropout probability
        activation: 'relu' | 'tanh' | 'gelu'
    """

    def __init__(
        self,
        input_dim:  int = 768,
        hidden_dim: int = 128,
        dropout:    float = 0.2,
        activation: str = "relu",
    ):
        super().__init__()

        act_map = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU}
        if activation not in act_map:
            raise ValueError(f"activation must be one of {list(act_map)}")

        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout1   = nn.Dropout(dropout)
        self.dense      = nn.Linear(input_dim, hidden_dim)
        self.activation = act_map[activation]()
        self.dropout2   = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)

        self._init_weights()
        logger.debug(
            f"SarcasmHead: {input_dim}→{hidden_dim}→1 | "
            f"act={activation} | dropout={dropout}"
        )

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, cls_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cls_embedding: (batch, input_dim)

        Returns:
            logit: (batch, 1)
        """
        x = self.layer_norm(cls_embedding)
        x = self.dropout1(x)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout2(x)
        return self.classifier(x)


# ─────────────────────────────────────────────
# Smoke-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    B, D = 8, 768

    # LID fusion
    lid_layer  = LIDFusionLayer(num_languages=12, lid_embed_dim=32, output_dim=64)
    lid_ids    = torch.randint(0, 12, (B, 128))
    attn_mask  = torch.ones(B, 128, dtype=torch.long)
    attn_mask[:, 100:] = 0   # simulate padding

    lid_out    = lid_layer(lid_ids, attn_mask)
    print(f"LID fusion output: {lid_out.shape}")   # (8, 64)
    assert lid_out.shape == (B, 64)

    # Sentiment head (with LID fusion)
    sent_head  = SentimentHead(input_dim=D + 64, hidden_dim=256, num_classes=3)
    cls_embed  = torch.randn(B, D)
    fused      = torch.cat([cls_embed, lid_out], dim=-1)
    sent_logits = sent_head(fused)
    print(f"Sentiment logits:  {sent_logits.shape}")   # (8, 3)
    assert sent_logits.shape == (B, 3)

    # Sarcasm head (no LID fusion)
    sarc_head  = SarcasmHead(input_dim=D, hidden_dim=128)
    sarc_logit = sarc_head(cls_embed)
    print(f"Sarcasm logit:     {sarc_logit.shape}")   # (8, 1)
    assert sarc_logit.shape == (B, 1)

    # Gradient flow
    loss = sent_logits.sum() + sarc_logit.sum()
    loss.backward()
    print("\nTask head smoke-test passed!")
