"""
Multi-Task Sentiment Transformer

Core model architecture combining:
    1. XLM-RoBERTa shared encoder (multilingual, code-switching aware)
    2. LID Feature Fusion Layer (optional)
    3. Sentiment Classification Head (3-class)
    4. Sarcasm Detection Head (binary)

Forward pass:
    input_ids, attention_mask, lid_token_ids
        ↓
    XLM-RoBERTa encoder
        ↓
    [CLS] token embedding
        ↓
    [Optional] Concatenate LID mean-pooled embedding
        ↓
    ├── SentimentHead → logits (B, 3)
    └── SarcasmHead   → logit  (B, 1)

Design decisions:
    - XLM-RoBERTa-base chosen over mBERT:
        • Trained on 2.5TB data vs mBERT's 68GB
        • No token-type IDs (simpler for code-switching)
        • Better on low-resource and mixed-language tasks
    - [CLS] pooling over mean-pooling:
        • XLM-RoBERTa is fine-tuned with [CLS] as sentence representation
        • More stable for classification than mean-pooling
    - LID features concatenated (not attention-injected):
        • Simpler, avoids attention modification in pre-trained layers
        • Can be ablated by setting lid_integration.enabled=False

Layer freezing strategy:
    - Embeddings always frozen first (most general features)
    - Lower N encoder layers can be frozen via config
    - Task heads always trainable
    - Allows parameter-efficient fine-tuning on CPU/small GPU
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

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
from models.task_heads import SentimentHead, SarcasmHead, LIDFusionLayer

logger = get_logger("multitask_transformer")


class MultiTaskSentimentModel(nn.Module):
    """
    Multi-task Transformer for joint sentiment + sarcasm prediction.

    Usage:
        model = MultiTaskSentimentModel.from_config(model_config)
        outputs = model(input_ids, attention_mask, lid_token_ids)
        # outputs['sentiment_logits'] → (B, 3)
        # outputs['sarcasm_logit']    → (B, 1)
        # outputs['pooled_output']    → (B, hidden_size)

    Args:
        base_model_name:      HuggingFace model ID
        sentiment_num_classes: 3 for neg/neu/pos
        sentiment_hidden_dim:  Sentiment head hidden size
        sentiment_dropout:     Sentiment head dropout
        sentiment_activation:  Activation function name
        sarcasm_hidden_dim:    Sarcasm head hidden size
        sarcasm_dropout:       Sarcasm head dropout
        sarcasm_activation:    Activation function name
        lid_enabled:           Whether to use LID feature fusion
        lid_num_languages:     Vocabulary size for LID embeddings
        lid_embed_dim:         LID embedding dimension
        lid_output_dim:        LID projection dimension (appended to CLS)
        freeze_embeddings:     Freeze the word embedding layer
        freeze_encoder_layers: Number of bottom encoder layers to freeze (0=none)
        hidden_dropout_prob:   Transformer hidden dropout override
        attention_dropout_prob: Transformer attention dropout override
    """

    def __init__(
        self,
        base_model_name:       str   = "xlm-roberta-base",
        sentiment_num_classes: int   = 3,
        sentiment_hidden_dim:  int   = 256,
        sentiment_dropout:     float = 0.2,
        sentiment_activation:  str   = "tanh",
        sarcasm_hidden_dim:    int   = 128,
        sarcasm_dropout:       float = 0.2,
        sarcasm_activation:    str   = "relu",
        lid_enabled:           bool  = True,
        lid_num_languages:     int   = 12,
        lid_embed_dim:         int   = 32,
        lid_output_dim:        int   = 64,
        freeze_embeddings:     bool  = False,
        freeze_encoder_layers: int   = 0,
        hidden_dropout_prob:   Optional[float] = None,
        attention_dropout_prob: Optional[float] = None,
    ):
        super().__init__()

        self.base_model_name  = base_model_name
        self.lid_enabled      = lid_enabled

        # ── Load pre-trained encoder ────────────────────────────────────────
        logger.info(f"Loading encoder: {base_model_name}")
        self.encoder = self._load_encoder(
            base_model_name,
            hidden_dropout_prob,
            attention_dropout_prob,
        )

        hidden_size = self.encoder.config.hidden_size   # 768 for base

        # ── LID Fusion ──────────────────────────────────────────────────────
        if lid_enabled:
            self.lid_fusion = LIDFusionLayer(
                num_languages=lid_num_languages,
                lid_embed_dim=lid_embed_dim,
                output_dim=lid_output_dim,
            )
            classifier_input_dim = hidden_size + lid_output_dim
            logger.info(
                f"LID fusion: {lid_num_languages} langs → "
                f"{lid_embed_dim}d embed → {lid_output_dim}d proj | "
                f"classifier_input={classifier_input_dim}"
            )
        else:
            self.lid_fusion = None
            classifier_input_dim = hidden_size
            logger.info("LID fusion disabled")

        # ── Task heads ──────────────────────────────────────────────────────
        self.sentiment_head = SentimentHead(
            input_dim=classifier_input_dim,
            hidden_dim=sentiment_hidden_dim,
            num_classes=sentiment_num_classes,
            dropout=sentiment_dropout,
            activation=sentiment_activation,
        )
        self.sarcasm_head = SarcasmHead(
            input_dim=classifier_input_dim,
            hidden_dim=sarcasm_hidden_dim,
            dropout=sarcasm_dropout,
            activation=sarcasm_activation,
        )

        # ── Freezing strategy ───────────────────────────────────────────────
        self._apply_freezing(freeze_embeddings, freeze_encoder_layers)

        # Log parameter counts
        total, trainable = self._count_parameters()
        logger.info(
            f"Model ready: {total:,} total params | "
            f"{trainable:,} trainable ({100*trainable/total:.1f}%)"
        )

    # ── Encoder loading ──────────────────────────────────────────────────────

    @staticmethod
    def _load_encoder(
        model_name: str,
        hidden_dropout: Optional[float],
        attention_dropout: Optional[float],
    ):
        """Load XLM-RoBERTa (or other HuggingFace) encoder."""
        try:
            from transformers import AutoModel, AutoConfig

            config = AutoConfig.from_pretrained(model_name)

            if hidden_dropout is not None:
                config.hidden_dropout_prob = hidden_dropout
            if attention_dropout is not None:
                config.attention_probs_dropout_prob = attention_dropout

            encoder = AutoModel.from_pretrained(model_name, config=config)
            logger.info(
                f"Loaded {model_name}: "
                f"hidden={config.hidden_size}, "
                f"layers={config.num_hidden_layers}, "
                f"heads={config.num_attention_heads}"
            )
            return encoder

        except ImportError:
            raise ImportError(
                "transformers package required. "
                "Run: pip install transformers"
            )
        except OSError as e:
            raise OSError(
                f"Could not load model '{model_name}'. "
                f"Check internet access or local model path.\n{e}"
            )

    # ── Parameter freezing ───────────────────────────────────────────────────

    def _apply_freezing(
        self,
        freeze_embeddings: bool,
        freeze_encoder_layers: int,
    ) -> None:
        """
        Freeze selected encoder layers to enable parameter-efficient training.

        Strategy:
            1. Optionally freeze word/position embeddings (most general features)
            2. Freeze bottom N encoder layers (earlier = more language-general)
            3. Task heads are always trainable
        """
        if freeze_embeddings:
            for param in self.encoder.embeddings.parameters():
                param.requires_grad = False
            logger.info("Frozen: encoder embeddings")

        if freeze_encoder_layers > 0:
            encoder_layers = self.encoder.encoder.layer
            num_layers = len(encoder_layers)

            if freeze_encoder_layers > num_layers:
                logger.warning(
                    f"freeze_encoder_layers={freeze_encoder_layers} > "
                    f"num_layers={num_layers}. Clamping."
                )
                freeze_encoder_layers = num_layers

            for i, layer in enumerate(encoder_layers[:freeze_encoder_layers]):
                for param in layer.parameters():
                    param.requires_grad = False

            logger.info(
                f"Frozen: encoder layers 0–{freeze_encoder_layers-1} "
                f"of {num_layers}"
            )

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids:      torch.Tensor,           # (B, seq_len)
        attention_mask: torch.Tensor,           # (B, seq_len)
        lid_token_ids:  Optional[torch.Tensor] = None,   # (B, seq_len)
        return_hidden:  bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Args:
            input_ids:      Tokenized subword IDs
            attention_mask: 1 for real tokens, 0 for padding
            lid_token_ids:  Per-subword language IDs (required if lid_enabled)
            return_hidden:  Also return full encoder hidden states

        Returns:
            Dict with:
                sentiment_logits  : (B, 3)
                sarcasm_logit     : (B, 1)
                pooled_output     : (B, hidden_size) — [CLS] embedding
                encoder_output    : (B, seq_len, hidden_size) if return_hidden
        """
        # ── Encoder forward pass ─────────────────────────────────────────
        encoder_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # [CLS] token: first token of last hidden state
        cls_output = encoder_out.last_hidden_state[:, 0, :]   # (B, hidden_size)

        # ── LID fusion ───────────────────────────────────────────────────
        if self.lid_enabled and self.lid_fusion is not None:
            if lid_token_ids is None:
                raise ValueError(
                    "lid_token_ids required when lid_enabled=True. "
                    "Pass lid_token_ids or disable LID integration."
                )
            lid_repr = self.lid_fusion(lid_token_ids, attention_mask)  # (B, lid_output_dim)
            classifier_input = torch.cat([cls_output, lid_repr], dim=-1)
        else:
            classifier_input = cls_output

        # ── Task heads ───────────────────────────────────────────────────
        sentiment_logits = self.sentiment_head(classifier_input)   # (B, 3)
        sarcasm_logit    = self.sarcasm_head(classifier_input)     # (B, 1)

        outputs = {
            "sentiment_logits": sentiment_logits,
            "sarcasm_logit":    sarcasm_logit,
            "pooled_output":    cls_output,
        }

        if return_hidden:
            outputs["encoder_hidden_states"] = encoder_out.last_hidden_state

        return outputs

    # ── Prediction helpers ───────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        lid_token_ids:  Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Run inference and return class probabilities.

        Returns:
            Dict with:
                sentiment_probs : (B, 3)  — softmax probabilities
                sarcasm_prob    : (B,)    — sigmoid probability of sarcasm
                sentiment_pred  : (B,)    — argmax class index
                sarcasm_pred    : (B,)    — binary 0/1 prediction
        """
        self.eval()
        outputs = self.forward(input_ids, attention_mask, lid_token_ids)

        sentiment_probs = torch.softmax(outputs["sentiment_logits"], dim=-1)
        sarcasm_prob    = torch.sigmoid(outputs["sarcasm_logit"]).squeeze(-1)

        return {
            "sentiment_probs": sentiment_probs,
            "sarcasm_prob":    sarcasm_prob,
            "sentiment_pred":  sentiment_probs.argmax(dim=-1),
            "sarcasm_pred":    (sarcasm_prob >= 0.5).long(),
        }

    # ── Serialisation ────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save model state dict and config to disk."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save state dict
        torch.save(self.state_dict(), save_path / "model.pt")

        # Save encoder config for reloading
        self.encoder.config.save_pretrained(save_path)

        logger.info(f"Model saved to: {save_path}")

    @classmethod
    def load(
        cls,
        path: str,
        model_config: dict,
        device: str = "cpu",
    ) -> "MultiTaskSentimentModel":
        """Load a saved model from disk."""
        model = cls.from_config(model_config)
        state = torch.load(Path(path) / "model.pt", map_location=device)
        model.load_state_dict(state)
        model.eval()
        logger.info(f"Model loaded from: {path}")
        return model

    @classmethod
    def from_config(cls, model_config: dict) -> "MultiTaskSentimentModel":
        """
        Instantiate from model_config.yaml dict.

        Args:
            model_config: Loaded model_config.yaml content

        Returns:
            Configured MultiTaskSentimentModel
        """
        cfg = model_config["model"]
        lid = cfg.get("lid_integration", {})

        return cls(
            base_model_name=cfg["base_model"],
            sentiment_num_classes=cfg["sentiment"]["num_classes"],
            sentiment_hidden_dim=cfg["sentiment"]["hidden_dim"],
            sentiment_dropout=cfg["sentiment"]["dropout"],
            sentiment_activation=cfg["sentiment"]["activation"],
            sarcasm_hidden_dim=cfg["sarcasm"]["hidden_dim"],
            sarcasm_dropout=cfg["sarcasm"]["dropout"],
            sarcasm_activation=cfg["sarcasm"]["activation"],
            lid_enabled=lid.get("enabled", True),
            lid_num_languages=lid.get("num_languages", 12),
            lid_embed_dim=lid.get("lid_embedding_dim", 32),
            lid_output_dim=lid.get("lid_embedding_dim", 32),
            freeze_embeddings=cfg.get("freeze_embeddings", False),
            freeze_encoder_layers=cfg.get("freeze_encoder_layers", 0),
            hidden_dropout_prob=cfg.get("hidden_dropout_prob"),
            attention_dropout_prob=cfg.get("attention_probs_dropout_prob"),
        )

    # ── Utilities ────────────────────────────────────────────────────────────

    def _count_parameters(self) -> Tuple[int, int]:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def get_optimizer_param_groups(
        self,
        encoder_lr:   float = 2e-5,
        head_lr:      float = 1e-4,
        weight_decay: float = 0.01,
    ) -> list:
        """
        Return parameter groups with different learning rates.

        Rationale:
            - Encoder (pre-trained): small LR to avoid catastrophic forgetting
            - Task heads (randomly initialised): larger LR for faster convergence
            - Biases and LayerNorm: no weight decay (standard practice)

        Args:
            encoder_lr:   LR for pre-trained encoder parameters
            head_lr:      LR for task head parameters
            weight_decay: L2 regularisation (applied to weights only)

        Returns:
            List of param group dicts for AdamW
        """
        no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}

        encoder_params_decay    = []
        encoder_params_no_decay = []
        head_params_decay       = []
        head_params_no_decay    = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            is_head = any(
                name.startswith(h)
                for h in ("sentiment_head", "sarcasm_head", "lid_fusion")
            )
            no_dec = any(nd in name for nd in no_decay)

            if is_head:
                (head_params_no_decay if no_dec else head_params_decay).append(param)
            else:
                (encoder_params_no_decay if no_dec else encoder_params_decay).append(param)

        return [
            {"params": encoder_params_decay,    "lr": encoder_lr, "weight_decay": weight_decay},
            {"params": encoder_params_no_decay, "lr": encoder_lr, "weight_decay": 0.0},
            {"params": head_params_decay,       "lr": head_lr,    "weight_decay": weight_decay},
            {"params": head_params_no_decay,    "lr": head_lr,    "weight_decay": 0.0},
        ]
