"""
Sarcasm Feature Extractors

Specialized feature engineering for detecting sarcasm in text.

Sarcasm signals (Riloff et al., 2013; Joshi et al., 2017):
    1. Sentiment-sarcasm incongruence: positive words + negative context
    2. Punctuation patterns: excessive !!! or ??? as emphasis markers
    3. Emoji-text mismatch: 😒 + "Great job!" (eye-roll emoji contradicts praise)
    4. ALL-CAPS intensity: "This is JUST what I needed!"
    5. Contrast embeddings: literal meaning vs. intended meaning divergence

Architecture:
    SarcasmFeatureExtractor
    ├── PunctuationFeatures (!!!, ???, ...)
    ├── EmojiTextMismatchFeatures
    ├── IntensityFeatures (ALL-CAPS, elongation)
    └── SentimentIncongruenceFeatures

These features are concatenated to the sarcasm head's input to provide
explicit sarcasm cues beyond what the transformer learns implicitly.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    class _NNStub:
        class Module: pass
    nn = _NNStub()  # type: ignore

from utils.logger import get_logger

logger = get_logger("sarcasm_features")


# ═══════════════════════════════════════════════════════════
# Rule-based Feature Extraction (applied to raw text)
# ═══════════════════════════════════════════════════════════

class SarcasmSignalExtractor:
    """
    Extract hand-crafted sarcasm signals from raw text (pre-tokenization).

    Returns a feature dict with:
        - punctuation_intensity: count of !!!, ???, ...
        - allcaps_ratio: fraction of words in ALL-CAPS
        - elongation_count: count of elongated words (sooo, noooo)
        - emoji_sentiment: dominant emoji sentiment (pos/neg/neu)

    These features are passed to the model as auxiliary inputs.
    """

    # Regex patterns
    EXCLAMATION_PATTERN = re.compile(r"!{2,}")       # !!, !!!, !!!!
    QUESTION_PATTERN    = re.compile(r"\?{2,}")      # ??, ???
    ELLIPSIS_PATTERN    = re.compile(r"\.{3,}")      # ..., ....
    ELONGATION_PATTERN  = re.compile(r"(.)\1{2,}")   # sooo, noooo
    ALLCAPS_PATTERN     = re.compile(r"\b[A-Z]{2,}\b")

    # Emoji sentiment (simplified — extend with full lexicon in production)
    EMOJI_SENTIMENT_MAP = {
        # Negative sarcasm markers
        "😒": -1,  # unamused
        "🙄": -1,  # eye roll
        "😑": -1,  # expressionless
        "😐": -1,  # neutral face (sarcasm marker)
        "🤨": -1,  # raised eyebrow
        "😏": -1,  # smirk
        # Positive (non-sarcastic usually)
        "😍": 1,
        "😊": 1,
        "❤️": 1,
        "🥰": 1,
        # Negative (genuine frustration)
        "😡": -1,
        "😤": -1,
        "😭": -1,
    }

    def extract(self, text: str) -> Dict[str, float]:
        """
        Extract sarcasm features from raw text.

        Args:
            text: Raw input string (not tokenized)

        Returns:
            Dict with normalized feature values
        """
        if not text or not text.strip():
            return self._zero_features()

        words = text.split()
        num_words = max(len(words), 1)

        # Punctuation intensity
        exclamations = len(self.EXCLAMATION_PATTERN.findall(text))
        questions    = len(self.QUESTION_PATTERN.findall(text))
        ellipses     = len(self.ELLIPSIS_PATTERN.findall(text))
        punct_intensity = (exclamations + questions + ellipses) / num_words

        # ALL-CAPS ratio
        allcaps_count = len(self.ALLCAPS_PATTERN.findall(text))
        allcaps_ratio = allcaps_count / num_words

        # Elongation count (normalized)
        elongation_count = len(self.ELONGATION_PATTERN.findall(text)) / num_words

        # Emoji sentiment
        emoji_scores = [
            self.EMOJI_SENTIMENT_MAP.get(char, 0) for char in text
        ]
        emoji_sentiment = sum(emoji_scores) / max(len(emoji_scores), 1)

        return {
            "punctuation_intensity": min(punct_intensity, 1.0),
            "allcaps_ratio":         min(allcaps_ratio, 1.0),
            "elongation_count":      min(elongation_count, 1.0),
            "emoji_sentiment":       max(-1.0, min(emoji_sentiment, 1.0)),
        }

    @staticmethod
    def _zero_features() -> Dict[str, float]:
        return {
            "punctuation_intensity": 0.0,
            "allcaps_ratio":         0.0,
            "elongation_count":      0.0,
            "emoji_sentiment":       0.0,
        }

    def extract_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Extract features for a batch of texts."""
        return [self.extract(text) for text in texts]


# ═══════════════════════════════════════════════════════════
# Neural Feature Modules (integrated into model)
# ═══════════════════════════════════════════════════════════

class SentimentIncongruenceModule(nn.Module):
    """
    Detect sentiment-sarcasm incongruence by comparing:
        - Sentiment head's predictions (positive/negative)
        - Sarcasm head's current state

    Incongruence signal: positive sentiment + high sarcasm probability
    → likely sarcasm (e.g., "Great job!" said sarcastically)

    This module is trained end-to-end with backprop through both heads.
    """

    def __init__(self, input_dim: int = 3 + 1):
        """
        Args:
            input_dim: sentiment_probs (3) + sarcasm_logit (1)
        """
        super().__init__()
        if torch is None:
            return
        self.projection = nn.Linear(input_dim, 1)
        self.activation = nn.Tanh()

    def forward(
        self,
        sentiment_logits: "torch.Tensor",  # (B, 3)
        sarcasm_logit:    "torch.Tensor",  # (B, 1)
    ) -> "torch.Tensor":
        """
        Compute incongruence score.

        Returns:
            (B, 1) incongruence feature
        """
        if torch is None:
            raise RuntimeError("torch not available")

        sentiment_probs = torch.softmax(sentiment_logits, dim=-1)  # (B, 3)

        # Concatenate: [neg_prob, neu_prob, pos_prob, sarcasm_logit]
        combined = torch.cat([sentiment_probs, sarcasm_logit], dim=-1)  # (B, 4)

        incongruence = self.projection(combined)   # (B, 1)
        incongruence = self.activation(incongruence)

        return incongruence


class SarcasmAuxiliaryFeatures(nn.Module):
    """
    Embed hand-crafted sarcasm features (from SarcasmSignalExtractor)
    and project them to a dense representation.

    Input: 4-dim feature vector per sample
    Output: projected feature vector to append to sarcasm head input
    """

    def __init__(self, output_dim: int = 16):
        super().__init__()
        if torch is None:
            return
        # 4 input features from SarcasmSignalExtractor
        self.projection = nn.Sequential(
            nn.Linear(4, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, features: "torch.Tensor") -> "torch.Tensor":
        """
        Args:
            features: (B, 4) tensor from SarcasmSignalExtractor

        Returns:
            (B, output_dim) projected features
        """
        if torch is None:
            raise RuntimeError("torch not available")
        return self.projection(features)


# ═══════════════════════════════════════════════════════════
# Enhanced Sarcasm Head (integrates all features)
# ═══════════════════════════════════════════════════════════

class EnhancedSarcasmHead(nn.Module):
    """
    Enhanced sarcasm head with auxiliary features and incongruence detection.

    Architecture:
        [CLS] embedding (+ LID features if enabled)
            ↓
        Concatenate: hand-crafted features (projected)
            ↓
        Linear → ReLU → Dropout → Linear(1)
            ↓
        raw sarcasm logit

    Additionally computes sentiment-incongruence feature after both heads
    run (requires two-pass or joint training).
    """

    def __init__(
        self,
        input_dim:  int = 768,
        hidden_dim: int = 128,
        aux_dim:    int = 16,     # projected auxiliary features
        dropout:    float = 0.2,
        use_aux_features: bool = True,
    ):
        super().__init__()
        if torch is None:
            return

        self.use_aux_features = use_aux_features

        if use_aux_features:
            self.aux_projection = SarcasmAuxiliaryFeatures(output_dim=aux_dim)
            classifier_input_dim = input_dim + aux_dim
        else:
            self.aux_projection = None
            classifier_input_dim = input_dim

        self.layer_norm = nn.LayerNorm(classifier_input_dim)
        self.dropout1   = nn.Dropout(dropout)
        self.dense      = nn.Linear(classifier_input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout2   = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)

        self._init_weights()
        logger.debug(
            f"EnhancedSarcasmHead: {input_dim}(+{aux_dim if use_aux_features else 0})"
            f"→{hidden_dim}→1 | dropout={dropout}"
        )

    def _init_weights(self):
        if torch is None:
            return
        nn.init.xavier_uniform_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        cls_embedding:     "torch.Tensor",           # (B, input_dim)
        aux_features:      Optional["torch.Tensor"] = None,  # (B, 4)
    ) -> "torch.Tensor":
        """
        Args:
            cls_embedding: Encoder [CLS] output (optionally with LID)
            aux_features:  Hand-crafted features (B, 4) from SarcasmSignalExtractor

        Returns:
            (B, 1) sarcasm logit
        """
        if torch is None:
            raise RuntimeError("torch not available")

        if self.use_aux_features and aux_features is not None:
            # Project and concatenate auxiliary features
            aux_proj = self.aux_projection(aux_features)         # (B, aux_dim)
            x = torch.cat([cls_embedding, aux_proj], dim=-1)     # (B, input+aux)
        else:
            x = cls_embedding

        x = self.layer_norm(x)
        x = self.dropout1(x)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout2(x)
        return self.classifier(x)


# ═══════════════════════════════════════════════════════════
# Smoke-test
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test signal extractor
    extractor = SarcasmSignalExtractor()

    test_texts = [
        "Oh GREAT, another delay!!!",
        "This is JUST what I NEEDED 🙄",
        "Amazing product, highly recommend!",
        "sooooo good omg!!!",
    ]

    print("Sarcasm Signal Extraction:")
    for text in test_texts:
        features = extractor.extract(text)
        print(f"  '{text}'")
        print(f"    {features}")

    if torch is not None:
        # Test neural modules
        B = 4
        input_dim = 768

        # EnhancedSarcasmHead
        head = EnhancedSarcasmHead(input_dim=input_dim, use_aux_features=True)
        cls_emb = torch.randn(B, input_dim)
        aux_feat = torch.rand(B, 4)   # 4 hand-crafted features
        logit = head(cls_emb, aux_feat)
        print(f"\nEnhancedSarcasmHead output: {logit.shape}")
        assert logit.shape == (B, 1)

        # SentimentIncongruenceModule
        incongruence_mod = SentimentIncongruenceModule()
        sent_logits = torch.randn(B, 3)
        sarc_logit  = torch.randn(B, 1)
        incong = incongruence_mod(sent_logits, sarc_logit)
        print(f"Incongruence feature: {incong.shape}")
        assert incong.shape == (B, 1)

        print("\n✓ Sarcasm feature modules smoke-test passed!")
    else:
        print("\n⚠ torch not available — skipping neural module tests")
