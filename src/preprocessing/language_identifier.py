"""
Token-Level Language Identification (LID) Module

Implements multi-strategy language identification at both token and sentence level.
Designed specifically for code-switched, multilingual social media text.

Architecture:
    TokenLevelLID
    ├── Primary: FastText (lid.176.bin) - fast, multilingual, token-tolerant
    ├── Fallback: langid (Python-native, no binary deps)
    └── Heuristics: script detection, known multilingual markers

Rationale for FastText:
    - Supports 176 languages
    - Works on short strings (suitable for tokens)
    - Outperforms langdetect on noisy, short text
    - Confidence scores available for thresholding
"""

import re
import unicodedata
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import get_logger

logger = get_logger("language_identifier")


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class TokenLanguage:
    """
    Language identification result for a single token.
    
    Attributes:
        token:      The original token string
        language:   ISO 639-1 language code (e.g., 'en', 'hi')
        confidence: Probability score [0.0, 1.0]
        script:     Unicode script name (e.g., 'Latin', 'Devanagari')
        is_ambiguous: True if confidence < threshold or token too short
    """
    token: str
    language: str
    confidence: float
    script: str
    is_ambiguous: bool = False


@dataclass
class SentenceLID:
    """
    Language identification result for a full sentence.
    
    Attributes:
        token_results:      Per-token LID results
        dominant_language:  Most frequent language in the sentence
        language_counts:    Dict mapping lang code → token count
        is_code_switched:   True if multiple languages detected
        switch_points:      Indices where language changes occur
    """
    token_results: List[TokenLanguage]
    dominant_language: str
    language_counts: Dict[str, int] = field(default_factory=dict)
    is_code_switched: bool = False
    switch_points: List[int] = field(default_factory=list)

    def language_sequence(self) -> List[str]:
        """Return ordered list of language codes per token."""
        return [t.language for t in self.token_results]

    def unique_languages(self) -> List[str]:
        """Return sorted list of unique languages detected."""
        return sorted(set(t.language for t in self.token_results
                          if not t.is_ambiguous))


# ─────────────────────────────────────────────
# Script Detection (Unicode-based heuristic)
# ─────────────────────────────────────────────

class ScriptDetector:
    """
    Detects Unicode script for tokens using character-level analysis.

    Used as a lightweight pre-filter before heavier LID models.
    Extremely fast (no ML); handles emoji, Devanagari, Arabic, CJK, etc.
    """

    # Regex patterns for major script families
    SCRIPT_PATTERNS: Dict[str, re.Pattern] = {
        "Devanagari": re.compile(r"[\u0900-\u097F\u0980-\u09FF]"),   # Hindi, Marathi, Bengali
        "Arabic":     re.compile(r"[\u0600-\u06FF\u0750-\u077F]"),   # Arabic, Urdu, Persian
        "CJK":        re.compile(r"[\u4E00-\u9FFF\u3400-\u4DBF]"),   # Chinese, Japanese Kanji
        "Hangul":     re.compile(r"[\uAC00-\uD7AF]"),                # Korean
        "Hiragana":   re.compile(r"[\u3041-\u3096]"),                # Japanese Hiragana
        "Katakana":   re.compile(r"[\u30A0-\u30FF]"),                # Japanese Katakana
        "Cyrillic":   re.compile(r"[\u0400-\u04FF]"),                # Russian, Ukrainian
        "Greek":      re.compile(r"[\u0370-\u03FF]"),                # Greek
        "Tamil":      re.compile(r"[\u0B80-\u0BFF]"),                # Tamil
        "Telugu":     re.compile(r"[\u0C00-\u0C7F]"),                # Telugu
        "Latin":      re.compile(r"[a-zA-ZÀ-ÿ]"),                   # English, Spanish, French…
        "Emoji":      re.compile(
            r"[\U0001F300-\U0001F9FF\U00002600-\U000027BF"
            r"\U0001FA00-\U0001FA9F\U00002702-\U000027B0]"
        ),
    }

    # Script → most probable language(s)
    SCRIPT_TO_LANG: Dict[str, str] = {
        "Devanagari": "hi",
        "Arabic":     "ar",
        "CJK":        "zh",
        "Hangul":     "ko",
        "Hiragana":   "ja",
        "Katakana":   "ja",
        "Cyrillic":   "ru",
        "Greek":      "el",
        "Tamil":      "ta",
        "Telugu":     "te",
        "Latin":      "en",   # Broad; refined by LID model
        "Emoji":      "emoji",
    }

    @classmethod
    def detect_script(cls, token: str) -> str:
        """
        Detect dominant Unicode script in a token.

        Args:
            token: Input token string

        Returns:
            Script name string (e.g., 'Devanagari', 'Latin', 'Unknown')
        """
        if not token.strip():
            return "Unknown"

        script_votes: Dict[str, int] = {}
        for script_name, pattern in cls.SCRIPT_PATTERNS.items():
            matches = len(pattern.findall(token))
            if matches > 0:
                script_votes[script_name] = matches

        if not script_votes:
            return "Unknown"

        return max(script_votes, key=script_votes.get)

    @classmethod
    def script_to_language(cls, script: str) -> Optional[str]:
        """
        Map a detected script to a most-probable language code.

        Args:
            script: Script name

        Returns:
            ISO 639-1 code, or None if unknown
        """
        return cls.SCRIPT_TO_LANG.get(script)


# ─────────────────────────────────────────────
# Core Token-Level LID
# ─────────────────────────────────────────────

class TokenLevelLID:
    """
    Token-Level Language Identification engine.

    Strategy (in priority order):
      1. Script detection  → high-confidence non-Latin scripts (Devanagari, Arabic…)
      2. FastText model    → Latin-script tokens and ambiguous cases
      3. langid library    → fallback when FastText unavailable
      4. Heuristics        → short tokens, punctuation, numbers, emoji

    Usage:
        lid = TokenLevelLID(confidence_threshold=0.5)
        result = lid.identify_sentence("yaar this movie is bakwas!")
        print(result.is_code_switched)  # True
        print(result.language_sequence())  # ['hi', 'en', 'en', 'en', 'hi']
    """

    # Tokens that are language-agnostic (skip LID)
    LANG_AGNOSTIC_PATTERNS = re.compile(
        r"^("
        r"https?://\S+"       # URLs
        r"|@\w+"              # Mentions
        r"|#\w+"              # Hashtags
        r"|\d+[\d.,\-/:%]*"  # Numbers
        r"|[^\w\s]+"          # Pure punctuation
        r")$",
        re.IGNORECASE
    )

    SUPPORTED_LANGUAGES = {
        "en", "hi", "es", "fr", "ar", "pt",
        "de", "zh", "ja", "ko", "it", "ru",
        "tr", "nl", "pl", "ta", "te", "bn"
    }

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        min_token_length: int = 3,
        fasttext_model_path: Optional[str] = None,
    ):
        """
        Initialize the LID engine.

        Args:
            confidence_threshold: Minimum confidence to accept a prediction.
                                  Tokens below threshold are marked ambiguous.
            min_token_length:     Tokens shorter than this use heuristics only.
            fasttext_model_path:  Path to FastText lid.176.bin model file.
                                  If None, falls back to langid.
        """
        self.confidence_threshold = confidence_threshold
        self.min_token_length = min_token_length
        self.fasttext_model_path = fasttext_model_path

        self._fasttext_model = None
        self._langid_available = False
        self._backend = "heuristic"

        self._initialize_backend()

    # ── Initialization ──────────────────────────────────────────────────────

    def _initialize_backend(self) -> None:
        """
        Initialize the best available LID backend.
        Priority: FastText > langid > heuristics.
        """
        # Try FastText first
        if self._try_load_fasttext():
            self._backend = "fasttext"
            logger.info("LID backend: FastText")
            return

        # Fall back to langid
        if self._try_load_langid():
            self._backend = "langid"
            logger.info("LID backend: langid (FastText model not available)")
            return

        # Last resort: heuristics only
        self._backend = "heuristic"
        logger.warning(
            "LID backend: heuristic only. "
            "Install fasttext or langid for better accuracy."
        )

    def _try_load_fasttext(self) -> bool:
        """Attempt to load FastText LID model."""
        try:
            import fasttext
            import fasttext.util

            model_path = self.fasttext_model_path or "models/lid.176.bin"
            if Path(model_path).exists():
                # Suppress FastText stdout during model load
                import os
                devnull = open(os.devnull, "w")
                import contextlib
                with contextlib.redirect_stdout(devnull):
                    self._fasttext_model = fasttext.load_model(model_path)
                logger.info(f"FastText model loaded from: {model_path}")
                return True
            else:
                logger.debug(f"FastText model not found at: {model_path}")
                return False
        except ImportError:
            logger.debug("fasttext package not installed")
            return False
        except Exception as e:
            logger.warning(f"FastText load failed: {e}")
            return False

    def _try_load_langid(self) -> bool:
        """Check if langid is available."""
        try:
            import langid
            langid.classify("test")   # warm-up
            self._langid_available = True
            return True
        except ImportError:
            logger.debug("langid package not installed")
            return False
        except Exception as e:
            logger.warning(f"langid init failed: {e}")
            return False

    # ── Public API ──────────────────────────────────────────────────────────

    def identify_token(self, token: str) -> TokenLanguage:
        """
        Identify the language of a single token.

        Args:
            token: Input token (word or sub-word)

        Returns:
            TokenLanguage with language, confidence, script info
        """
        script = ScriptDetector.detect_script(token)

        # ── Heuristic fast-paths ───────────────────────────────
        # Emoji checked FIRST (before lang-agnostic pattern which catches punctuation)
        if script == "Emoji":
            return TokenLanguage(
                token=token, language="emoji", confidence=1.0,
                script=script, is_ambiguous=False
            )

        # Punctuation, URLs, numbers → language-agnostic
        if self.LANG_AGNOSTIC_PATTERNS.match(token):
            return TokenLanguage(
                token=token, language="und", confidence=1.0,
                script=script, is_ambiguous=False
            )

        # Non-Latin scripts: script detection is highly reliable
        if script not in ("Latin", "Unknown"):
            lang = ScriptDetector.script_to_language(script) or "und"
            return TokenLanguage(
                token=token, language=lang, confidence=0.95,
                script=script, is_ambiguous=False
            )

        # Very short tokens → heuristic only (LID unreliable)
        if len(token) < self.min_token_length:
            return TokenLanguage(
                token=token, language="en", confidence=0.4,
                script=script, is_ambiguous=True
            )

        # ── ML backends for Latin-script tokens ───────────────
        if self._backend == "fasttext":
            return self._identify_fasttext(token, script)
        elif self._backend == "langid":
            return self._identify_langid(token, script)
        else:
            return self._identify_heuristic(token, script)

    def identify_sentence(self, text: str) -> SentenceLID:
        """
        Identify languages for every whitespace-split token in a sentence.

        Args:
            text: Input sentence (may be code-switched)

        Returns:
            SentenceLID with per-token results and aggregate statistics
        """
        if not text or not text.strip():
            return SentenceLID(
                token_results=[],
                dominant_language="und",
                language_counts={},
                is_code_switched=False,
                switch_points=[]
            )

        tokens = text.split()
        token_results = [self.identify_token(t) for t in tokens]

        return self._aggregate_results(token_results)

    def identify_batch(self, texts: List[str]) -> List[SentenceLID]:
        """
        Identify languages for a list of sentences.

        Args:
            texts: List of input sentences

        Returns:
            List of SentenceLID results
        """
        results = []
        for text in texts:
            try:
                results.append(self.identify_sentence(text))
            except Exception as e:
                logger.error(f"LID failed for text '{text[:50]}...': {e}")
                results.append(SentenceLID(
                    token_results=[],
                    dominant_language="und",
                    is_code_switched=False,
                    switch_points=[]
                ))
        return results

    # ── Private: ML backends ────────────────────────────────────────────────

    def _identify_fasttext(self, token: str, script: str) -> TokenLanguage:
        """Run FastText LID on a single token."""
        try:
            # FastText expects single-line input
            clean = token.replace("\n", " ").strip()
            labels, probs = self._fasttext_model.predict(clean, k=1)

            raw_label = labels[0]                        # "__label__en"
            lang = raw_label.replace("__label__", "")
            confidence = float(probs[0])

            # Normalize to supported set; fall back to 'en' for Latin
            if lang not in self.SUPPORTED_LANGUAGES:
                lang = "en"
                confidence = max(confidence * 0.7, 0.3)

            is_ambiguous = confidence < self.confidence_threshold

            return TokenLanguage(
                token=token, language=lang,
                confidence=confidence, script=script,
                is_ambiguous=is_ambiguous
            )
        except Exception as e:
            logger.debug(f"FastText prediction failed for '{token}': {e}")
            return self._identify_heuristic(token, script)

    def _identify_langid(self, token: str, script: str) -> TokenLanguage:
        """Run langid LID on a single token."""
        try:
            import langid
            lang, confidence_raw = langid.classify(token)

            # langid returns log-probability; convert to ~[0,1]
            confidence = max(0.0, min(1.0, (confidence_raw + 10.0) / 10.0))

            if lang not in self.SUPPORTED_LANGUAGES:
                lang = "en"
                confidence = 0.4

            is_ambiguous = confidence < self.confidence_threshold

            return TokenLanguage(
                token=token, language=lang,
                confidence=confidence, script=script,
                is_ambiguous=is_ambiguous
            )
        except Exception as e:
            logger.debug(f"langid failed for '{token}': {e}")
            return self._identify_heuristic(token, script)

    def _identify_heuristic(self, token: str, script: str) -> TokenLanguage:
        """Heuristic LID: treat Latin-script as English by default."""
        lang = ScriptDetector.script_to_language(script) or "en"
        return TokenLanguage(
            token=token, language=lang,
            confidence=0.35, script=script, is_ambiguous=True
        )

    # ── Private: Aggregation ─────────────────────────────────────────────────

    def _aggregate_results(
        self, token_results: List[TokenLanguage]
    ) -> SentenceLID:
        """
        Compute sentence-level statistics from per-token results.

        Detects code-switching, dominant language, and switch points.
        Language-agnostic tokens ('und', 'emoji') are excluded from counts.
        """
        # Count substantive language tokens (exclude und/emoji)
        lang_counts: Dict[str, int] = {}
        for tr in token_results:
            if tr.language not in ("und", "emoji") and not tr.is_ambiguous:
                lang_counts[tr.language] = lang_counts.get(tr.language, 0) + 1

        dominant = max(lang_counts, key=lang_counts.get) if lang_counts else "und"

        # Detect switch points
        switch_points: List[int] = []
        prev_lang = None
        for idx, tr in enumerate(token_results):
            if tr.language in ("und", "emoji") or tr.is_ambiguous:
                continue
            if prev_lang is not None and tr.language != prev_lang:
                switch_points.append(idx)
            prev_lang = tr.language

        is_code_switched = len(set(
            tr.language for tr in token_results
            if tr.language not in ("und", "emoji") and not tr.is_ambiguous
        )) > 1

        return SentenceLID(
            token_results=token_results,
            dominant_language=dominant,
            language_counts=lang_counts,
            is_code_switched=is_code_switched,
            switch_points=switch_points,
        )

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def backend(self) -> str:
        """Return the active LID backend name."""
        return self._backend


# ─────────────────────────────────────────────
# Singleton convenience accessor
# ─────────────────────────────────────────────

_default_lid: Optional[TokenLevelLID] = None


def get_lid(
    confidence_threshold: float = 0.5,
    min_token_length: int = 3,
    fasttext_model_path: Optional[str] = None,
) -> TokenLevelLID:
    """
    Return (or create) a default singleton TokenLevelLID instance.

    Useful for avoiding repeated model loading across modules.
    """
    global _default_lid
    if _default_lid is None:
        _default_lid = TokenLevelLID(
            confidence_threshold=confidence_threshold,
            min_token_length=min_token_length,
            fasttext_model_path=fasttext_model_path,
        )
    return _default_lid


# ─────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    lid = get_lid()
    print(f"Active backend: {lid.backend}\n")

    test_sentences = [
        "yaar this movie was totally bakwas!",
        "This product is absolutely amazing! 😍",
        "Oh great, another delay. Just what I needed! 🙄",
        "मुझे यह product बहुत pasand आया",
        "Qué película tan aburrida, de verdad.",
        "@user check this out: https://example.com",
    ]

    for sentence in test_sentences:
        result = lid.identify_sentence(sentence)
        print(f"Input:       {sentence}")
        print(f"Dominant:    {result.dominant_language}")
        print(f"Languages:   {result.unique_languages()}")
        print(f"Code-switch: {result.is_code_switched}")
        print(f"Sequence:    {result.language_sequence()}")
        print("-" * 60)
