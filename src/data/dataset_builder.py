"""
Dataset Builder Module

Constructs, preprocesses, and serves datasets for multi-task
sentiment + sarcasm training. Integrates LID and normalization
into a single reproducible pipeline.

DataSample schema:
    {
        'text':            str,             # raw input
        'normalized_text': str,             # after normalization
        'tokens':          List[str],       # whitespace-split tokens
        'token_languages': List[str],       # per-token ISO 639-1 codes
        'dominant_language': str,           # sentence-level dominant lang
        'is_code_switched': bool,           # True if >1 language
        'sentiment_label': int,             # 0=neg, 1=neu, 2=pos
        'sarcasm_label':   int,             # 0=not sarcastic, 1=sarcastic
        'input_ids':       List[int],       # tokenizer output
        'attention_mask':  List[int],       # tokenizer output
        'lid_token_ids':   List[int],       # encoded language IDs per token
    }
"""

import os
import json
import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import get_logger
from preprocessing.language_identifier import TokenLevelLID, get_lid
from preprocessing.text_normalizer import TextNormalizer

logger = get_logger("dataset_builder")


# ─────────────────────────────────────────────
# Label Mappings
# ─────────────────────────────────────────────

SENTIMENT_LABEL_MAP = {
    # String variants → int
    "negative": 0, "neg": 0, "0": 0, "-1": 0,
    "neutral":  1, "neu": 1, "1": 1,
    "positive": 2, "pos": 2, "2": 2, "+1": 2,
}

SARCASM_LABEL_MAP = {
    "not_sarcastic": 0, "0": 0, "false": 0, "no": 0,
    "sarcastic":     1, "1": 1, "true":  1, "yes": 1,
}

SENTIMENT_ID_TO_LABEL = {0: "negative", 1: "neutral", 2: "positive"}
SARCASM_ID_TO_LABEL   = {0: "not_sarcastic", 1: "sarcastic"}

# Language code → integer ID for LID feature embedding
LANGUAGE_ID_MAP = {
    "en": 0,  "hi": 1,  "es": 2,  "fr": 3,
    "ar": 4,  "pt": 5,  "de": 6,  "zh": 7,
    "ja": 8,  "ko": 9,  "und": 10, "emoji": 11,
}
UNK_LANGUAGE_ID = 10  # 'und' = undetermined


# ─────────────────────────────────────────────
# Single Sample Processor
# ─────────────────────────────────────────────

class SampleProcessor:
    """
    Processes a single raw text sample through:
        LID → normalization → tokenization → LID embedding
    """

    def __init__(
        self,
        lid: TokenLevelLID,
        normalizer: TextNormalizer,
        tokenizer,                          # HuggingFace tokenizer
        max_length: int = 128,
        language_id_map: Dict[str, int] = LANGUAGE_ID_MAP,
    ):
        self.lid = lid
        self.normalizer = normalizer
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.language_id_map = language_id_map

    def process(
        self,
        text: str,
        sentiment_label: Optional[int] = None,
        sarcasm_label: Optional[int] = None,
    ) -> Dict:
        """
        Full processing pipeline for one sample.

        Args:
            text:            Raw input string
            sentiment_label: Integer label (0/1/2) or None for inference
            sarcasm_label:   Integer label (0/1) or None for inference

        Returns:
            Dict with all model inputs and metadata
        """
        if not text or not text.strip():
            raise ValueError("Empty or whitespace-only text")

        # ── Step 1: Token-level LID ─────────────────────────────
        lid_result = self.lid.identify_sentence(text)
        token_languages = lid_result.language_sequence()

        # ── Step 2: LID-guided normalization ────────────────────
        normalized = self.normalizer.normalize(text, token_languages)

        # ── Step 3: HuggingFace tokenization ────────────────────
        encoding = self.tokenizer(
            normalized,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
        )

        # ── Step 4: Build LID feature tensor ────────────────────
        # Align token-level language IDs to subword token positions.
        # Strategy: word-piece tokens inherit the language of their
        # originating whitespace token.
        lid_token_ids = self._align_lid_to_subwords(
            normalized, encoding, token_languages
        )

        sample = {
            "text":              text,
            "normalized_text":   normalized,
            "tokens":            text.split(),
            "token_languages":   token_languages,
            "dominant_language": lid_result.dominant_language,
            "is_code_switched":  lid_result.is_code_switched,
            "input_ids":         encoding["input_ids"],
            "attention_mask":    encoding["attention_mask"],
            "lid_token_ids":     lid_token_ids,
        }

        if sentiment_label is not None:
            sample["sentiment_label"] = sentiment_label
        if sarcasm_label is not None:
            sample["sarcasm_label"] = sarcasm_label

        return sample

    def _align_lid_to_subwords(
        self,
        text: str,
        encoding,
        token_languages: List[str],
    ) -> List[int]:
        """
        Map whitespace-level language codes → subword token positions.

        For each subword token, inherit the language of the whitespace
        word it came from. Special tokens ([CLS], [SEP], [PAD]) get
        LANGUAGE_ID_MAP['und'].
        """
        und_id = self.language_id_map.get("und", UNK_LANGUAGE_ID)

        # Use the tokenizer's word_ids() to find word → subword alignment
        try:
            word_ids = encoding.word_ids()
        except AttributeError:
            # Fallback: return uniform 'und' if word_ids unavailable
            return [und_id] * self.max_length

        result = []
        for word_id in word_ids:
            if word_id is None:
                # Special token: [CLS], [SEP], [PAD]
                result.append(und_id)
            elif word_id < len(token_languages):
                lang = token_languages[word_id]
                result.append(
                    self.language_id_map.get(lang, und_id)
                )
            else:
                result.append(und_id)

        return result


# ─────────────────────────────────────────────
# Dataset Builder
# ─────────────────────────────────────────────

class SentimentDatasetBuilder:
    """
    Builds training, validation, and test datasets from raw data sources.

    Supported input formats:
        - CSV  (columns: text, sentiment, sarcasm)
        - JSON / JSONL
        - List[Dict]

    Output: HuggingFace Dataset (or raw list of dicts for flexibility)

    Usage:
        builder = SentimentDatasetBuilder.from_config(config)
        datasets = builder.build_from_csv("data/raw/train.csv")
        datasets['train'].save_to_disk("data/processed/train")
    """

    def __init__(
        self,
        lid: TokenLevelLID,
        normalizer: TextNormalizer,
        tokenizer,
        max_length: int = 128,
        cache_dir: str = "data/cache",
        use_cache: bool = True,
        validation_split: float = 0.15,
        test_split: float = 0.15,
        random_seed: int = 42,
    ):
        self.processor = SampleProcessor(lid, normalizer, tokenizer, max_length)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache
        self.validation_split = validation_split
        self.test_split = test_split
        self.random_seed = random_seed

    @classmethod
    def from_config(
        cls,
        config: dict,
        tokenizer,
        lid: Optional[TokenLevelLID] = None,
        normalizer: Optional[TextNormalizer] = None,
    ) -> "SentimentDatasetBuilder":
        """
        Construct from a loaded preprocessing_config.yaml dict.

        Args:
            config:    Full preprocessing config dict
            tokenizer: HuggingFace tokenizer instance
            lid:       Pre-built LID instance (optional)
            normalizer: Pre-built TextNormalizer (optional)
        """
        lid = lid or get_lid(
            confidence_threshold=config["preprocessing"]
                                 ["language_identification"]
                                 .get("threshold", 0.5)
        )
        normalizer = normalizer or TextNormalizer.from_config(
            config["preprocessing"]
        )

        return cls(
            lid=lid,
            normalizer=normalizer,
            tokenizer=tokenizer,
            max_length=config["preprocessing"]["tokenization"]["max_length"],
            cache_dir=config["preprocessing"]["cache"]["cache_dir"],
            use_cache=config["preprocessing"]["cache"]["cache_processed_data"],
            validation_split=config["validation"]["split_ratio"],
            test_split=config["test"]["split_ratio"],
            random_seed=config["training"]["seed"]
                if "training" in config else 42,
        )

    # ── Input loaders ────────────────────────────────────────────────────────

    def load_csv(
        self,
        filepath: Union[str, Path],
        text_col:      str = "text",
        sentiment_col: str = "sentiment",
        sarcasm_col:   str = "sarcasm",
    ) -> pd.DataFrame:
        """
        Load and validate a CSV dataset.

        Expected columns: text, sentiment (0/1/2 or neg/neu/pos),
                          sarcasm (0/1 or true/false).
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        df = pd.read_csv(filepath, encoding="utf-8")
        logger.info(f"Loaded {len(df)} rows from {filepath.name}")

        return self._validate_and_normalize_df(
            df, text_col, sentiment_col, sarcasm_col
        )

    def load_json(
        self,
        filepath: Union[str, Path],
        text_col:      str = "text",
        sentiment_col: str = "sentiment",
        sarcasm_col:   str = "sarcasm",
    ) -> pd.DataFrame:
        """Load a JSON or JSONL dataset file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        try:
            # Try JSONL first
            df = pd.read_json(filepath, lines=True)
        except ValueError:
            df = pd.read_json(filepath)

        logger.info(f"Loaded {len(df)} rows from {filepath.name}")
        return self._validate_and_normalize_df(
            df, text_col, sentiment_col, sarcasm_col
        )

    def load_from_records(
        self, records: List[Dict]
    ) -> pd.DataFrame:
        """Load from an in-memory list of dicts."""
        df = pd.DataFrame(records)
        return self._validate_and_normalize_df(df, "text", "sentiment", "sarcasm")

    def _validate_and_normalize_df(
        self,
        df: pd.DataFrame,
        text_col: str,
        sentiment_col: str,
        sarcasm_col: str,
    ) -> pd.DataFrame:
        """Validate schema and normalise label columns."""
        # Column presence check
        for col in [text_col, sentiment_col, sarcasm_col]:
            if col not in df.columns:
                raise ValueError(
                    f"Missing required column '{col}'. "
                    f"Found: {list(df.columns)}"
                )

        # Rename to standard names
        df = df.rename(columns={
            text_col:      "text",
            sentiment_col: "sentiment",
            sarcasm_col:   "sarcasm",
        })

        # Drop nulls
        before = len(df)
        df = df.dropna(subset=["text", "sentiment", "sarcasm"])
        dropped = before - len(df)
        if dropped:
            logger.warning(f"Dropped {dropped} rows with null values")

        # Drop duplicates
        df = df.drop_duplicates(subset=["text"])

        # Normalise string labels → integer labels
        df["sentiment_label"] = df["sentiment"].apply(self._map_sentiment)
        df["sarcasm_label"]   = df["sarcasm"].apply(self._map_sarcasm)

        # Filter remaining nulls (unmapped labels)
        before = len(df)
        df = df.dropna(subset=["sentiment_label", "sarcasm_label"])
        dropped = before - len(df)
        if dropped:
            logger.warning(f"Dropped {dropped} rows with unrecognised labels")

        df["sentiment_label"] = df["sentiment_label"].astype(int)
        df["sarcasm_label"]   = df["sarcasm_label"].astype(int)

        logger.info(
            f"Dataset: {len(df)} samples | "
            f"Sentiment dist: {df['sentiment_label'].value_counts().to_dict()} | "
            f"Sarcasm dist: {df['sarcasm_label'].value_counts().to_dict()}"
        )
        return df

    @staticmethod
    def _map_sentiment(value) -> Optional[int]:
        key = str(value).lower().strip()
        return SENTIMENT_LABEL_MAP.get(key)

    @staticmethod
    def _map_sarcasm(value) -> Optional[int]:
        key = str(value).lower().strip()
        return SARCASM_LABEL_MAP.get(key)

    # ── Processing ───────────────────────────────────────────────────────────

    def process_dataframe(
        self,
        df: pd.DataFrame,
        desc: str = "Processing",
    ) -> List[Dict]:
        """
        Apply the full processing pipeline to every row in a DataFrame.

        Args:
            df:   DataFrame with columns: text, sentiment_label, sarcasm_label
            desc: Progress description string

        Returns:
            List of processed sample dicts
        """
        cache_key = self._compute_cache_key(df)
        cached = self._load_cache(cache_key)
        if cached is not None:
            logger.info(f"Loaded {len(cached)} samples from cache ({cache_key})")
            return cached

        samples = []
        failed = 0
        total = len(df)

        try:
            from tqdm import tqdm
            iterator = tqdm(df.iterrows(), total=total, desc=desc)
        except ImportError:
            iterator = df.iterrows()

        for idx, row in iterator:
            try:
                sample = self.processor.process(
                    text=str(row["text"]),
                    sentiment_label=int(row["sentiment_label"]),
                    sarcasm_label=int(row["sarcasm_label"]),
                )
                samples.append(sample)
            except Exception as e:
                failed += 1
                logger.debug(f"Sample {idx} failed: {e}")

        logger.info(
            f"Processed {len(samples)}/{total} samples "
            f"({failed} failed)"
        )

        self._save_cache(cache_key, samples)
        return samples

    # ── Splitting ────────────────────────────────────────────────────────────

    def split_dataset(
        self, samples: List[Dict]
    ) -> Dict[str, List[Dict]]:
        """
        Stratified split into train / validation / test.

        Stratification key: (sentiment_label, sarcasm_label)
        """
        from sklearn.model_selection import train_test_split

        labels = [
            (s["sentiment_label"], s["sarcasm_label"]) for s in samples
        ]

        # First split off test set
        train_val, test, lbl_tv, _ = train_test_split(
            samples, labels,
            test_size=self.test_split,
            stratify=labels,
            random_state=self.random_seed,
        )

        # Then split train vs validation from the remainder
        val_ratio_adjusted = self.validation_split / (1.0 - self.test_split)
        train, val, _, _ = train_test_split(
            train_val, lbl_tv,
            test_size=val_ratio_adjusted,
            stratify=lbl_tv,
            random_state=self.random_seed,
        )

        logger.info(
            f"Split: train={len(train)} | "
            f"val={len(val)} | test={len(test)}"
        )

        return {"train": train, "validation": val, "test": test}

    # ── HuggingFace Dataset export ────────────────────────────────────────────

    def to_hf_dataset(self, samples: List[Dict]):
        """
        Convert list of processed sample dicts to a HuggingFace Dataset.

        Requires `datasets` package.
        """
        try:
            from datasets import Dataset
            return Dataset.from_list(samples)
        except ImportError:
            logger.warning(
                "HuggingFace 'datasets' not installed. "
                "Returning raw list of dicts."
            )
            return samples

    def build_and_split(
        self,
        df: pd.DataFrame,
        as_hf_dataset: bool = True,
    ) -> Dict:
        """
        Full pipeline: process → split → (optionally) convert to HF Dataset.

        Returns:
            Dict with keys: 'train', 'validation', 'test'
        """
        samples = self.process_dataframe(df)
        splits = self.split_dataset(samples)

        if as_hf_dataset:
            return {
                split: self.to_hf_dataset(data)
                for split, data in splits.items()
            }
        return splits

    # ── Caching ──────────────────────────────────────────────────────────────

    def _compute_cache_key(self, df: pd.DataFrame) -> str:
        """Generate a stable cache key from the DataFrame content."""
        h = hashlib.md5(
            pd.util.hash_pandas_object(df, index=True).values.tobytes()
        ).hexdigest()
        return h[:12]

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"dataset_{key}.pkl"

    def _load_cache(self, key: str) -> Optional[List[Dict]]:
        if not self.use_cache:
            return None
        path = self._cache_path(key)
        if path.exists():
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Cache load failed: {e}")
        return None

    def _save_cache(self, key: str, samples: List[Dict]) -> None:
        if not self.use_cache:
            return
        try:
            with open(self._cache_path(key), "wb") as f:
                pickle.dump(samples, f)
            logger.debug(f"Cached {len(samples)} samples → {self._cache_path(key)}")
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")


# ─────────────────────────────────────────────
# Synthetic data generator (for testing when no real data available)
# ─────────────────────────────────────────────

def generate_synthetic_dataset(n_samples: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Generate a small synthetic multilingual dataset for pipeline testing.

    Produces:
        - English, Hinglish, and mixed-language samples
        - All 3 sentiment classes
        - Both sarcasm labels
        - Slang, emojis, elongations, code-switching
    """
    rng = np.random.default_rng(seed)

    templates = [
        # (text, sentiment, sarcasm)
        ("This product is absolutely amazing! 😍 Love it!", "positive", "0"),
        ("yaar this movie was totally bakwas!", "negative", "0"),
        ("The service was okay, nothing special.", "neutral", "0"),
        ("Oh great, another delay. Just what I needed! 🙄", "negative", "1"),
        ("mast product hai yaar, bilkul sahi!", "positive", "0"),
        ("I LOVE waiting for hours! So much fun!! 😒", "negative", "1"),
        ("The quality is decent for the price.", "neutral", "0"),
        ("bahut bekar service, bakwas experience", "negative", "0"),
        ("Yeah this is TOTALLY worth the money! 😤", "negative", "1"),
        ("Product arrived on time, works as expected.", "neutral", "0"),
        ("sooooo amazing, literally the best ever! 🔥", "positive", "0"),
        ("Wow, they actually delivered on the promise for once 😏", "positive", "1"),
        ("bekaar service, time waste kiya", "negative", "0"),
        ("It's okay I guess, not bad not great.", "neutral", "0"),
        ("LIT product, bussin fr fr no cap 🔥", "positive", "0"),
        ("Such great customer support... waited 3 hours 😑", "negative", "1"),
        ("accha experience tha, recommended!", "positive", "0"),
        ("The battery life could be better tbh", "neutral", "0"),
        ("PATHETIC service smh, never ordering again", "negative", "0"),
        ("Pretty good overall, minor issues but fine", "positive", "0"),
    ]

    rows = []
    for _ in range(n_samples):
        template = templates[rng.integers(len(templates))]
        # Add slight variation
        text = template[0]
        if rng.random() < 0.2:
            text = text + " " + rng.choice(["tbh", "ngl", "lol", "smh", "omg"])
        rows.append({
            "text":      text,
            "sentiment": template[1],
            "sarcasm":   template[2],
        })

    df = pd.DataFrame(rows)
    logger.info(f"Generated {len(df)} synthetic samples")
    return df


# ─────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from preprocessing.language_identifier import get_lid
    from preprocessing.text_normalizer import TextNormalizer

    # Test without HuggingFace to keep it lightweight
    print("Testing DatasetBuilder components...")

    lid = get_lid()
    normalizer = TextNormalizer.default()

    # Generate and validate synthetic data
    df = generate_synthetic_dataset(n_samples=30)
    print(f"\nSynthetic dataset shape: {df.shape}")
    print(df[["text", "sentiment", "sarcasm"]].head(5).to_string())

    # Test label mapping
    builder_cls = SentimentDatasetBuilder
    print("\nLabel mapping tests:")
    for val in ["positive", "pos", "2", "negative", "neg", "0", "neutral"]:
        print(f"  '{val}' → {builder_cls._map_sentiment(val)}")

    for val in ["sarcastic", "1", "not_sarcastic", "0", "true", "false"]:
        print(f"  '{val}' → {builder_cls._map_sarcasm(val)}")
    print("\nDatasetBuilder smoke-test complete.")
