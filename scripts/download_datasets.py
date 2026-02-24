"""
Dataset Download & Preparation Script

Downloads public multilingual sentiment + sarcasm datasets from HuggingFace
Hub and converts them to the format expected by SentimentDatasetBuilder.

Expected output schema (matches dataset_builder.py):
    text       : str   — raw input text
    sentiment  : str   — "negative" | "neutral" | "positive"
    sarcasm    : str   — "0" (not sarcastic) | "1" (sarcastic)

Output files written to data/raw/:
    sentiment_combined.csv   — merged sentiment samples (Amazon + Twitter)
    sarcasm_combined.csv     — merged sarcasm samples
    combined_full.csv        — everything merged + ready for training

Usage:
    python scripts/download_datasets.py
    python scripts/download_datasets.py --max-samples 50000 --output-dir data/raw
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' package not installed.")
    print("Run: pip install datasets")
    sys.exit(1)

from utils.logger import get_logger

logger = get_logger("download_datasets")

# ─────────────────────────────────────────────────────────
# Label converters
# ─────────────────────────────────────────────────────────

def stars_to_sentiment(stars: int) -> str:
    """Convert 1-5 star rating → sentiment label."""
    if stars <= 2:
        return "negative"
    elif stars == 3:
        return "neutral"
    else:
        return "positive"


def int_to_sentiment(label: int) -> str:
    """Convert 0/1/2 integer → sentiment label (Twitter dataset)."""
    mapping = {0: "negative", 1: "neutral", 2: "positive"}
    return mapping.get(label, "neutral")


# ─────────────────────────────────────────────────────────
# Dataset 1: Amazon Reviews Multi (multilingual sentiment)
# ─────────────────────────────────────────────────────────

def load_amazon_reviews(max_per_language: int = 10000) -> pd.DataFrame:
    """
    Load Amazon Reviews Multi dataset.
    HF: amazon_reviews_multi
    Languages: en, de, es, fr, zh, ja
    Labels: 1-5 stars → negative / neutral / positive
    """
    languages = ["en", "de", "es", "fr", "zh", "ja"]
    dfs = []

    for lang in languages:
        try:
            logger.info(f"Downloading Amazon Reviews [{lang}]...")
            ds = load_dataset(
                "amazon_reviews_multi",
                lang,
                split="train",
                trust_remote_code=True,
            )
            df = ds.to_pandas()[["review_body", "stars"]].copy()
            df = df.rename(columns={"review_body": "text"})
            df["sentiment"] = df["stars"].apply(stars_to_sentiment)
            df["sarcasm"] = "0"  # Amazon reviews have no sarcasm labels
            df["source"] = f"amazon_{lang}"
            df["language"] = lang

            # Balance classes & cap per language
            df = _balance_and_cap(df, max_per_language)
            dfs.append(df[["text", "sentiment", "sarcasm", "source", "language"]])
            logger.info(f"  ✓ {lang}: {len(df)} samples")

        except Exception as e:
            logger.warning(f"  ✗ Failed to load amazon [{lang}]: {e}")

    if not dfs:
        logger.error("No Amazon Reviews data loaded.")
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Amazon Reviews total: {len(combined)} samples")
    return combined


# ─────────────────────────────────────────────────────────
# Dataset 2: Twitter Sentiment Multilingual (CardiffNLP)
# ─────────────────────────────────────────────────────────

def load_twitter_sentiment(max_per_language: int = 5000) -> pd.DataFrame:
    """
    Load CardiffNLP multilingual Twitter sentiment dataset.
    HF: cardiffnlp/tweet_sentiment_multilingual
    Languages: en, ar, de, fr, hi, it, pt, es
    Labels: 0=negative, 1=neutral, 2=positive
    """
    languages = ["en", "ar", "de", "fr", "hi", "pt", "es"]
    dfs = []

    for lang in languages:
        try:
            logger.info(f"Downloading Twitter Sentiment [{lang}]...")
            ds = load_dataset(
                "cardiffnlp/tweet_sentiment_multilingual",
                lang,
                split="train",
                trust_remote_code=True,
            )
            df = ds.to_pandas()[["text", "label"]].copy()
            df["sentiment"] = df["label"].apply(int_to_sentiment)
            df["sarcasm"] = "0"
            df["source"] = f"twitter_{lang}"
            df["language"] = lang

            df = _balance_and_cap(df, max_per_language)
            dfs.append(df[["text", "sentiment", "sarcasm", "source", "language"]])
            logger.info(f"  ✓ {lang}: {len(df)} samples")

        except Exception as e:
            logger.warning(f"  ✗ Failed to load twitter [{lang}]: {e}")

    if not dfs:
        logger.error("No Twitter Sentiment data loaded.")
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Twitter Sentiment total: {len(combined)} samples")
    return combined


# ─────────────────────────────────────────────────────────
# Dataset 3: News Headlines Sarcasm
# ─────────────────────────────────────────────────────────

def load_sarcasm_news(max_samples: int = 20000) -> pd.DataFrame:
    """
    Load News Headlines Sarcasm dataset.
    HF: raquiba/sarcasm_news_headline
    Labels: 0=not sarcastic, 1=sarcastic
    All English, sentiment inferred from sarcasm label.
    """
    try:
        logger.info("Downloading News Headlines Sarcasm...")
        ds = load_dataset(
            "raquiba/sarcasm_news_headline",
            split="train",
            trust_remote_code=True,
        )
        df = ds.to_pandas().copy()

        # Map columns (headline + is_sarcastic)
        if "headline" in df.columns:
            df = df.rename(columns={"headline": "text"})
        if "is_sarcastic" in df.columns:
            df = df.rename(columns={"is_sarcastic": "sarcasm_int"})

        df["sarcasm"] = df["sarcasm_int"].astype(str)

        # Sarcastic headlines → negative sentiment (ironic statements)
        # Non-sarcastic → use neutral as default (news headlines are factual)
        df["sentiment"] = df["sarcasm_int"].apply(
            lambda x: "negative" if x == 1 else "neutral"
        )

        df["source"] = "sarcasm_news"
        df["language"] = "en"

        df = df[["text", "sentiment", "sarcasm", "source", "language"]]
        df = df.dropna(subset=["text"])
        df = df.head(max_samples)

        logger.info(f"  ✓ Sarcasm News: {len(df)} samples")
        return df

    except Exception as e:
        logger.error(f"  ✗ Failed to load sarcasm news: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────
# Dataset 4: iSarcasmEval
# ─────────────────────────────────────────────────────────

def load_isarcasm(max_samples: int = 10000) -> pd.DataFrame:
    """
    Load iSarcasmEval dataset.
    HF: iabufarha/iSarcasmEval
    High-quality English sarcasm annotations.
    """
    try:
        logger.info("Downloading iSarcasmEval...")
        ds = load_dataset(
            "iabufarha/iSarcasmEval",
            split="train",
            trust_remote_code=True,
        )
        df = ds.to_pandas().copy()

        # Flexible column mapping
        text_col = next((c for c in ["tweet", "text", "sentence"] if c in df.columns), None)
        label_col = next((c for c in ["sarcastic", "label", "is_sarcastic"] if c in df.columns), None)

        if text_col is None or label_col is None:
            logger.warning(f"iSarcasmEval: unexpected columns {list(df.columns)}, skipping")
            return pd.DataFrame()

        df = df.rename(columns={text_col: "text", label_col: "sarcasm_int"})
        df["sarcasm"] = df["sarcasm_int"].astype(int).astype(str)
        df["sentiment"] = df["sarcasm_int"].apply(
            lambda x: "negative" if int(x) == 1 else "neutral"
        )
        df["source"] = "isarcasm"
        df["language"] = "en"

        df = df[["text", "sentiment", "sarcasm", "source", "language"]]
        df = df.dropna(subset=["text"])
        df = df.head(max_samples)

        logger.info(f"  ✓ iSarcasmEval: {len(df)} samples")
        return df

    except Exception as e:
        logger.warning(f"  ✗ Failed to load iSarcasmEval: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────
# Dataset 5: SentiMix (Hindi-English code-switched)
# ─────────────────────────────────────────────────────────

def load_sentimix(max_samples: int = 10000) -> pd.DataFrame:
    """
    Load SentiMix Hindi-English code-switched dataset.
    HF: sid321axn/Hindi_English_Truncated_Corpus (or submodule)
    Labels: positive / negative / neutral
    """
    try:
        logger.info("Downloading SentiMix (EN-HI code-switched)...")
        ds = load_dataset(
            "senti_lex",
            split="train",
            trust_remote_code=True,
        )
        df = ds.to_pandas().copy()

        text_col = next((c for c in ["tweet", "text", "sentence"] if c in df.columns), None)
        label_col = next((c for c in ["sentiment", "label"] if c in df.columns), None)

        if text_col is None:
            logger.warning("SentiMix: text column not found, skipping")
            return pd.DataFrame()

        df = df.rename(columns={text_col: "text"})

        if label_col:
            df["sentiment"] = df[label_col].apply(
                lambda x: str(x).lower() if str(x).lower() in
                          ["positive", "negative", "neutral"] else "neutral"
            )
        else:
            df["sentiment"] = "neutral"

        df["sarcasm"] = "0"
        df["source"] = "sentimix"
        df["language"] = "hi-en"

        df = df[["text", "sentiment", "sarcasm", "source", "language"]]
        df = df.dropna(subset=["text"])
        df = df.head(max_samples)

        logger.info(f"  ✓ SentiMix: {len(df)} samples")
        return df

    except Exception as e:
        logger.warning(f"  ✗ SentiMix not available ({e}), skipping")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────
# Helper: balance & cap
# ─────────────────────────────────────────────────────────

def _balance_and_cap(df: pd.DataFrame, max_total: int) -> pd.DataFrame:
    """
    Undersample majority classes to balance, then cap total rows.
    Stratifies by sentiment label.
    """
    per_class = max_total // 3
    balanced = []
    for label in ["negative", "neutral", "positive"]:
        subset = df[df["sentiment"] == label]
        if len(subset) > per_class:
            subset = subset.sample(per_class, random_state=42)
        balanced.append(subset)
    return pd.concat(balanced, ignore_index=True).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)


# ─────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────

def download_and_prepare(
    output_dir: str = "data/raw",
    max_samples: int = 100_000,
) -> None:
    """
    Download all datasets, merge, and save to output_dir.

    Args:
        output_dir:  Directory to save CSV files
        max_samples: Approximate total sample cap
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    per_source = max_samples // 5  # distribute budget across sources

    # ── Load all sources ──────────────────────────────────
    dfs = []

    amazon = load_amazon_reviews(max_per_language=per_source // 6)
    if not amazon.empty:
        dfs.append(amazon)
        amazon.to_csv(out / "amazon_reviews.csv", index=False)
        logger.info(f"Saved → {out / 'amazon_reviews.csv'}")

    twitter = load_twitter_sentiment(max_per_language=per_source // 7)
    if not twitter.empty:
        dfs.append(twitter)
        twitter.to_csv(out / "twitter_sentiment.csv", index=False)
        logger.info(f"Saved → {out / 'twitter_sentiment.csv'}")

    sarcasm_news = load_sarcasm_news(max_samples=per_source)
    if not sarcasm_news.empty:
        dfs.append(sarcasm_news)
        sarcasm_news.to_csv(out / "sarcasm_news.csv", index=False)
        logger.info(f"Saved → {out / 'sarcasm_news.csv'}")

    isarcasm = load_isarcasm(max_samples=per_source)
    if not isarcasm.empty:
        dfs.append(isarcasm)
        isarcasm.to_csv(out / "isarcasm.csv", index=False)
        logger.info(f"Saved → {out / 'isarcasm.csv'}")

    sentimix = load_sentimix(max_samples=per_source)
    if not sentimix.empty:
        dfs.append(sentimix)
        sentimix.to_csv(out / "sentimix.csv", index=False)
        logger.info(f"Saved → {out / 'sentimix.csv'}")

    # ── Merge & deduplicate ───────────────────────────────
    if not dfs:
        logger.error("No datasets were loaded. Check your internet connection.")
        sys.exit(1)

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["text"])
    combined = combined.dropna(subset=["text", "sentiment", "sarcasm"])

    # Shuffle
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # Cap total
    if len(combined) > max_samples:
        combined = combined.head(max_samples)

    # ── Save combined ─────────────────────────────────────
    combined_path = out / "combined_full.csv"
    combined.to_csv(combined_path, index=False)

    # ── Print summary ─────────────────────────────────────
    print("\n" + "="*60)
    print("DATASET DOWNLOAD COMPLETE")
    print("="*60)
    print(f"Total samples : {len(combined):,}")
    print(f"Output dir    : {out.resolve()}\n")

    print("Sentiment distribution:")
    for label, count in combined["sentiment"].value_counts().items():
        pct = count / len(combined) * 100
        print(f"  {label:10s}: {count:6,}  ({pct:.1f}%)")

    print("\nSarcasm distribution:")
    for label, count in combined["sarcasm"].value_counts().items():
        name = "sarcastic" if str(label) == "1" else "not_sarcastic"
        pct = count / len(combined) * 100
        print(f"  {name:14s}: {count:6,}  ({pct:.1f}%)")

    print("\nSources:")
    for source, count in combined["source"].value_counts().items():
        print(f"  {source:25s}: {count:6,}")

    print("\nLanguages:")
    for lang, count in combined["language"].value_counts().items():
        print(f"  {lang:8s}: {count:6,}")

    print(f"\n✓ Saved combined dataset → {combined_path}")
    print("\nNext step:")
    print("  python src/training/trainer.py  (uses data/raw/combined_full.csv)")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare datasets for multilingual sentiment training"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="data/raw",
        help="Directory to save CSV files (default: data/raw)"
    )
    parser.add_argument(
        "--max-samples", "-n",
        type=int,
        default=100_000,
        help="Approximate total sample cap (default: 100,000)"
    )
    args = parser.parse_args()

    download_and_prepare(
        output_dir=args.output_dir,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
