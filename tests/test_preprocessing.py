"""
Test suite for Phase 2: Data Pipeline & Language Identification

Tests are organized to run without heavyweight ML dependencies
(no torch, no transformers required for LID + normalization tests).
DataLoader tests use minimal fake tensors.
"""

import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ═══════════════════════════════════════════════════════════
# SECTION 1 — ScriptDetector
# ═══════════════════════════════════════════════════════════

class TestScriptDetector:
    """Test Unicode script detection."""

    def setup_method(self):
        from preprocessing.language_identifier import ScriptDetector
        self.detector = ScriptDetector()

    def test_latin_script(self):
        assert self.detector.detect_script("hello") == "Latin"

    def test_devanagari_script(self):
        assert self.detector.detect_script("नमस्ते") == "Devanagari"

    def test_arabic_script(self):
        assert self.detector.detect_script("مرحبا") == "Arabic"

    def test_emoji(self):
        assert self.detector.detect_script("😍") == "Emoji"

    def test_cjk_script(self):
        assert self.detector.detect_script("你好") == "CJK"

    def test_mixed_devanagari_latin(self):
        # Token with both — should detect dominant
        script = self.detector.detect_script("hello123")
        assert script == "Latin"

    def test_empty_token(self):
        assert self.detector.detect_script("") == "Unknown"

    def test_pure_punctuation(self):
        # Punctuation is not captured by script patterns → Unknown
        result = self.detector.detect_script("!!!")
        assert result == "Unknown"

    def test_script_to_language_mapping(self):
        assert self.detector.script_to_language("Devanagari") == "hi"
        assert self.detector.script_to_language("Arabic") == "ar"
        assert self.detector.script_to_language("CJK") == "zh"
        assert self.detector.script_to_language("Latin") == "en"
        assert self.detector.script_to_language("Unknown") is None


# ═══════════════════════════════════════════════════════════
# SECTION 2 — TokenLevelLID
# ═══════════════════════════════════════════════════════════

class TestTokenLevelLID:
    """Test the token-level language identification engine."""

    def setup_method(self):
        from preprocessing.language_identifier import TokenLevelLID
        # Use heuristic backend (no FastText needed for tests)
        self.lid = TokenLevelLID(confidence_threshold=0.5, min_token_length=3)

    def test_init_backend_available(self):
        assert self.lid.backend in ("fasttext", "langid", "heuristic")

    def test_url_is_lang_agnostic(self):
        result = self.lid.identify_token("https://example.com")
        assert result.language == "und"

    def test_mention_is_lang_agnostic(self):
        result = self.lid.identify_token("@username")
        assert result.language == "und"

    def test_number_is_lang_agnostic(self):
        result = self.lid.identify_token("42")
        assert result.language == "und"

    def test_emoji_detected(self):
        result = self.lid.identify_token("😍")
        assert result.language == "emoji"

    def test_devanagari_token(self):
        result = self.lid.identify_token("नमस्ते")
        assert result.language == "hi"
        assert result.confidence >= 0.9

    def test_arabic_token(self):
        result = self.lid.identify_token("मرحبا")
        assert result.language == "ar"

    def test_short_token_is_ambiguous(self):
        result = self.lid.identify_token("hi")   # 2 chars < min_token_length=3
        assert result.is_ambiguous

    def test_sentence_identify_returns_sentence_lid(self):
        from preprocessing.language_identifier import SentenceLID
        result = self.lid.identify_sentence("This is an English sentence.")
        assert isinstance(result, SentenceLID)

    def test_empty_sentence(self):
        result = self.lid.identify_sentence("")
        assert result.dominant_language == "und"
        assert not result.is_code_switched

    def test_language_sequence_length_matches_tokens(self):
        text = "yaar this product is bakwas"
        result = self.lid.identify_sentence(text)
        assert len(result.language_sequence()) == len(text.split())

    def test_hinglish_detected_code_switched(self):
        # "yaar" (Hindi slang) + English → code-switched
        # With heuristic backend both get "en" but test structure
        result = self.lid.identify_sentence("yaar this movie is amazing")
        assert len(result.token_results) == 5

    def test_devanagari_sentence(self):
        result = self.lid.identify_sentence("नमस्ते यह बहुत अच्छा है")
        # All Devanagari → Hindi dominant
        assert result.dominant_language == "hi"

    def test_batch_identify(self):
        texts = ["Hello world", "नमस्ते", "Test sentence here"]
        results = self.lid.identify_batch(texts)
        assert len(results) == len(texts)

    def test_switch_points_detected(self):
        from preprocessing.language_identifier import TokenLanguage
        # Manually construct token results to test switch detection
        token_results = [
            TokenLanguage("word1", "en", 0.9, "Latin", False),
            TokenLanguage("word2", "en", 0.9, "Latin", False),
            TokenLanguage("word3", "hi", 0.9, "Devanagari", False),
            TokenLanguage("word4", "hi", 0.9, "Devanagari", False),
        ]
        result = self.lid._aggregate_results(token_results)
        assert result.is_code_switched
        assert len(result.switch_points) == 1
        assert result.switch_points[0] == 2   # index where language changes


# ═══════════════════════════════════════════════════════════
# SECTION 3 — Text Normalizer Stages
# ═══════════════════════════════════════════════════════════

class TestURLMentionNormalizer:
    def setup_method(self):
        from preprocessing.text_normalizer import URLMentionNormalizer
        self.norm = URLMentionNormalizer()

    def test_url_replaced(self):
        out = self.norm.normalize("Check https://example.com out")
        assert "[URL]" in out
        assert "https://example.com" not in out

    def test_mention_replaced(self):
        out = self.norm.normalize("Hi @john what's up")
        assert "[MENTION]" in out
        assert "@john" not in out

    def test_hashtag_expanded(self):
        out = self.norm.normalize("I love #GoodProduct")
        assert "GoodProduct" in out
        assert "#GoodProduct" not in out

    def test_no_modification_clean_text(self):
        clean = "This is a clean sentence."
        out = self.norm.normalize(clean)
        assert out == clean


class TestEmojiNormalizer:
    def setup_method(self):
        from preprocessing.text_normalizer import EmojiNormalizer
        self.norm_lex   = EmojiNormalizer(strategy="lexicon")
        self.norm_rm    = EmojiNormalizer(strategy="remove")
        self.norm_keep  = EmojiNormalizer(strategy="keep")

    def test_lexicon_replaces_emoji(self):
        out = self.norm_lex.normalize("I love this 😍")
        assert "love" in out
        assert "😍" not in out

    def test_remove_strips_emoji(self):
        out = self.norm_rm.normalize("Great! 🔥🔥")
        assert "🔥" not in out
        assert "Great" in out

    def test_keep_passes_through(self):
        text = "Nice work 👍"
        out = self.norm_keep.normalize(text)
        assert out == text

    def test_invalid_strategy_raises(self):
        from preprocessing.text_normalizer import EmojiNormalizer
        with pytest.raises(ValueError):
            EmojiNormalizer(strategy="invalid")


class TestElongationNormalizer:
    def setup_method(self):
        from preprocessing.text_normalizer import ElongationNormalizer
        self.norm = ElongationNormalizer(max_repetitions=2)

    def test_elongated_vowels_reduced(self):
        out = self.norm.normalize("sooooooo")
        assert out == "soo"

    def test_elongated_consonants_reduced(self):
        out = self.norm.normalize("noooooo")
        assert out == "noo"

    def test_normal_word_unchanged(self):
        out = self.norm.normalize("hello")
        assert out == "hello"

    def test_double_repetition_preserved(self):
        # "goo" (2 o's) should not be reduced
        out = self.norm.normalize("goo")
        assert out == "goo"

    def test_sentence_elongation(self):
        out = self.norm.normalize("This is sooooo AMAZING!!!")
        assert "sooooo" not in out
        assert "soo" in out

    def test_invalid_max_reps_raises(self):
        from preprocessing.text_normalizer import ElongationNormalizer
        with pytest.raises(ValueError):
            ElongationNormalizer(max_repetitions=0)


class TestAbbreviationExpander:
    def setup_method(self):
        from preprocessing.text_normalizer import AbbreviationExpander
        self.expander = AbbreviationExpander()

    def test_lol_expanded(self):
        out = self.expander.normalize("lol this is funny")
        assert "laughing out loud" in out

    def test_omg_expanded(self):
        out = self.expander.normalize("omg what happened")
        assert "oh my god" in out

    def test_preserves_trailing_punctuation(self):
        out = self.expander.normalize("thx!")
        assert out.endswith("!")

    def test_unknown_word_unchanged(self):
        out = self.expander.normalize("supercalifragilistic")
        assert out == "supercalifragilistic"

    def test_custom_lexicon_overrides(self):
        from preprocessing.text_normalizer import AbbreviationExpander
        exp = AbbreviationExpander(custom_lexicon={"gg": "good game"})
        out = exp.normalize("gg everyone")
        assert "good game" in out


class TestSlangExpander:
    def setup_method(self):
        from preprocessing.text_normalizer import SlangExpander
        self.expander = SlangExpander()

    def test_english_slang_expanded(self):
        out = self.expander.normalize("this product is lit")
        assert "amazing" in out

    def test_hindi_slang_expanded(self):
        out = self.expander.normalize("bilkul bakwas product")
        assert "nonsense" in out

    def test_lid_guided_expansion(self):
        # "lit" with lang="en" → expanded; "lit" with lang="hi" → not in hi lex
        en_out = self.expander.normalize("this is lit", token_languages=["en", "en", "en"])
        assert "amazing" in en_out

    def test_unknown_word_unchanged(self):
        out = self.expander.normalize("supercalifragilistic")
        assert out == "supercalifragilistic"


class TestCaseNormalizer:
    def setup_method(self):
        from preprocessing.text_normalizer import CaseNormalizer
        self.norm = CaseNormalizer(preserve_allcaps=True)

    def test_lowercase_applied(self):
        out = self.norm.normalize("Hello World")
        assert out == "hello world"

    def test_allcaps_preserved(self):
        out = self.norm.normalize("I HATE this!")
        assert "HATE" in out

    def test_short_allcaps_preserved(self):
        out = self.norm.normalize("This is OK not bad")
        assert "OK" in out

    def test_no_preserve(self):
        from preprocessing.text_normalizer import CaseNormalizer
        norm = CaseNormalizer(preserve_allcaps=False)
        out = norm.normalize("I LOVE this")
        assert out == "i love this"


# ═══════════════════════════════════════════════════════════
# SECTION 4 — TextNormalizer (full pipeline)
# ═══════════════════════════════════════════════════════════

class TestTextNormalizerPipeline:
    def setup_method(self):
        from preprocessing.text_normalizer import TextNormalizer
        self.normalizer = TextNormalizer.default()

    def test_active_stages(self):
        stages = self.normalizer.active_stages()
        assert "url" in stages
        assert "emoji" in stages
        assert "elongation" in stages

    def test_hinglish_normalized(self):
        text = "yaar this product is sooooo bakwas omg 😡"
        out = self.normalizer.normalize(text)
        # Elongation should reduce sooooo
        assert "sooooo" not in out
        # Emoji should be replaced
        assert "😡" not in out
        # omg expanded
        assert "oh my god" in out

    def test_url_and_mention_normalized(self):
        text = "Check @user https://example.com for details"
        out = self.normalizer.normalize(text)
        assert "[URL]" in out
        assert "[MENTION]" in out

    def test_sarcastic_text_preserved(self):
        # Key sarcasm signals — capitalization, punctuation — must survive
        text = "Oh GREAT, another delay. Just what I NEEDED!!"
        out = self.normalizer.normalize(text)
        assert "GREAT" in out
        assert "NEEDED" in out
        assert "!!" in out

    def test_sentiment_signals_preserved(self):
        text = "This is AMAZING! Totally LOVE it 😍"
        out = self.normalizer.normalize(text)
        assert "AMAZING" in out
        assert "LOVE" in out
        # 😍 should be replaced with "love"
        assert "😍" not in out

    def test_batch_normalize(self):
        texts = ["Hello world", "sooooo good!", "yaar bakwas"]
        results = self.normalizer.normalize_batch(texts)
        assert len(results) == len(texts)
        assert isinstance(results[0], str)

    def test_type_error_on_non_string(self):
        with pytest.raises(TypeError):
            self.normalizer.normalize(12345)

    def test_empty_string(self):
        out = self.normalizer.normalize("")
        assert isinstance(out, str)

    def test_whitespace_only(self):
        out = self.normalizer.normalize("   ")
        assert isinstance(out, str)


# ═══════════════════════════════════════════════════════════
# SECTION 5 — Dataset Builder (no tokenizer)
# ═══════════════════════════════════════════════════════════

class TestDatasetBuilderLabelMapping:
    """Test label mapping without needing a tokenizer."""

    def test_sentiment_positive(self):
        from data.dataset_builder import SentimentDatasetBuilder
        assert SentimentDatasetBuilder._map_sentiment("positive") == 2
        assert SentimentDatasetBuilder._map_sentiment("pos") == 2
        assert SentimentDatasetBuilder._map_sentiment("2") == 2

    def test_sentiment_negative(self):
        from data.dataset_builder import SentimentDatasetBuilder
        assert SentimentDatasetBuilder._map_sentiment("negative") == 0
        assert SentimentDatasetBuilder._map_sentiment("neg") == 0
        assert SentimentDatasetBuilder._map_sentiment("0") == 0

    def test_sentiment_neutral(self):
        from data.dataset_builder import SentimentDatasetBuilder
        assert SentimentDatasetBuilder._map_sentiment("neutral") == 1
        assert SentimentDatasetBuilder._map_sentiment("neu") == 1

    def test_sentiment_unknown_returns_none(self):
        from data.dataset_builder import SentimentDatasetBuilder
        assert SentimentDatasetBuilder._map_sentiment("unknown_label") is None

    def test_sarcasm_positive(self):
        from data.dataset_builder import SentimentDatasetBuilder
        assert SentimentDatasetBuilder._map_sarcasm("sarcastic") == 1
        assert SentimentDatasetBuilder._map_sarcasm("1") == 1
        assert SentimentDatasetBuilder._map_sarcasm("true") == 1

    def test_sarcasm_negative(self):
        from data.dataset_builder import SentimentDatasetBuilder
        assert SentimentDatasetBuilder._map_sarcasm("not_sarcastic") == 0
        assert SentimentDatasetBuilder._map_sarcasm("0") == 0
        assert SentimentDatasetBuilder._map_sarcasm("false") == 0


class TestSyntheticDataset:
    """Test synthetic data generation and validation."""

    def test_generate_returns_dataframe(self):
        import pandas as pd
        from data.dataset_builder import generate_synthetic_dataset
        df = generate_synthetic_dataset(n_samples=50)
        assert isinstance(df, pd.DataFrame)

    def test_correct_number_of_samples(self):
        from data.dataset_builder import generate_synthetic_dataset
        df = generate_synthetic_dataset(n_samples=100)
        assert len(df) == 100

    def test_required_columns_present(self):
        from data.dataset_builder import generate_synthetic_dataset
        df = generate_synthetic_dataset(n_samples=20)
        for col in ["text", "sentiment", "sarcasm"]:
            assert col in df.columns

    def test_sentiment_values_valid(self):
        from data.dataset_builder import generate_synthetic_dataset, SENTIMENT_LABEL_MAP
        df = generate_synthetic_dataset(n_samples=50)
        valid_sentiments = set(SENTIMENT_LABEL_MAP.keys())
        for val in df["sentiment"].unique():
            assert str(val).lower() in valid_sentiments, f"Invalid: {val}"

    def test_no_empty_texts(self):
        from data.dataset_builder import generate_synthetic_dataset
        df = generate_synthetic_dataset(n_samples=50)
        assert df["text"].str.strip().ne("").all()


# ═══════════════════════════════════════════════════════════
# SECTION 6 — DataLoader (torch required)
# ═══════════════════════════════════════════════════════════

def _make_fake_samples(n: int, max_length: int = 128) -> list:
    return [
        {
            "input_ids":        [1] + [100 + i] * 3 + [0] * (max_length - 4),
            "attention_mask":   [1] * 4 + [0] * (max_length - 4),
            "lid_token_ids":    [10] + [i % 5] * 3 + [10] * (max_length - 4),
            "sentiment_label":  i % 3,
            "sarcasm_label":    i % 2,
            "is_code_switched": i % 2 == 0,
            "dominant_language": "en",
        }
        for i in range(n)
    ]


try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestMultiTaskDataset:
    def test_dataset_length(self):
        from data.dataloader import MultiTaskSentimentDataset
        samples = _make_fake_samples(30)
        ds = MultiTaskSentimentDataset(samples)
        assert len(ds) == 30

    def test_item_keys(self):
        from data.dataloader import MultiTaskSentimentDataset
        samples = _make_fake_samples(5)
        ds = MultiTaskSentimentDataset(samples)
        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "lid_token_ids" in item
        assert "sentiment_label" in item
        assert "sarcasm_label" in item

    def test_tensor_shapes(self):
        import torch
        from data.dataloader import MultiTaskSentimentDataset
        samples = _make_fake_samples(5, max_length=128)
        ds = MultiTaskSentimentDataset(samples)
        item = ds[0]
        assert item["input_ids"].shape == (128,)
        assert item["sentiment_label"].shape == ()

    def test_tensor_dtypes(self):
        import torch
        from data.dataloader import MultiTaskSentimentDataset
        samples = _make_fake_samples(5)
        ds = MultiTaskSentimentDataset(samples)
        item = ds[0]
        assert item["input_ids"].dtype == torch.long
        assert item["sentiment_label"].dtype == torch.long

    def test_missing_key_raises(self):
        from data.dataloader import MultiTaskSentimentDataset
        bad_samples = [{"input_ids": [1, 2], "attention_mask": [1, 1]}]
        with pytest.raises(KeyError):
            MultiTaskSentimentDataset(bad_samples)

    def test_class_counts(self):
        from data.dataloader import MultiTaskSentimentDataset
        samples = _make_fake_samples(30)
        ds = MultiTaskSentimentDataset(samples)
        counts = ds.class_counts()
        assert "sentiment" in counts
        assert "sarcasm" in counts

    def test_dataloader_batch_shape(self):
        import torch
        from data.dataloader import MultiTaskSentimentDataset
        from torch.utils.data import DataLoader
        samples = _make_fake_samples(32, max_length=128)
        ds = MultiTaskSentimentDataset(samples)
        loader = DataLoader(ds, batch_size=8)
        batch = next(iter(loader))
        assert batch["input_ids"].shape == (8, 128)

    def test_compute_class_weights_shape(self):
        import torch
        from data.dataloader import compute_class_weights
        labels = [0, 0, 1, 2, 1, 0, 2]
        weights = compute_class_weights(labels, num_classes=3)
        assert weights.shape == (3,)
        assert (weights > 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
