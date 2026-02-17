"""
Text Normalization Pipeline

Multi-stage normalization for noisy, multilingual, social-media text.
Each stage is independently configurable and preserves sentiment-relevant signals.

Pipeline order (order matters — do NOT reorder without analysis):
    1. URL / mention / hashtag standardization  (before any text manipulation)
    2. Emoji conversion → sentiment-preserving text
    3. Elongation reduction                     ("sooooo" → "so")
    4. Informal abbreviation expansion          ("lol" → "laughing out loud")
    5. Slang expansion (language-aware)         ("bakwas" → kept, "grt" → "great")
    6. Case normalization                       (lowercase, preserve acronyms)
    7. Whitespace cleanup                       (final pass)

Design decisions:
    - Sentiment-critical tokens (punctuation, emojis, caps) are preserved or
      mapped rather than deleted.
    - Slang lexicons are language-specific to avoid cross-lingual collisions.
    - All stages are individually toggle-able via config.
"""

import re
import json
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import get_logger

logger = get_logger("text_normalizer")


# ─────────────────────────────────────────────
# Built-in lexicons (no external file needed for bootstrap)
# ─────────────────────────────────────────────

# Common internet abbreviations (English)
DEFAULT_ABBREVIATIONS: Dict[str, str] = {
    "lol":   "laughing out loud",
    "lmao":  "laughing my ass off",
    "rofl":  "rolling on the floor laughing",
    "omg":   "oh my god",
    "omfg":  "oh my god",
    "wtf":   "what the hell",
    "idk":   "i don't know",
    "imo":   "in my opinion",
    "imho":  "in my humble opinion",
    "tbh":   "to be honest",
    "ngl":   "not gonna lie",
    "smh":   "shaking my head",
    "irl":   "in real life",
    "afaik": "as far as i know",
    "fwiw":  "for what it's worth",
    "iirc":  "if i remember correctly",
    "tbt":   "throwback thursday",
    "tgif":  "thank god it's friday",
    "fyi":   "for your information",
    "asap":  "as soon as possible",
    "btw":   "by the way",
    "brb":   "be right back",
    "nvm":   "never mind",
    "thx":   "thanks",
    "ty":    "thank you",
    "np":    "no problem",
    "yw":    "you're welcome",
    "dm":    "direct message",
    "rt":    "retweet",
    "gr8":   "great",
    "grt":   "great",
    "b4":    "before",
    "u":     "you",
    "ur":    "your",
    "r":     "are",
    "2":     "to",
    "4":     "for",
    "ppl":   "people",
    "w/":    "with",
    "w/o":   "without",
    "bc":    "because",
    "coz":   "because",
    "cuz":   "because",
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    "kinda": "kind of",
    "sorta": "sort of",
    "lemme": "let me",
}

# English slang → sentiment-neutral expansions
DEFAULT_SLANG_EN: Dict[str, str] = {
    "lit":       "amazing",
    "goat":      "greatest of all time",
    "banger":    "great song",
    "bussin":    "very good",
    "fire":      "excellent",
    "slay":      "perform excellently",
    "lowkey":    "somewhat",
    "highkey":   "very much",
    "no cap":    "honestly",
    "cap":       "lie",
    "salty":     "upset",
    "triggered": "upset",
    "shade":     "disrespect",
    "vibe":      "atmosphere",
    "fam":       "family friend",
    "bae":       "beloved",
    "bougie":    "extravagant",
    "flex":      "show off",
    "clout":     "influence",
    "ghost":     "ignore",
    "sus":       "suspicious",
    "sus af":    "very suspicious",
    "mid":       "mediocre",
    "based":     "admirable",
    "cringe":    "embarrassing",
    "vibe check":"evaluation",
}

# Hindi/Hinglish slang → sentiment-neutral expansions
DEFAULT_SLANG_HI: Dict[str, str] = {
    "bakwas":  "nonsense",
    "yaar":    "friend",
    "dost":    "friend",
    "mast":    "great",
    "zabardast": "excellent",
    "bekaar":  "useless",
    "bekar":   "useless",
    "faltu":   "worthless",
    "sahi":    "correct",
    "bilkul":  "absolutely",
    "zyada":   "too much",
    "thoda":   "a little",
    "jaldi":   "quickly",
    "kal":     "yesterday or tomorrow",
    "kuch":    "something",
    "accha":   "good",
    "acha":    "good",
    "bahut":   "very",
    "pagal":   "crazy",
    "dhamaka": "explosion great",
    "bindaas": "carefree",
}

# Common emoji → sentiment description mapping
EMOJI_SENTIMENT_MAP: Dict[str, str] = {
    "😀": "happy",         "😁": "happy",        "😂": "laughing",
    "🤣": "very funny",    "😍": "love",          "😎": "cool",
    "🥰": "love",          "😊": "happy",         "😇": "innocent",
    "🤩": "excited",       "😏": "smug",          "😒": "unamused",
    "😔": "sad",           "😢": "crying",        "😭": "very sad",
    "😡": "angry",         "🤬": "very angry",    "😤": "frustrated",
    "🙄": "eye roll",      "😑": "expressionless","😐": "neutral",
    "🤔": "thinking",      "🤨": "suspicious",    "😬": "awkward",
    "🥴": "dizzy",         "😴": "sleepy",        "🤯": "mind blown",
    "🥳": "celebrating",   "😩": "exhausted",     "😫": "tired",
    "😤": "angry",         "🤮": "disgusted",     "🤢": "sick",
    "👍": "thumbs up",     "👎": "thumbs down",   "❤️": "love",
    "💔": "heartbreak",    "✨": "sparkle",        "🔥": "fire excellent",
    "💯": "perfect",       "🙏": "thank you",      "💀": "dead laughing",
    "😆": "very funny",    "🥹": "moved",          "🤗": "hug",
    "😵": "confused",      "🤡": "clown",          "👏": "applause",
    "💩": "terrible",      "🤑": "money",          "🎉": "celebration",
}


# ─────────────────────────────────────────────
# Individual Normalizer Stages
# ─────────────────────────────────────────────

class URLMentionNormalizer:
    """Replace URLs, @mentions, and extract #hashtag text."""

    URL_PATTERN = re.compile(
        r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2})|[/?=&#+@~!$'()*,;:])+",
        re.IGNORECASE
    )
    MENTION_PATTERN = re.compile(r"@\w+")
    HASHTAG_PATTERN = re.compile(r"#(\w+)")

    def __init__(
        self,
        url_token:     str = "[URL]",
        mention_token: str = "[MENTION]",
        expand_hashtags: bool = True,
    ):
        self.url_token = url_token
        self.mention_token = mention_token
        self.expand_hashtags = expand_hashtags

    def normalize(self, text: str) -> str:
        text = self.URL_PATTERN.sub(self.url_token, text)
        text = self.MENTION_PATTERN.sub(self.mention_token, text)
        if self.expand_hashtags:
            # "#GoodProduct" → "GoodProduct"
            text = self.HASHTAG_PATTERN.sub(r"\1", text)
        return text


class EmojiNormalizer:
    """
    Map emojis to sentiment-descriptive text or demojize/remove them.

    Strategies:
        'lexicon'  → replace with curated sentiment word  (recommended)
        'demojize' → replace with :emoji_name: via emoji package
        'remove'   → strip all emojis
        'keep'     → pass-through (transformer handles them)
    """

    def __init__(
        self,
        strategy: str = "lexicon",
        custom_map: Optional[Dict[str, str]] = None,
    ):
        if strategy not in ("lexicon", "demojize", "remove", "keep"):
            raise ValueError(f"Unknown emoji strategy: {strategy}")
        self.strategy = strategy
        self.emoji_map = {**EMOJI_SENTIMENT_MAP, **(custom_map or {})}

    def normalize(self, text: str) -> str:
        if self.strategy == "keep":
            return text

        if self.strategy == "remove":
            return self._strip_emoji(text)

        if self.strategy == "demojize":
            try:
                import emoji as emoji_pkg
                return emoji_pkg.demojize(text, delimiters=(" ", " "))
            except ImportError:
                logger.warning("emoji package not installed; falling back to lexicon")
                self.strategy = "lexicon"

        # Lexicon strategy (default)
        for emoji_char, description in self.emoji_map.items():
            text = text.replace(emoji_char, f" {description} ")
        return text

    @staticmethod
    def _strip_emoji(text: str) -> str:
        """Remove emoji characters from text."""
        emoji_pattern = re.compile(
            "["
            "\U0001F300-\U0001FAFF"
            "\U00002600-\U000027BF"
            "\U0001F900-\U0001F9FF"
            "\U00002702-\U000027B0"
            "]+",
            flags=re.UNICODE
        )
        return emoji_pattern.sub(" ", text)


class ElongationNormalizer:
    """
    Reduce repeated characters to a maximum repetition count.

    Examples:
        "sooooooo" → "soo"  (max_reps=2)
        "noooo"    → "noo"
        "aaahhhh"  → "aahh"

    We preserve TWO characters (not one) to retain the sentiment signal
    of elongation (e.g., "good" vs "goood" both express positive sentiment,
    but the latter more intensely).
    """

    def __init__(self, max_repetitions: int = 2):
        if max_repetitions < 1:
            raise ValueError("max_repetitions must be >= 1")
        self.max_repetitions = max_repetitions
        # Match any character repeated more than max_repetitions times
        self._pattern = re.compile(
            r"(.)\1{" + str(max_repetitions) + r",}", re.UNICODE
        )
        self._replacement = r"\1" * max_repetitions

    def normalize(self, text: str) -> str:
        return self._pattern.sub(self._replacement, text)


class AbbreviationExpander:
    """
    Expand common informal abbreviations (case-insensitive).

    Only expands standalone words (not substrings) to avoid
    false positives (e.g., "gr8" → "great" but "great8" stays).
    """

    def __init__(
        self,
        custom_lexicon: Optional[Dict[str, str]] = None,
        case_sensitive: bool = False,
    ):
        self.case_sensitive = case_sensitive
        raw = {**DEFAULT_ABBREVIATIONS, **(custom_lexicon or {})}
        self.lexicon = raw if case_sensitive else {
            k.lower(): v for k, v in raw.items()
        }

    def normalize(self, text: str) -> str:
        tokens = text.split()
        expanded = []
        for token in tokens:
            key = token if self.case_sensitive else token.lower()
            # Strip trailing punctuation before lookup
            stripped_key = key.rstrip(".,!?;:")
            if stripped_key in self.lexicon:
                replacement = self.lexicon[stripped_key]
                # Preserve any trailing punctuation
                trailing = key[len(stripped_key):]
                expanded.append(replacement + trailing)
            else:
                expanded.append(token)
        return " ".join(expanded)


class SlangExpander:
    """
    Language-aware slang expander.

    Uses per-language lexicons. Only expands tokens whose LID-detected
    language matches the lexicon's language to prevent cross-lingual
    false expansions.
    """

    def __init__(
        self,
        custom_lexicons: Optional[Dict[str, Dict[str, str]]] = None,
        lexicon_dir: Optional[str] = None,
    ):
        # Merge built-in lexicons with any custom ones
        self.lexicons: Dict[str, Dict[str, str]] = {
            "en": dict(DEFAULT_SLANG_EN),
            "hi": dict(DEFAULT_SLANG_HI),
        }
        if custom_lexicons:
            for lang, lex in custom_lexicons.items():
                self.lexicons.setdefault(lang, {}).update(lex)

        # Load from JSON files if directory provided
        if lexicon_dir:
            self._load_lexicon_dir(lexicon_dir)

    def _load_lexicon_dir(self, lexicon_dir: str) -> None:
        """Load per-language JSON lexicon files from a directory."""
        lex_path = Path(lexicon_dir)
        if not lex_path.exists():
            logger.warning(f"Slang lexicon dir not found: {lexicon_dir}")
            return
        for json_file in lex_path.glob("*.json"):
            lang_code = json_file.stem
            try:
                with open(json_file, encoding="utf-8") as f:
                    loaded = json.load(f)
                self.lexicons.setdefault(lang_code, {}).update(loaded)
                logger.info(
                    f"Loaded {len(loaded)} slang entries for '{lang_code}'"
                )
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load {json_file}: {e}")

    def normalize(
        self,
        text: str,
        token_languages: Optional[List[str]] = None,
    ) -> str:
        """
        Expand slang in text, optionally using per-token language info.

        Args:
            text:             Input text string
            token_languages:  List of language codes (one per whitespace token).
                              If None, applies all lexicons.

        Returns:
            Text with slang terms expanded
        """
        tokens = text.split()
        expanded = []

        for i, token in enumerate(tokens):
            key = token.lower().rstrip(".,!?;:")
            trailing = token[len(token.rstrip(".,!?;:")):]

            # Determine which lexicons to try
            if token_languages and i < len(token_languages):
                lang = token_languages[i]
                lexicons_to_try = [self.lexicons.get(lang, {})]
            else:
                # Try all lexicons
                lexicons_to_try = list(self.lexicons.values())

            replaced = False
            for lex in lexicons_to_try:
                if key in lex:
                    expanded.append(lex[key] + trailing)
                    replaced = True
                    break

            if not replaced:
                expanded.append(token)

        return " ".join(expanded)


class CaseNormalizer:
    """
    Lowercase text while preserving acronyms and ALL-CAPS words
    (which carry sentiment intensity signals).

    Examples:
        "I LOVE this!"   → "I LOVE this!"   (LOVE preserved — emphasis)
        "NASA is great"  → "NASA is great"  (acronym preserved)
        "Great product"  → "great product"  (normal sentence case)
    """

    # Regex for ALL-CAPS words of 2+ characters (sentiment markers)
    ALLCAPS_PATTERN = re.compile(r"\b[A-Z]{2,}\b")

    def __init__(self, preserve_allcaps: bool = True):
        self.preserve_allcaps = preserve_allcaps

    def normalize(self, text: str) -> str:
        if not self.preserve_allcaps:
            return text.lower()

        # Find all ALL-CAPS words (indices)
        protected: List[Tuple[int, int, str]] = [
            (m.start(), m.end(), m.group())
            for m in self.ALLCAPS_PATTERN.finditer(text)
        ]

        # Lowercase everything
        lowered = text.lower()

        # Restore ALL-CAPS words
        offset = 0
        result = list(lowered)
        for start, end, original in protected:
            result[start:end] = list(original)

        return "".join(result)


class WhitespaceNormalizer:
    """Clean up irregular whitespace."""

    MULTI_SPACE = re.compile(r" {2,}")
    MULTI_NEWLINE = re.compile(r"\n{2,}")

    def normalize(self, text: str) -> str:
        text = text.strip()
        text = self.MULTI_SPACE.sub(" ", text)
        text = self.MULTI_NEWLINE.sub("\n", text)
        return text


# ─────────────────────────────────────────────
# Master Pipeline
# ─────────────────────────────────────────────

class TextNormalizer:
    """
    Orchestrates all normalization stages in the correct order.

    Each stage can be individually enabled/disabled via config.

    Usage:
        normalizer = TextNormalizer.from_config(cfg["preprocessing"])
        normalized = normalizer.normalize(raw_text)

        # With LID-guided slang expansion:
        normalized = normalizer.normalize(raw_text, token_languages=['en','hi','en'])
    """

    def __init__(
        self,
        url_normalizer:      Optional[URLMentionNormalizer]  = None,
        emoji_normalizer:    Optional[EmojiNormalizer]       = None,
        elongation_normalizer: Optional[ElongationNormalizer] = None,
        abbreviation_expander: Optional[AbbreviationExpander] = None,
        slang_expander:      Optional[SlangExpander]         = None,
        case_normalizer:     Optional[CaseNormalizer]        = None,
        whitespace_normalizer: Optional[WhitespaceNormalizer] = None,
    ):
        # Store stages (None means skip)
        self._stages = [
            ("url",          url_normalizer),
            ("emoji",        emoji_normalizer),
            ("elongation",   elongation_normalizer),
            ("abbreviation", abbreviation_expander),
            ("slang",        slang_expander),
            ("case",         case_normalizer),
            ("whitespace",   whitespace_normalizer),
        ]

    @classmethod
    def default(cls) -> "TextNormalizer":
        """
        Create a TextNormalizer with all stages enabled using default settings.
        Suitable for quick-start and testing.
        """
        return cls(
            url_normalizer=URLMentionNormalizer(),
            emoji_normalizer=EmojiNormalizer(strategy="lexicon"),
            elongation_normalizer=ElongationNormalizer(max_repetitions=2),
            abbreviation_expander=AbbreviationExpander(),
            slang_expander=SlangExpander(),
            case_normalizer=CaseNormalizer(preserve_allcaps=True),
            whitespace_normalizer=WhitespaceNormalizer(),
        )

    @classmethod
    def from_config(cls, config: dict) -> "TextNormalizer":
        """
        Instantiate TextNormalizer from a preprocessing config dict.

        Args:
            config: preprocessing sub-dict from preprocessing_config.yaml

        Returns:
            Configured TextNormalizer instance
        """
        cleaning = config.get("cleaning", {})

        url_norm = URLMentionNormalizer(
            url_token=config.get("special_tokens", {}).get("url_token", "[URL]"),
            mention_token=config.get("special_tokens", {}).get("mention_token", "[MENTION]"),
        )

        emoji_cfg = config.get("emoji", {})
        emoji_norm = EmojiNormalizer(
            strategy=emoji_cfg.get("strategy", "lexicon"),
        )

        elong_cfg = config.get("elongation", {})
        elong_norm = (
            ElongationNormalizer(
                max_repetitions=elong_cfg.get("max_repetitions", 2)
            ) if elong_cfg.get("enabled", True) else None
        )

        abbrev_norm = (
            AbbreviationExpander()
            if config.get("abbreviations", {}).get("enabled", True) else None
        )

        slang_cfg = config.get("slang", {})
        slang_norm = (
            SlangExpander(
                lexicon_dir=slang_cfg.get("lexicon_dir")
            ) if slang_cfg.get("enabled", True) else None
        )

        case_norm = CaseNormalizer(preserve_allcaps=True)
        ws_norm = WhitespaceNormalizer()

        return cls(
            url_normalizer=url_norm,
            emoji_normalizer=emoji_norm,
            elongation_normalizer=elong_norm,
            abbreviation_expander=abbrev_norm,
            slang_expander=slang_norm,
            case_normalizer=case_norm,
            whitespace_normalizer=ws_norm,
        )

    def normalize(
        self,
        text: str,
        token_languages: Optional[List[str]] = None,
    ) -> str:
        """
        Run the full normalization pipeline on input text.

        Args:
            text:             Raw input string
            token_languages:  Optional per-token language codes for LID-guided
                              slang expansion.

        Returns:
            Normalized text string
        """
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")

        for stage_name, stage in self._stages:
            if stage is None:
                continue
            try:
                if stage_name == "slang" and token_languages is not None:
                    text = stage.normalize(text, token_languages)
                else:
                    text = stage.normalize(text)
            except Exception as e:
                logger.error(f"Normalization stage '{stage_name}' failed: {e}")
                # Continue with unnormalized text for this stage

        return text

    def normalize_batch(
        self,
        texts: List[str],
        token_languages_list: Optional[List[Optional[List[str]]]] = None,
    ) -> List[str]:
        """
        Normalize a batch of texts.

        Args:
            texts:               List of raw input strings
            token_languages_list: Optional per-text token language lists

        Returns:
            List of normalized strings
        """
        results = []
        for i, text in enumerate(texts):
            tl = token_languages_list[i] if token_languages_list else None
            try:
                results.append(self.normalize(text, tl))
            except Exception as e:
                logger.error(f"Normalization failed for text index {i}: {e}")
                results.append(text)   # Fallback to raw text
        return results

    def active_stages(self) -> List[str]:
        """Return names of active (non-None) pipeline stages."""
        return [name for name, stage in self._stages if stage is not None]


# ─────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    normalizer = TextNormalizer.default()
    print("Active stages:", normalizer.active_stages())
    print()

    test_cases = [
        ("yaar this movie was sooooo bakwas omg 😡 check @user https://t.co/xyz",
         ["hi", "en", "en", "en", "en", "hi", "en", "und", "und", "und"]),
        ("I LOVE this product!! It's absolutely AMAZING 😍😍",
         None),
        ("Oh great, another delay. Just what I needed! 🙄",
         None),
        ("gr8 mast product tbh, bilkul bakwas nahi hai",
         ["en", "hi", "en", "en", "hi", "hi", "en", "en"]),
    ]

    for raw, langs in test_cases:
        normalized = normalizer.normalize(raw, langs)
        print(f"Raw:        {raw}")
        print(f"Normalized: {normalized}")
        print("-" * 70)
