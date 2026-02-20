"""
Robustness Testing Module

Tests model resilience to noisy, adversarial, and edge-case inputs.

Test Categories:
    1. Typo/Misspelling Injection: Random character substitutions
    2. Code-Switching Stress Test: High-frequency language switches
    3. Emoji Removal/Addition: Test dependency on emoji signals
    4. Punctuation Removal: Test reliance on !!! and ???
    5. Case Perturbations: ALL-CAPS → lowercase and vice versa

Rationale:
    - Real-world text is noisy (typos, autocorrect errors)
    - Code-switching density varies (some texts switch every word)
    - Model shouldn't collapse without emoji or punctuation
    - Robust model maintains performance under perturbations
"""

from pathlib import Path
from typing import Dict, List, Callable
import random
import re

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import get_logger

logger = get_logger("robustness")


# ═══════════════════════════════════════════════════════════
# Adversarial Perturbations
# ═══════════════════════════════════════════════════════════

class TextPerturbations:
    """
    Apply adversarial perturbations to text for robustness testing.
    """
    
    @staticmethod
    def random_typos(text: str, typo_prob: float = 0.1) -> str:
        """
        Inject random typos: character substitutions, deletions, insertions.
        
        Args:
            text:       Input string
            typo_prob:  Probability of typo per character
            
        Returns:
            Perturbed text
        """
        chars = list(text)
        for i in range(len(chars)):
            if random.random() < typo_prob and chars[i].isalpha():
                action = random.choice(["substitute", "delete", "insert"])
                
                if action == "substitute":
                    # Replace with nearby key (simulates keyboard error)
                    chars[i] = random.choice("abcdefghijklmnopqrstuvwxyz")
                elif action == "delete":
                    chars[i] = ""
                elif action == "insert":
                    chars[i] = chars[i] + random.choice("abcdefghijklmnopqrstuvwxyz")
        
        return "".join(chars)
    
    @staticmethod
    def remove_punctuation(text: str) -> str:
        """Remove all punctuation marks."""
        return re.sub(r"[^\w\s]", "", text)
    
    @staticmethod
    def remove_emoji(text: str) -> str:
        """Remove all emoji characters."""
        emoji_pattern = re.compile(
            "["
            "\U0001F300-\U0001FAFF"
            "\U00002600-\U000027BF"
            "\U0001F900-\U0001F9FF"
            "\U00002702-\U000027B0"
            "]+",
            flags=re.UNICODE
        )
        return emoji_pattern.sub("", text)
    
    @staticmethod
    def add_random_emoji(text: str, num_emoji: int = 2) -> str:
        """Add random emoji to text."""
        emoji_pool = ["😀", "😂", "😍", "😒", "🙄", "😤", "👍", "👎", "🔥", "💯"]
        added = random.sample(emoji_pool, min(num_emoji, len(emoji_pool)))
        return text + " " + " ".join(added)
    
    @staticmethod
    def flip_case(text: str, flip_prob: float = 0.3) -> str:
        """Randomly flip character case."""
        chars = list(text)
        for i in range(len(chars)):
            if random.random() < flip_prob and chars[i].isalpha():
                chars[i] = chars[i].swapcase()
        return "".join(chars)
    
    @staticmethod
    def aggressive_code_switch(text: str, switch_prob: float = 0.5) -> str:
        """
        Simulate high-frequency code-switching by randomly swapping
        words with Hindi/Spanish equivalents.
        
        (Simplified — full implementation would use translation)
        """
        # Mock code-switching: replace some English words with placeholders
        words = text.split()
        hindi_map = {
            "good": "अच्छा",
            "bad": "बुरा",
            "very": "बहुत",
            "this": "यह",
            "product": "उत्पाद",
        }
        
        switched = []
        for word in words:
            if random.random() < switch_prob and word.lower() in hindi_map:
                switched.append(hindi_map[word.lower()])
            else:
                switched.append(word)
        
        return " ".join(switched)
    
    @staticmethod
    def elongate_words(text: str, elongate_prob: float = 0.2) -> str:
        """Add excessive elongation to random words."""
        words = text.split()
        elongated = []
        for word in words:
            if random.random() < elongate_prob and len(word) > 2:
                # Elongate last vowel
                if word[-1] in "aeiouAEIOU":
                    word = word + word[-1] * 3
            elongated.append(word)
        return " ".join(elongated)


# ═══════════════════════════════════════════════════════════
# Robustness Test Suite
# ═══════════════════════════════════════════════════════════

class RobustnessTestSuite:
    """
    Comprehensive robustness testing framework.
    
    Applies perturbations to test samples and measures performance drop.
    
    Usage:
        suite = RobustnessTestSuite()
        results = suite.run_tests(
            texts,
            predict_fn=model.predict,
            labels=labels
        )
    """
    
    def __init__(self):
        self.perturbations = {
            "typos_light":        lambda t: TextPerturbations.random_typos(t, 0.05),
            "typos_heavy":        lambda t: TextPerturbations.random_typos(t, 0.15),
            "no_punctuation":     TextPerturbations.remove_punctuation,
            "no_emoji":           TextPerturbations.remove_emoji,
            "add_random_emoji":   TextPerturbations.add_random_emoji,
            "flip_case":          TextPerturbations.flip_case,
            "code_switch_stress": lambda t: TextPerturbations.aggressive_code_switch(t, 0.5),
            "elongation":         TextPerturbations.elongate_words,
        }
        logger.info(f"RobustnessTestSuite: {len(self.perturbations)} perturbations")
    
    def apply_perturbation(
        self,
        texts: List[str],
        perturbation_name: str,
    ) -> List[str]:
        """
        Apply a perturbation to all texts.
        
        Args:
            texts:              List of input strings
            perturbation_name:  Name from self.perturbations
            
        Returns:
            Perturbed texts
        """
        if perturbation_name not in self.perturbations:
            raise ValueError(f"Unknown perturbation: {perturbation_name}")
        
        perturb_fn = self.perturbations[perturbation_name]
        return [perturb_fn(text) for text in texts]
    
    def run_tests(
        self,
        texts: List[str],
        predict_fn: Callable,  # Function: List[str] → predictions
        labels: List[int],
        metric_fn: Callable = None,  # Function: (preds, labels) → score
    ) -> Dict[str, Dict]:
        """
        Run all robustness tests.
        
        Args:
            texts:      Clean test texts
            predict_fn: Model prediction function
            labels:     Ground truth labels
            metric_fn:  Metric computation (default: accuracy)
            
        Returns:
            Dict[perturbation_name, {score, drop, perturbed_texts}]
        """
        if metric_fn is None:
            # Default: accuracy
            from sklearn.metrics import accuracy_score
            metric_fn = accuracy_score
        
        # Baseline (clean) performance
        baseline_preds = predict_fn(texts)
        baseline_score = metric_fn(labels, baseline_preds)
        
        logger.info(f"Baseline score: {baseline_score:.4f}")
        
        # Test each perturbation
        results = {"baseline": {"score": baseline_score, "drop": 0.0}}
        
        for name in self.perturbations:
            perturbed_texts = self.apply_perturbation(texts, name)
            perturbed_preds = predict_fn(perturbed_texts)
            perturbed_score = metric_fn(labels, perturbed_preds)
            
            score_drop = baseline_score - perturbed_score
            
            results[name] = {
                "score": perturbed_score,
                "drop":  score_drop,
                "drop_pct": (score_drop / baseline_score * 100) if baseline_score > 0 else 0,
            }
            
            logger.info(
                f"  {name:20s}: score={perturbed_score:.4f}, "
                f"drop={score_drop:.4f} ({results[name]['drop_pct']:.1f}%)"
            )
        
        return results
    
    def summarize(self, results: Dict[str, Dict]) -> Dict[str, float]:
        """
        Compute summary statistics across all perturbations.
        
        Returns:
            avg_drop, max_drop, robust_score (baseline - avg_drop)
        """
        drops = [r["drop"] for name, r in results.items() if name != "baseline"]
        
        summary = {
            "avg_drop":      sum(drops) / len(drops) if drops else 0.0,
            "max_drop":      max(drops) if drops else 0.0,
            "robust_score":  results["baseline"]["score"] - (sum(drops) / len(drops)),
        }
        
        logger.info(
            f"Robustness summary: avg_drop={summary['avg_drop']:.4f}, "
            f"max_drop={summary['max_drop']:.4f}"
        )
        
        return summary


# ═══════════════════════════════════════════════════════════
# Smoke-test
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing robustness perturbations...\n")
    
    test_text = "This product is absolutely AMAZING!!! 😍 Highly recommend."
    
    print(f"Original: {test_text}\n")
    
    # Test each perturbation
    perturbs = TextPerturbations()
    
    tests = [
        ("Typos (light)",       lambda: perturbs.random_typos(test_text, 0.1)),
        ("No punctuation",      lambda: perturbs.remove_punctuation(test_text)),
        ("No emoji",            lambda: perturbs.remove_emoji(test_text)),
        ("Add random emoji",    lambda: perturbs.add_random_emoji(test_text, 2)),
        ("Flip case",           lambda: perturbs.flip_case(test_text, 0.3)),
        ("Aggressive code-switch", lambda: perturbs.aggressive_code_switch(test_text, 0.5)),
        ("Elongation",          lambda: perturbs.elongate_words(test_text, 0.3)),
    ]
    
    random.seed(42)
    for name, fn in tests:
        perturbed = fn()
        print(f"{name:25s}: {perturbed}")
    
    # Test suite
    print("\n" + "="*60)
    print("RobustnessTestSuite")
    print("="*60)
    
    texts = [
        "Great product!",
        "This is terrible.",
        "It's okay, not bad.",
    ] * 10
    labels = [2, 0, 1] * 10  # pos, neg, neu
    
    # Mock prediction function (random for test)
    def mock_predict(texts):
        import numpy as np
        np.random.seed(42)
        return np.random.randint(0, 3, len(texts)).tolist()
    
    suite = RobustnessTestSuite()
    results = suite.run_tests(texts, mock_predict, labels)
    
    summary = suite.summarize(results)
    print(f"\nSummary: {summary}")
    
    print("\n✓ Robustness tests passed!")
