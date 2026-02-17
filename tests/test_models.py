"""
Phase 3 Test Suite

Comprehensive tests for model architecture and training components:
- EarlyStopping logic
- Metrics computation (multi-task F1, accuracy)
- Configuration validation
- Module structure and imports
- Phase 2 regression tests
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import unittest
from typing import List


class TestEarlyStopping(unittest.TestCase):
    """Test early stopping logic."""

    def setUp(self):
        from training.train_utils import EarlyStopping
        self.EarlyStopping = EarlyStopping

    def test_patience_triggers_stop(self):
        """Early stopping should trigger after patience epochs without improvement."""
        es = self.EarlyStopping(patience=2, mode="max")
        self.assertFalse(es.step(0.5))
        self.assertFalse(es.step(0.4))  # No improvement
        self.assertTrue(es.step(0.3))   # Patience exhausted

    def test_improvement_resets_counter(self):
        """Improvement should reset the patience counter."""
        es = self.EarlyStopping(patience=3, mode="max")
        self.assertFalse(es.step(0.5))
        self.assertFalse(es.step(0.6))  # Improvement - counter resets
        self.assertFalse(es.step(0.5))
        self.assertFalse(es.step(0.5))
        self.assertTrue(es.step(0.5))   # Now patience exhausted

    def test_mode_min_works(self):
        """Mode='min' should work for loss minimization."""
        es = self.EarlyStopping(patience=2, mode="min")
        self.assertFalse(es.step(1.0))
        self.assertFalse(es.step(0.9))  # Improvement
        self.assertFalse(es.step(1.0))
        self.assertTrue(es.step(1.0))   # Patience exhausted

    def test_bad_mode_raises_error(self):
        """Invalid mode should raise ValueError."""
        with self.assertRaises(ValueError):
            self.EarlyStopping(mode="diagonal")

    def test_first_step_never_stops(self):
        """First step should never trigger early stopping."""
        es = self.EarlyStopping(patience=1, mode="max")
        self.assertFalse(es.step(0.9))

    def test_reset_works(self):
        """Reset should clear all state."""
        es = self.EarlyStopping(patience=2, mode="max")
        es.step(0.5)
        es.step(0.4)
        es.reset()
        self.assertEqual(es.counter, 0)
        self.assertIsNone(es.best_score)
        self.assertFalse(es.should_stop)


class TestMetricsComputation(unittest.TestCase):
    """Test multi-task metrics computation."""

    def setUp(self):
        from training.train_utils import compute_epoch_metrics
        self.compute_epoch_metrics = compute_epoch_metrics

    def test_all_keys_present(self):
        """All expected metric keys should be present."""
        m = self.compute_epoch_metrics(
            [0, 1, 2, 0], [0, 1, 2, 1],
            [0, 1, 0, 1], [0, 1, 1, 0]
        )
        expected_keys = [
            "sentiment_f1", "sentiment_acc", "sentiment_precision", "sentiment_recall",
            "sarcasm_f1", "sarcasm_acc", "sarcasm_precision", "sarcasm_recall",
            "combined_f1"
        ]
        for key in expected_keys:
            self.assertIn(key, m, f"Missing key: {key}")

    def test_combined_formula_correct(self):
        """Combined F1 should follow the formula: 0.6*sent_f1 + 0.4*sarc_f1."""
        m = self.compute_epoch_metrics(
            [0] * 10, [0] * 10,
            [1] * 10, [1] * 10
        )
        expected = round(0.6 * m["sentiment_f1"] + 0.4 * m["sarcasm_f1"], 4)
        self.assertAlmostEqual(m["combined_f1"], expected, places=4)

    def test_perfect_predictions(self):
        """Perfect predictions should yield F1=1.0."""
        labels = [0, 1, 2, 0, 1, 2]
        m = self.compute_epoch_metrics(
            labels, labels,
            [0, 1, 0, 0, 1, 0], [0, 1, 0, 0, 1, 0]
        )
        self.assertEqual(m["sentiment_f1"], 1.0)
        self.assertEqual(m["sarcasm_acc"], 1.0)

    def test_values_in_valid_range(self):
        """All metric values should be in [0, 1]."""
        m = self.compute_epoch_metrics(
            [0] * 5, [0] * 5,
            [0] * 5, [0] * 5
        )
        for key, value in m.items():
            self.assertGreaterEqual(value, 0.0, f"{key} < 0")
            self.assertLessEqual(value, 1.0, f"{key} > 1")

    def test_values_are_floats(self):
        """All metric values should be floats."""
        m = self.compute_epoch_metrics(
            [0, 1], [0, 1],
            [0, 1], [0, 1]
        )
        for key, value in m.items():
            self.assertIsInstance(value, float, f"{key} is not float")


class TestConfigValidation(unittest.TestCase):
    """Test configuration file validation."""

    def setUp(self):
        from utils.config_loader import load_config
        self.load_config = load_config

    def test_model_config_fields(self):
        """Model config should have all required fields."""
        c = self.load_config("config/model_config.yaml")
        self.assertEqual(c["model"]["base_model"], "xlm-roberta-base")
        self.assertEqual(c["model"]["sentiment"]["num_classes"], 3)
        self.assertEqual(c["multitask"]["sentiment_weight"], 0.6)
        self.assertEqual(c["multitask"]["sarcasm_weight"], 0.4)

    def test_lid_config_fields(self):
        """LID config should have all required fields."""
        c = self.load_config("config/model_config.yaml")
        lid = c["model"]["lid_integration"]
        self.assertTrue(lid["enabled"])
        self.assertEqual(lid["num_languages"], 10)

    def test_loss_config_fields(self):
        """Loss config should have all required fields."""
        c = self.load_config("config/model_config.yaml")
        self.assertEqual(c["multitask"]["sentiment_loss"]["type"], "focal")
        self.assertEqual(c["multitask"]["sarcasm_loss"]["pos_weight"], 2.0)

    def test_training_config_fields(self):
        """Training config should have all required fields."""
        c = self.load_config("config/training_config.yaml")
        t = c["training"]
        self.assertEqual(t["num_epochs"], 10)
        self.assertEqual(t["batch_size"], 16)
        self.assertEqual(t["optimizer"]["learning_rate"], 2e-5)
        self.assertEqual(t["early_stopping"]["patience"], 3)

    def test_preprocessing_config_fields(self):
        """Preprocessing config should have all required fields."""
        c = self.load_config("config/preprocessing_config.yaml")
        self.assertTrue(c["preprocessing"]["language_identification"]["enabled"])
        self.assertEqual(c["preprocessing"]["tokenization"]["max_length"], 128)


class TestModuleStructure(unittest.TestCase):
    """Test module files exist and can be imported."""

    def test_files_exist(self):
        """All Phase 3 module files should exist."""
        files = [
            "src/training/focal_loss.py",
            "src/models/task_heads.py",
            "src/models/multitask_transformer.py",
            "src/training/trainer.py",
            "src/training/train_utils.py",
        ]
        for filepath in files:
            path = Path(__file__).parent.parent / filepath
            self.assertTrue(path.exists(), f"Missing file: {filepath}")

    def test_train_utils_imports(self):
        """train_utils should export EarlyStopping and compute_epoch_metrics."""
        from training.train_utils import EarlyStopping, compute_epoch_metrics
        self.assertIsNotNone(EarlyStopping)
        self.assertIsNotNone(compute_epoch_metrics)

    def test_training_init_exports(self):
        """training __init__ should re-export key components."""
        from training import EarlyStopping, compute_epoch_metrics
        self.assertIsNotNone(EarlyStopping)
        self.assertIsNotNone(compute_epoch_metrics)


class TestPhase2Regression(unittest.TestCase):
    """Regression tests to ensure Phase 2 components still work."""

    def test_lid_still_works(self):
        """LID identify_sentence should still work."""
        from preprocessing.language_identifier import TokenLevelLID
        lid = TokenLevelLID()
        result = lid.identify_sentence("yaar this is sooo bakwas!")
        self.assertEqual(len(result.token_results), 5)

    def test_normalizer_still_works(self):
        """TextNormalizer pipeline should still work."""
        from preprocessing.text_normalizer import TextNormalizer
        normalizer = TextNormalizer.default()
        output = normalizer.normalize("sooooo AMAZING 😍 omg @user https://x.com")
        self.assertNotIn("sooooo", output)  # Elongation reduced
        self.assertIn("AMAZING", output)    # Case preserved
        self.assertIn("[URL]", output)      # URL replaced


def run_tests():
    """Run all tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEarlyStopping))
    suite.addTests(loader.loadTestsFromTestCase(TestMetricsComputation))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestModuleStructure))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase2Regression))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
