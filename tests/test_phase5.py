"""
Phase 5 test suite — Evaluation, Robustness & Calibration

Tests all Phase 5 components without requiring torch for imports.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

passed = failed = skipped = 0


def run(name, fn):
    global passed, failed
    try:
        fn()
        print(f"{GREEN}  PASS{RESET}  {name}")
        passed += 1
    except Exception as e:
        print(f"{RED}  FAIL{RESET}  {name} → {type(e).__name__}: {str(e)[:70]}")
        failed += 1


print("\n" + "=" * 60)
print("PHASE 5 TEST SUITE")
print("=" * 60)


# ═══════════════════════════════════════════════════════════
# [1] Classification Metrics
# ═══════════════════════════════════════════════════════════

print("\n[1] ClassificationMetrics")
from evaluation.metrics import ClassificationMetrics
import numpy as np

np.random.seed(42)
N = 100
preds = np.random.randint(0, 3, N)
labels = np.random.randint(0, 3, N)

def t_compute_keys():
    m = ClassificationMetrics.compute(preds, labels, num_classes=3)
    for k in ["precision", "recall", "f1", "accuracy"]:
        assert k in m

def t_compute_values():
    m = ClassificationMetrics.compute(preds, labels, num_classes=3)
    assert 0.0 <= m["f1"] <= 1.0
    assert 0.0 <= m["accuracy"] <= 1.0

def t_per_class():
    pc = ClassificationMetrics.per_class_metrics(preds, labels, num_classes=3)
    assert len(pc) == 3
    assert "class_0" in pc

def t_per_class_custom_names():
    pc = ClassificationMetrics.per_class_metrics(
        preds, labels, num_classes=3, class_names=["neg", "neu", "pos"]
    )
    assert "neg" in pc and "neu" in pc and "pos" in pc

for name, fn in [
    ("compute returns all keys",       t_compute_keys),
    ("values in [0,1]",                t_compute_values),
    ("per_class metrics",              t_per_class),
    ("per_class custom names",         t_per_class_custom_names),
]:
    run(name, fn)


# ═══════════════════════════════════════════════════════════
# [2] Calibration Metrics
# ═══════════════════════════════════════════════════════════

print("\n[2] CalibrationMetrics")
from evaluation.metrics import CalibrationMetrics

probs = np.random.dirichlet([1, 1, 1], N)
labels_cal = np.random.randint(0, 3, N)

def t_ece_keys():
    ece = CalibrationMetrics.expected_calibration_error(probs, labels_cal)
    for k in ["ece", "mce", "num_bins", "bins"]:
        assert k in ece

def t_ece_values():
    ece = CalibrationMetrics.expected_calibration_error(probs, labels_cal)
    assert 0.0 <= ece["ece"] <= 1.0
    assert 0.0 <= ece["mce"] <= 1.0

def t_ece_bins():
    ece = CalibrationMetrics.expected_calibration_error(probs, labels_cal, num_bins=10)
    assert ece["num_bins"] == 10
    assert len(ece["bins"]) <= 10

def t_reliability_diagram():
    centers, accs, counts = CalibrationMetrics.reliability_diagram_data(probs, labels_cal)
    assert len(centers) == len(accs) == len(counts)

for name, fn in [
    ("ECE returns all keys",           t_ece_keys),
    ("ECE/MCE in [0,1]",               t_ece_values),
    ("ECE respects num_bins",          t_ece_bins),
    ("reliability diagram data",       t_reliability_diagram),
]:
    run(name, fn)


# ═══════════════════════════════════════════════════════════
# [3] Per-Language Metrics
# ═══════════════════════════════════════════════════════════

print("\n[3] PerLanguageMetrics")
from evaluation.metrics import PerLanguageMetrics

langs = np.random.choice(["en", "hi", "es"], N).tolist()

def t_per_lang_keys():
    pl = PerLanguageMetrics.compute(preds, labels, langs, num_classes=3)
    assert len(pl) > 0
    first_lang = list(pl.keys())[0]
    for k in ["f1", "accuracy", "count"]:
        assert k in pl[first_lang]

def t_per_lang_counts():
    pl = PerLanguageMetrics.compute(preds, labels, langs, num_classes=3)
    total_count = sum(m["count"] for m in pl.values())
    assert total_count == N

for name, fn in [
    ("per-language keys correct",      t_per_lang_keys),
    ("counts sum to N",                t_per_lang_counts),
]:
    run(name, fn)


# ═══════════════════════════════════════════════════════════
# [4] Confusion Matrix
# ═══════════════════════════════════════════════════════════

print("\n[4] ConfusionMatrixMetrics")
from evaluation.metrics import ConfusionMatrixMetrics

def t_cm_shape():
    cm = ConfusionMatrixMetrics.compute(preds, labels, num_classes=3)
    assert cm.shape == (3, 3)

def t_cm_normalize():
    cm = ConfusionMatrixMetrics.compute(preds, labels, num_classes=3, normalize="true")
    # Row sums should be ~1 (allowing for rounding errors and zero rows)
    row_sums = cm.sum(axis=1)
    non_zero_rows = row_sums > 0
    assert np.allclose(row_sums[non_zero_rows], 1.0, atol=0.01)

def t_most_confused():
    cm = ConfusionMatrixMetrics.compute(preds, labels, num_classes=3)
    confused = ConfusionMatrixMetrics.most_confused_pairs(cm, top_k=3)
    assert len(confused) <= 3

for name, fn in [
    ("confusion matrix shape",         t_cm_shape),
    ("normalize='true' sums to 1",     t_cm_normalize),
    ("most_confused_pairs",            t_most_confused),
]:
    run(name, fn)


# ═══════════════════════════════════════════════════════════
# [5] MultiTaskEvaluator
# ═══════════════════════════════════════════════════════════

print("\n[5] MultiTaskEvaluator")
from evaluation.metrics import MultiTaskEvaluator

evaluator = MultiTaskEvaluator()

sent_preds = np.random.randint(0, 3, N)
sent_labels = np.random.randint(0, 3, N)
sent_probs = np.random.dirichlet([1, 1, 1], N)

sarc_preds = np.random.randint(0, 2, N)
sarc_labels = np.random.randint(0, 2, N)
sarc_probs = np.random.rand(N)

def t_eval_keys():
    r = evaluator.evaluate(
        sent_preds, sent_labels, sent_probs,
        sarc_preds, sarc_labels, sarc_probs
    )
    assert "sentiment" in r and "sarcasm" in r and "combined" in r

def t_eval_combined_f1():
    r = evaluator.evaluate(
        sent_preds, sent_labels, sent_probs,
        sarc_preds, sarc_labels, sarc_probs
    )
    # Combined F1 formula
    expected = 0.6 * r["sentiment"]["overall"]["f1"] + 0.4 * r["sarcasm"]["overall"]["f1"]
    assert abs(r["combined"]["combined_f1"] - expected) < 1e-6

def t_eval_with_languages():
    r = evaluator.evaluate(
        sent_preds, sent_labels, sent_probs,
        sarc_preds, sarc_labels, sarc_probs,
        languages=langs
    )
    assert "per_language" in r

for name, fn in [
    ("evaluate returns all keys",      t_eval_keys),
    ("combined_f1 formula correct",    t_eval_combined_f1),
    ("evaluate with languages",        t_eval_with_languages),
]:
    run(name, fn)


# ═══════════════════════════════════════════════════════════
# [6] Text Perturbations
# ═══════════════════════════════════════════════════════════

print("\n[6] TextPerturbations")
from evaluation.robustness import TextPerturbations
import random

random.seed(42)
test_text = "This product is AMAZING!!! 😍"

def t_random_typos():
    perturbed = TextPerturbations.random_typos(test_text, typo_prob=0.1)
    assert isinstance(perturbed, str)

def t_remove_punctuation():
    p = TextPerturbations.remove_punctuation(test_text)
    assert "!" not in p

def t_remove_emoji():
    p = TextPerturbations.remove_emoji(test_text)
    assert "😍" not in p

def t_add_emoji():
    p = TextPerturbations.add_random_emoji(test_text, num_emoji=2)
    assert len(p) > len(test_text)

def t_flip_case():
    p = TextPerturbations.flip_case(test_text, flip_prob=0.3)
    assert isinstance(p, str)

for name, fn in [
    ("random_typos",                   t_random_typos),
    ("remove_punctuation",             t_remove_punctuation),
    ("remove_emoji",                   t_remove_emoji),
    ("add_random_emoji",               t_add_emoji),
    ("flip_case",                      t_flip_case),
]:
    run(name, fn)


# ═══════════════════════════════════════════════════════════
# [7] Robustness Test Suite
# ═══════════════════════════════════════════════════════════

print("\n[7] RobustnessTestSuite")
from evaluation.robustness import RobustnessTestSuite

suite = RobustnessTestSuite()

texts_robust = ["Great!", "Terrible.", "Okay."] * 5
labels_robust = [2, 0, 1] * 5

def mock_predict(texts):
    # Deterministic for testing
    return [len(t) % 3 for t in texts]

def t_suite_apply():
    perturbed = suite.apply_perturbation(texts_robust, "no_punctuation")
    assert len(perturbed) == len(texts_robust)

def t_suite_run():
    results = suite.run_tests(texts_robust, mock_predict, labels_robust)
    assert "baseline" in results
    assert "typos_light" in results

def t_suite_summarize():
    results = suite.run_tests(texts_robust, mock_predict, labels_robust)
    summary = suite.summarize(results)
    assert "avg_drop" in summary and "max_drop" in summary

for name, fn in [
    ("apply_perturbation",             t_suite_apply),
    ("run_tests",                      t_suite_run),
    ("summarize",                      t_suite_summarize),
]:
    run(name, fn)


# ═══════════════════════════════════════════════════════════
# [8] Regression (Phases 2-4)
# ═══════════════════════════════════════════════════════════

print("\n[8] Regression")

def t_lid():
    from preprocessing.language_identifier import TokenLevelLID
    lid = TokenLevelLID()
    r = lid.identify_sentence("test")
    assert len(r.token_results) == 1

def t_normalizer():
    from preprocessing.text_normalizer import TextNormalizer
    n = TextNormalizer.default()
    out = n.normalize("sooooo AMAZING")
    assert "sooooo" not in out and "AMAZING" in out

def t_early_stopping():
    from training.train_utils import EarlyStopping
    es = EarlyStopping(patience=2, mode="max")
    es.step(0.5); es.step(0.4); assert es.step(0.3)

def t_sarcasm_extractor():
    import sys, importlib.util
    spec = importlib.util.spec_from_file_location("sf", "src/models/sarcasm_features.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ext = mod.SarcasmSignalExtractor()
    f = ext.extract("AMAZING!!!")
    assert f["punctuation_intensity"] > 0

for name, fn in [
    ("LID",                            t_lid),
    ("Normalizer",                     t_normalizer),
    ("EarlyStopping",                  t_early_stopping),
    ("SarcasmExtractor",               t_sarcasm_extractor),
]:
    run(name, fn)


# ═══════════════════════════════════════════════════════════
# Results
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
total = passed + failed + skipped
print(f"  {GREEN}PASSED{RESET}:  {passed}")
print(f"  {RED}FAILED{RESET}:  {failed}")
print(f"  {YELLOW}SKIPPED{RESET}: {skipped}")
print(f"  TOTAL:   {total}")

if failed == 0:
    print(f"\n{GREEN}✓ All Phase 5 tests passed!{RESET}")
else:
    print(f"\n{RED}✗ {failed} test(s) failed.{RESET}")
    sys.exit(1)
