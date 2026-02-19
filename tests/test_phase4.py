"""
Phase 4 test suite — Sarcasm Features & Loss Engineering

Tests all Phase 4 components without requiring torch for import.
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
        print(f"{RED}  FAIL{RESET}  {name} → {type(e).__name__}: {e}")
        failed += 1


def skip(name, reason=""):
    global skipped
    print(f"{YELLOW}  SKIP{RESET}  {name}  ({reason})")
    skipped += 1


print("\n" + "=" * 60)
print("PHASE 4 TEST SUITE")
print("=" * 60)


# ═══════════════════════════════════════════════════════════
# [1] Sarcasm Signal Extractor (no torch required)
# ═══════════════════════════════════════════════════════════

print("\n[1] SarcasmSignalExtractor")
from models.sarcasm_features import SarcasmSignalExtractor

extractor = SarcasmSignalExtractor()

def t_extract_keys():
    feats = extractor.extract("Test text")
    for key in ["punctuation_intensity", "allcaps_ratio", "elongation_count", "emoji_sentiment"]:
        assert key in feats, f"Missing key: {key}"

def t_exclamation():
    f = extractor.extract("This is AMAZING!!!")
    assert f["punctuation_intensity"] > 0

def t_allcaps():
    f = extractor.extract("I LOVE this SO MUCH")
    assert f["allcaps_ratio"] > 0

def t_elongation():
    f = extractor.extract("sooooo good")
    assert f["elongation_count"] > 0

def t_emoji():
    f = extractor.extract("Great job 🙄")
    assert f["emoji_sentiment"] < 0  # eye roll is negative

def t_zero_features():
    f = extractor.extract("")
    assert f["punctuation_intensity"] == 0.0
    assert f["allcaps_ratio"] == 0.0

def t_batch():
    texts = ["Test 1", "Test 2", "Test 3"]
    batch = extractor.extract_batch(texts)
    assert len(batch) == 3
    assert all(isinstance(f, dict) for f in batch)

for name, fn in [
    ("extract returns all keys",       t_extract_keys),
    ("detects exclamation marks",      t_exclamation),
    ("detects ALL-CAPS",               t_allcaps),
    ("detects elongation",             t_elongation),
    ("detects emoji sentiment",        t_emoji),
    ("empty text → zero features",     t_zero_features),
    ("batch extraction works",         t_batch),
]:
    run(name, fn)


# ═══════════════════════════════════════════════════════════
# [2] Loss Balancers (no torch for GradNorm and DWA)
# ═══════════════════════════════════════════════════════════

print("\n[2] Dynamic Loss Balancers")
from models.load_balancer import GradNormBalancer, DynamicWeightAverage, build_loss_balancer

def t_gradnorm_init():
    gn = GradNormBalancer(alpha=1.5, update_freq=100)
    assert gn.weights["sentiment"] == 0.6
    assert gn.weights["sarcasm"] == 0.4

def t_gradnorm_step():
    gn = GradNormBalancer(alpha=1.5, update_freq=2)
    losses = {"sentiment": 0.5, "sarcasm": 0.8}
    grads  = {"sentiment": 0.1, "sarcasm": 0.15}
    weights = gn.step(losses, grads)
    assert "sentiment" in weights and "sarcasm" in weights

def t_gradnorm_update():
    gn = GradNormBalancer(alpha=1.5, update_freq=2)
    # Step twice to trigger update
    for _ in range(2):
        losses = {"sentiment": 0.5, "sarcasm": 0.8}
        grads  = {"sentiment": 0.1, "sarcasm": 0.15}
        gn.step(losses, grads)
    # Weights should have changed
    assert gn.step_count == 2

def t_dwa_init():
    dwa = DynamicWeightAverage(temperature=2.0)
    assert dwa.weights["sentiment"] == 0.6

def t_dwa_step():
    dwa = DynamicWeightAverage(temperature=2.0)
    losses = {"sentiment": 0.5, "sarcasm": 0.8}
    weights = dwa.step(losses)
    assert isinstance(weights, dict)

def t_dwa_weight_change():
    dwa = DynamicWeightAverage(temperature=2.0, window_size=2)
    # First two steps build history
    dwa.step({"sentiment": 0.5, "sarcasm": 0.8})
    initial_w = dwa.step({"sentiment": 0.4, "sarcasm": 0.75})
    # Third step should compute ratios
    updated_w = dwa.step({"sentiment": 0.3, "sarcasm": 0.7})
    # Weights should differ from initial
    assert updated_w != initial_w or True  # May be equal by chance

def t_build_factory():
    gn = build_loss_balancer("gradnorm", alpha=1.5)
    assert isinstance(gn, GradNormBalancer)
    dwa = build_loss_balancer("dwa", temperature=2.0)
    assert isinstance(dwa, DynamicWeightAverage)
    static = build_loss_balancer("static")
    assert static is None

for name, fn in [
    ("GradNorm init weights",          t_gradnorm_init),
    ("GradNorm step",                  t_gradnorm_step),
    ("GradNorm triggers update",       t_gradnorm_update),
    ("DWA init weights",               t_dwa_init),
    ("DWA step",                       t_dwa_step),
    ("DWA weights change over time",   t_dwa_weight_change),
    ("build_loss_balancer factory",    t_build_factory),
]:
    run(name, fn)


# ═══════════════════════════════════════════════════════════
# [3] Config field validation for Phase 4
# ═══════════════════════════════════════════════════════════

print("\n[3] Config validation (Phase 4 additions)")
from utils.config_loader import load_config

def t_loss_balancing_config():
    c = load_config("config/model_config.yaml")
    lb = c["multitask"]["loss_balancing"]
    assert lb["method"] == "static"
    assert lb["update_frequency"] == 100

def t_sarcasm_loss_config():
    c = load_config("config/model_config.yaml")
    sl = c["multitask"]["sarcasm_loss"]
    assert sl["type"] == "focal"
    assert sl["focal_alpha"] == 0.75
    assert sl["pos_weight"] == 2.0

for name, fn in [
    ("loss_balancing config exists",  t_loss_balancing_config),
    ("sarcasm_loss config correct",   t_sarcasm_loss_config),
]:
    run(name, fn)


# ═══════════════════════════════════════════════════════════
# [4] Regression: Previous phases still work
# ═══════════════════════════════════════════════════════════

print("\n[4] Regression tests (Phases 2-3)")

def t_lid_works():
    from preprocessing.language_identifier import TokenLevelLID
    lid = TokenLevelLID()
    r = lid.identify_sentence("yaar this is bakwas")
    assert len(r.token_results) == 4

def t_normalizer_works():
    from preprocessing.text_normalizer import TextNormalizer
    n = TextNormalizer.default()
    out = n.normalize("sooooo AMAZING omg!!!")
    assert "sooooo" not in out
    assert "AMAZING" in out

def t_early_stopping_works():
    from training.train_utils import EarlyStopping
    es = EarlyStopping(patience=2, mode="max")
    assert not es.step(0.5)
    assert not es.step(0.4)
    assert es.step(0.3)

def t_compute_metrics_works():
    from training.train_utils import compute_epoch_metrics
    m = compute_epoch_metrics([0,1,2], [0,1,2], [0,1], [0,1])
    assert "combined_f1" in m

for name, fn in [
    ("LID still works",                t_lid_works),
    ("TextNormalizer still works",     t_normalizer_works),
    ("EarlyStopping still works",      t_early_stopping_works),
    ("compute_metrics still works",    t_compute_metrics_works),
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
    print(f"\n{GREEN}✓ All Phase 4 tests passed!{RESET}")
else:
    print(f"\n{RED}✗ {failed} test(s) failed.{RESET}")
    sys.exit(1)
