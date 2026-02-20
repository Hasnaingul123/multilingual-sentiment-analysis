"""Evaluation module — metrics, calibration, and robustness testing."""

from .metrics import (
    ClassificationMetrics,
    CalibrationMetrics,
    PerLanguageMetrics,
    ConfusionMatrixMetrics,
    MultiTaskEvaluator,
)
from .robustness import (
    TextPerturbations,
    RobustnessTestSuite,
)

# Torch-dependent exports
try:
    from .calibration import (
        TemperatureScaling,
        BinaryTemperatureScaling,
        MultiTaskCalibrator,
    )
    _CALIB_EXPORTS = [
        "TemperatureScaling",
        "BinaryTemperatureScaling",
        "MultiTaskCalibrator",
    ]
except Exception:
    _CALIB_EXPORTS = []

__all__ = [
    "ClassificationMetrics",
    "CalibrationMetrics",
    "PerLanguageMetrics",
    "ConfusionMatrixMetrics",
    "MultiTaskEvaluator",
    "TextPerturbations",
    "RobustnessTestSuite",
] + _CALIB_EXPORTS
