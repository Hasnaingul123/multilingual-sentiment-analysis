"""Data module — dataset building, loading, and torch wrappers."""

from .dataset_builder import (
    SentimentDatasetBuilder,
    SampleProcessor,
    generate_synthetic_dataset,
    SENTIMENT_LABEL_MAP,
    SARCASM_LABEL_MAP,
    LANGUAGE_ID_MAP,
)

try:
    from .dataloader import (
        MultiTaskSentimentDataset,
        build_dataloaders,
        build_inference_loader,
        compute_class_weights,
    )
    _TORCH_EXPORTS = [
        "MultiTaskSentimentDataset",
        "build_dataloaders",
        "build_inference_loader",
        "compute_class_weights",
    ]
except Exception:
    _TORCH_EXPORTS = []

__all__ = [
    "SentimentDatasetBuilder",
    "SampleProcessor",
    "generate_synthetic_dataset",
    "SENTIMENT_LABEL_MAP",
    "SARCASM_LABEL_MAP",
    "LANGUAGE_ID_MAP",
] + _TORCH_EXPORTS
