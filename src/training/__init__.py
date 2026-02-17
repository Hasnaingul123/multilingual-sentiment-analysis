"""
Training Module

Model training pipeline:
- Training loops and optimization
- Multi-task loss computation
- Learning rate scheduling
- Checkpointing and model saving
- Training metrics logging
"""

# Conditional imports for PyTorch-dependent modules
try:
    from training.focal_loss import (
        FocalLoss,
        BinaryFocalLoss,
        CompositeLoss,
    )
    from training.trainer import (
        Trainer,
        build_trainer,
        set_seed,
        build_linear_warmup_scheduler,
    )
    from training.train_utils import (
        EarlyStopping,
        compute_epoch_metrics,
        print_classification_report,
    )
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    FocalLoss = None
    BinaryFocalLoss = None
    CompositeLoss = None
    Trainer = None
    build_trainer = None
    set_seed = None
    build_linear_warmup_scheduler = None
    EarlyStopping = None
    compute_epoch_metrics = None
    print_classification_report = None

__all__ = [
    # Loss functions
    "FocalLoss",
    "BinaryFocalLoss",
    "CompositeLoss",
    # Trainer
    "Trainer",
    "build_trainer",
    "set_seed",
    "build_linear_warmup_scheduler",
    # Utilities
    "EarlyStopping",
    "compute_epoch_metrics",
    "print_classification_report",
]
