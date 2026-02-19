"""Models module — neural architectures, loss functions, and task heads."""

try:
    from .focal_loss import FocalLoss, BinaryFocalLoss, CompositeLoss
    from .task_heads import SentimentHead, SarcasmHead, LIDFusionLayer
    from .multitask_transformer import MultiTaskSentimentModel
    from .sarcasm_features import (
        SarcasmSignalExtractor,
        EnhancedSarcasmHead,
        SarcasmAuxiliaryFeatures,
        SentimentIncongruenceModule,
    )
    from .load_balancer import (
        GradNormBalancer,
        DynamicWeightAverage,
        UncertaintyWeighting,
        build_loss_balancer,
    )
    from .contrastive_loss import (
        SupervisedContrastiveLoss,
        TripletContrastiveLoss,
        ContrastiveLossWrapper,
    )
    __all__ = [
        "FocalLoss", "BinaryFocalLoss", "CompositeLoss",
        "SentimentHead", "SarcasmHead", "LIDFusionLayer",
        "MultiTaskSentimentModel",
        "SarcasmSignalExtractor", "EnhancedSarcasmHead",
        "SarcasmAuxiliaryFeatures", "SentimentIncongruenceModule",
        "GradNormBalancer", "DynamicWeightAverage", "UncertaintyWeighting",
        "build_loss_balancer",
        "SupervisedContrastiveLoss", "TripletContrastiveLoss",
        "ContrastiveLossWrapper",
    ]
except ImportError:
    __all__ = []

