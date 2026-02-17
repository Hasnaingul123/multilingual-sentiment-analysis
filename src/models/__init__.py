"""
Models Module

Contains model architectures for multilingual sentiment analysis:
- Base transformer models (XLM-RoBERTa, mBERT)
- Multi-task learning heads (sentiment, sarcasm)
- Language identification integration
- Custom attention mechanisms
"""

# Conditional imports for PyTorch-dependent modules
try:
    from models.task_heads import (
        LIDFusionLayer,
        SentimentHead,
        SarcasmHead,
    )
    from models.multitask_transformer import MultiTaskSentimentModel
    
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    LIDFusionLayer = None
    SentimentHead = None
    SarcasmHead = None
    MultiTaskSentimentModel = None

__all__ = [
    "LIDFusionLayer",
    "SentimentHead",
    "SarcasmHead",
    "MultiTaskSentimentModel",
]
