# Multilingual Customer Sentiment Analysis System

A production-grade Transformer-based NLP pipeline for sentiment analysis on multilingual, code-switched text with slang and sarcasm detection.

## 🎯 Project Overview

This system addresses the challenge of analyzing customer sentiment in social media text that contains:
- **Code-switching** (e.g., Hinglish: "yaar this product is bakwas")
- **Slang and informal language** (e.g., "lit", "salty", "GOAT")
- **Sarcasm** (e.g., "Oh great, another delay. Just what I needed!")
- **Multilingual content** (10+ languages supported)

### Key Features

✅ **Multi-task Learning**: Joint sentiment classification + sarcasm detection  
✅ **Token-level Language Identification**: Handles intra-sentence language switches  
✅ **Robust Preprocessing**: Slang normalization, emoji handling, elongation reduction  
✅ **Transformer-based Architecture**: Built on XLM-RoBERTa for multilingual understanding  
✅ **Uncertainty Estimation**: Confidence scores via temperature scaling / MC Dropout  
✅ **Production-ready**: Modular design, comprehensive logging, error handling

## 🏗️ Architecture

```
Input Text (Multilingual, Code-switched)
    ↓
Token-level Language Identification
    ↓
Preprocessing Pipeline (Normalization + Tokenization)
    ↓
XLM-RoBERTa Encoder (Shared)
    ↓
    ├─→ Sentiment Head (3-class: neg/neu/pos)
    └─→ Sarcasm Head (binary)
    ↓
Calibrated Predictions + Uncertainty Scores
```

### Multi-task Loss Function

```
L_total = λ₁ * FocalLoss(sentiment) + λ₂ * FocalLoss(sarcasm)
```

Where Focal Loss addresses class imbalance:
```
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
```

## 📋 Requirements

- Python 3.8+
- PyTorch 2.1+
- Transformers 4.36+
- 8GB+ RAM (16GB recommended)
- GPU optional but recommended for training

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Hasnaingul123/multilingual-sentiment-analysis.git
cd multilingual-sentiment-analysis
```

### 2. Create Virtual Environment

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install core dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt

# Install package in editable mode
pip install -e .
```

### 4. Verify Installation

```bash
python -c "import torch; import transformers; print('✓ Installation successful!')"
```

## 📁 Project Structure

```
multilingual_sentiment_analysis/
├── config/                      # Configuration files
│   ├── model_config.yaml       # Model architecture settings
│   ├── training_config.yaml    # Training hyperparameters
│   └── preprocessing_config.yaml # Data preprocessing settings
├── src/
│   ├── data/                   # Data loading and processing
│   ├── models/                 # Model architectures
│   ├── preprocessing/          # Text preprocessing pipelines
│   ├── training/               # Training loops and optimization
│   ├── evaluation/             # Evaluation metrics and analysis
│   ├── inference/              # Inference API and deployment
│   └── utils/                  # Utilities (config, logging)
├── notebooks/                  # Jupyter notebooks for exploration
├── tests/                      # Unit and integration tests
├── data/
│   ├── raw/                    # Raw datasets
│   ├── processed/              # Processed datasets
│   └── cache/                  # Cached preprocessed data
├── logs/                       # Training and application logs
├── checkpoints/                # Model checkpoints
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
└── README.md                   # This file
```

## 🔧 Configuration

All system parameters are configured via YAML files in `config/`:

### Model Configuration (`config/model_config.yaml`)

```yaml
model:
  base_model: "xlm-roberta-base"
  sentiment:
    num_classes: 3
  sarcasm:
    num_classes: 2
  lid_integration:
    enabled: true
    method: "feature_concat"
```

### Training Configuration (`config/training_config.yaml`)

```yaml
training:
  num_epochs: 10
  batch_size: 16
  learning_rate: 2.0e-5
  optimizer:
    type: "adamw"
    weight_decay: 0.01
```

### Preprocessing Configuration (`config/preprocessing_config.yaml`)

```yaml
preprocessing:
  language_identification:
    enabled: true
    token_level: true
  slang:
    enabled: true
  emoji:
    strategy: "lexicon"
```

## 🎓 Usage

### Training

```python
from src.utils import load_config, setup_logging
from src.training.trainer import Trainer

# Setup
logger = setup_logging()
config = load_config('config/model_config.yaml')

# Train model
trainer = Trainer(config)
trainer.train()
```

### Inference

```python
from src.inference.predictor import SentimentPredictor

predictor = SentimentPredictor('checkpoints/best_model.pt')

text = "yaar this movie was totally bakwas! What a waste of time 🙄"
result = predictor.predict(text)

print(result)
# {
#   'sentiment': {'label': 'negative', 'confidence': 0.92},
#   'sarcasm': {'detected': False, 'confidence': 0.78},
#   'uncertainty': 0.15,
#   'language_mix': ['en', 'hi']
# }
```

## 📊 Development Roadmap

### Phase 1: ✅ Local Environment & Project Setup (Current)
- [x] Virtual environment configuration
- [x] Project structure
- [x] Configuration management
- [x] Logging system

### Phase 2: 🔄 Data Pipeline & Token-Level Language Identification
- [ ] Token-level LID implementation
- [ ] Preprocessing pipeline
- [ ] Data loaders

### Phase 3: 🔄 Multi-Task Sentiment Model Backbone
- [ ] Transformer encoder integration
- [ ] Multi-task head architecture
- [ ] Composite loss implementation

### Phase 4: 🔄 Sarcasm Modeling & Loss Engineering
- [ ] Sarcasm-specific features
- [ ] Dynamic loss balancing
- [ ] Contrastive learning

### Phase 5: 🔄 Evaluation, Robustness & Calibration
- [ ] Comprehensive metrics
- [ ] Temperature scaling
- [ ] Robustness testing

### Phase 6: 🔄 Inference & Interface Layer
- [ ] Inference API
- [ ] CLI tool
- [ ] Web interface (Gradio)

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_preprocessing.py
```

## 📝 Logging

Logs are automatically generated in the `logs/` directory:
- `multilingual_sentiment_YYYYMMDD_HHMMSS.log` - Application logs
- `metrics.log` - Training metrics (CSV format)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face Transformers library
- XLM-RoBERTa pre-trained models
- FastText for language identification
- PyTorch community

## 📧 Contact
Phone: +92 3479555964
Mail: contact@hasnaingul.me
For questions or issues, please open an issue on GitHub or contact the research team.

---

**Status**: Phase 1 Complete ✅  
**Last Updated**: February 2026  
**Version**: 0.1.0