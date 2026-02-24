"""
HuggingFace Hub Upload Script

Uploads trained model to HuggingFace Hub for sharing and deployment.

Prerequisites:
    pip install huggingface_hub
    
Usage:
    python scripts/upload_to_hub.py --checkpoint checkpoints/best_model.pt
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import os

# HuggingFace API token (READ permission)
HF_TOKEN = os.getenv("HF_TOKEN")


def upload_model(
    checkpoint_path: str,
    repo_name: str = "multilingual-sentiment-xlm-roberta",
    private: bool = False,
):
    """
    Upload model to HuggingFace Hub.
    
    Args:
        checkpoint_path: Path to model checkpoint
        repo_name:       Repository name on HF Hub
        private:         Create private repository
    """
    try:
        from huggingface_hub import HfApi, create_repo, upload_file, upload_folder
    except ImportError:
        print("Error: huggingface_hub not installed")
        print("Run: pip install huggingface_hub")
        sys.exit(1)
    
    api = HfApi(token=HF_TOKEN)
    
    # Get username
    user_info = api.whoami(token=HF_TOKEN)
    username = user_info["name"]
    repo_id = f"{username}/{repo_name}"
    
    print(f"\n{'='*60}")
    print(f"Uploading Model to HuggingFace Hub")
    print(f"{'='*60}\n")
    print(f"Repository: {repo_id}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Private: {private}\n")
    
    # Create repository
    try:
        create_repo(
            repo_id=repo_id,
            token=HF_TOKEN,
            private=private,
            repo_type="model",
            exist_ok=True,
        )
        print(f"✓ Repository created/verified: {repo_id}")
    except Exception as e:
        print(f"Error creating repository: {e}")
        sys.exit(1)
    
    # Upload checkpoint
    try:
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            print(f"Error: Checkpoint not found: {checkpoint_path}")
            sys.exit(1)
        
        print(f"\nUploading {checkpoint_file.name}...")
        upload_file(
            path_or_fileobj=str(checkpoint_file),
            path_in_repo=checkpoint_file.name,
            repo_id=repo_id,
            token=HF_TOKEN,
        )
        print(f"✓ Checkpoint uploaded")
    except Exception as e:
        print(f"Error uploading checkpoint: {e}")
        sys.exit(1)
    
    # Upload config files
    config_dir = Path("config")
    if config_dir.exists():
        try:
            print(f"\nUploading config files...")
            for config_file in config_dir.glob("*.yaml"):
                upload_file(
                    path_or_fileobj=str(config_file),
                    path_in_repo=f"config/{config_file.name}",
                    repo_id=repo_id,
                    token=HF_TOKEN,
                )
                print(f"  ✓ {config_file.name}")
        except Exception as e:
            print(f"Warning: Failed to upload configs: {e}")
    
    # Create model card (README.md)
    model_card = f"""---
language:
- en
- hi
- es
- fr
- ar
- pt
- de
- zh
- ja
- ko
tags:
- sentiment-analysis
- sarcasm-detection
- multilingual
- xlm-roberta
- multi-task-learning
license: apache-2.0
---

# Multilingual Sentiment & Sarcasm Detection

Multi-task XLM-RoBERTa model for:
- **Sentiment Analysis** (negative, neutral, positive)
- **Sarcasm Detection** (literal, sarcastic)

Trained on multilingual social media text with code-switching support.

## Model Details

- **Base Model:** xlm-roberta-base
- **Tasks:** Multi-task learning (sentiment + sarcasm)
- **Languages:** 10+ languages (en, hi, es, fr, ar, pt, de, zh, ja, ko)
- **Features:** 
  - Token-level language identification
  - Sarcasm-specific feature extraction
  - Focal loss for class imbalance
  - Temperature-scaled calibration

## Usage

```python
from transformers import AutoTokenizer
import torch

# Load model (replace with actual loading code)
checkpoint = "best_model.pt"
model = torch.load(checkpoint)

# Predict
text = "This is amazing!!!"
# ... (add prediction code)
```

## Training Details

- **Architecture:** XLM-RoBERTa + Multi-task heads
- **Loss:** Composite Focal Loss (sentiment 3-class + sarcasm binary)
- **Optimizer:** AdamW with differential learning rates
- **Scheduler:** Linear warmup + decay

## Performance

Evaluated on multilingual test set with robustness testing.

## Citation

```bibtex
@misc{{multilingual-sentiment-2026,
  author = {{Your Name}},
  title = {{Multilingual Sentiment Analysis with Sarcasm Detection}},
  year = {{2026}},
  publisher = {{HuggingFace}},
  howpublished = {{\\url{{https://huggingface.co/{repo_id}}}}}
}}
```
"""
    
    try:
        readme_path = Path("README_HF.md")
        readme_path.write_text(model_card)
        
        print(f"\nUploading README...")
        upload_file(
            path_or_fileobj="README_HF.md",
            path_in_repo="README.md",
            repo_id=repo_id,
            token=HF_TOKEN,
        )
        print(f"✓ Model card created")
        readme_path.unlink()  # Clean up temp file
    except Exception as e:
        print(f"Warning: Failed to create model card: {e}")
    
    print(f"\n{'='*60}")
    print(f"✓ Upload Complete!")
    print(f"{'='*60}")
    print(f"\nModel URL: https://huggingface.co/{repo_id}")
    print(f"\nYou can now:")
    print(f"  1. View your model on HuggingFace Hub")
    print(f"  2. Share with others")
    print(f"  3. Deploy with Inference API")
    print()


def main():
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--repo-name",
        default="multilingual-sentiment-xlm-roberta",
        help="Repository name on HuggingFace"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private repository"
    )
    
    args = parser.parse_args()
    
    upload_model(
        checkpoint_path=args.checkpoint,
        repo_name=args.repo_name,
        private=args.private,
    )


if __name__ == "__main__":
    main()
