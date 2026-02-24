#!/usr/bin/env python3
"""
CLI Tool for Multilingual Sentiment Analysis

Usage:
    python -m src.inference.cli "This is amazing!!!"
    python -m src.inference.cli --file input.txt --output results.json
    python -m src.inference.cli --interactive
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def predict_single(pipeline, text: str, verbose: bool = False):
    """Predict and print for single text."""
    result = pipeline.predict_single(text, return_probabilities=verbose)
    
    print(f"\nText: {text}")
    print(f"Sentiment: {result['sentiment']} ({result['sentiment_confidence']:.2%})")
    print(f"Sarcasm: {result['sarcasm']} ({result['sarcasm_confidence']:.2%})")
    
    if verbose and "sentiment_probs" in result:
        print("\nSentiment probs:")
        for label, prob in result["sentiment_probs"].items():
            print(f"  {label:8s}: {prob:.4f}")


def predict_batch(pipeline, texts, output_file=None):
    """Batch prediction."""
    print(f"Processing {len(texts)} texts...")
    results = pipeline.predict_batch(texts, batch_size=32)
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Saved to {output_file}")
    else:
        for i, r in enumerate(results, 1):
            print(f"[{i}] {r['sentiment']} | {r['sarcasm']}")


def interactive_mode(pipeline):
    """Interactive REPL."""
    print("\n" + "="*60)
    print("Interactive Sentiment Analysis")
    print("="*60)
    print("Enter text (or 'quit' to exit)\n")
    
    while True:
        try:
            text = input(">>> ").strip()
            if not text or text.lower() in ("quit", "exit", "q"):
                break
            predict_single(pipeline, text, verbose=True)
        except KeyboardInterrupt:
            break


def main():
    parser = argparse.ArgumentParser(description="Sentiment Analysis CLI")
    parser.add_argument("text", nargs="?", help="Text to analyze")
    parser.add_argument("--file", "-f", help="Input file (one text per line)")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--verbose", "-v", action="store_true")
    
    args = parser.parse_args()
    
    if not any([args.text, args.file, args.interactive]):
        parser.error("Provide text, --file, or --interactive")
    
    try:
        from inference.pipeline import InferencePipeline
        print(f"Loading model from {args.checkpoint}...")
        pipeline = InferencePipeline.from_checkpoint(
            args.checkpoint, args.config_dir
        )
        print("✓ Loaded\n")
    except FileNotFoundError:
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    if args.interactive:
        interactive_mode(pipeline)
    elif args.file:
        with open(args.file) as f:
            texts = [line.strip() for line in f if line.strip()]
        predict_batch(pipeline, texts, args.output)
    else:
        predict_single(pipeline, args.text, args.verbose)


if __name__ == "__main__":
    main()
