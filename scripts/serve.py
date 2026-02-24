"""
FastAPI Inference Server
========================
REST API for the Multilingual Sentiment Analysis model.
Wraps the offline InferencePipeline with a production-ready FastAPI app.

Usage:
    # From project root:
    python scripts/serve.py
    # Or with uvicorn directly:
    python -m uvicorn scripts.serve:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /health            — Health check + model info
    POST /predict           — Single text prediction
    POST /predict_batch     — Batch prediction
    GET  /docs              — Swagger UI (auto-generated)
    GET  /redoc             — ReDoc UI (auto-generated)

Requirements:
    pip install fastapi uvicorn[standard]
"""

import sys
import time
import json
import csv
import random
from pathlib import Path
from typing import List, Optional
from datetime import datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    torch = None

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    FASTAPI_OK = True
except ImportError:
    print("ERROR: FastAPI not installed. Run: pip install fastapi uvicorn[standard]")
    sys.exit(1)

from utils.logger import get_logger

logger = get_logger("serve")


# ─────────────────────────────────────────────────────────────────────────────
# Offline model (same BiLSTM as training — no HuggingFace required)
# ─────────────────────────────────────────────────────────────────────────────

class SimpleTokenizer:
    def __init__(self, vocab=None, max_vocab=20000):
        self.vocab = vocab or {"<PAD>": 0, "<UNK>": 1}
        self.max_vocab = max_vocab

    def build_vocab(self, texts):
        from collections import Counter
        counts = Counter()
        for t in texts:
            counts.update(t.lower().split())
        for word, _ in counts.most_common(self.max_vocab - 2):
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)

    def encode(self, text, max_length=128):
        tokens = text.lower().split()[:max_length]
        ids  = [self.vocab.get(w, 1) for w in tokens]
        ids  += [0] * (max_length - len(ids))
        mask = [1 if i > 0 else 0 for i in ids]
        return ids, mask


class OfflineSentimentModel(nn.Module if TORCH_OK else object):
    """
    Matches exactly the architecture saved in checkpoints/best_model.pt.
    Layer names: embed, lid_embed, lstm, sentiment_head, sarcasm_head
    """
    def __init__(self, vocab_size=30004, embed_dim=128, hidden_dim=256,
                 num_lid_langs=10, num_sentiment_classes=3):
        super().__init__()
        self.embed     = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lid_embed = torch.nn.Embedding(num_lid_langs + 1, 16, padding_idx=0)
        self.lstm      = torch.nn.LSTM(
            embed_dim + 16, hidden_dim, num_layers=2,
            batch_first=True, dropout=0.0,  # dropout off in eval mode
            bidirectional=True
        )
        self.dropout   = torch.nn.Dropout(0.0)  # off in eval mode
        feat_dim       = hidden_dim * 2  # 512
        self.sentiment_head = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, 128), torch.nn.Tanh(),
            torch.nn.Dropout(0.0), torch.nn.Linear(128, num_sentiment_classes),
        )
        self.sarcasm_head = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, 64), torch.nn.ReLU(),
            torch.nn.Dropout(0.0), torch.nn.Linear(64, 1),
        )

    def forward(self, input_ids, attention_mask=None, lid_token_ids=None):
        if lid_token_ids is None:
            lid_token_ids = torch.zeros_like(input_ids)
        lid_token_ids = lid_token_ids.clamp(0, 10)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        x = torch.cat([self.embed(input_ids), self.lid_embed(lid_token_ids)], dim=-1)
        out, _ = self.lstm(x)
        mask   = attention_mask.unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        pooled = self.dropout(pooled)
        return {
            "sentiment_logits": self.sentiment_head(pooled),
            "sarcasm_logit":    self.sarcasm_head(pooled),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Model Manager (singleton loaded once at startup)
# ─────────────────────────────────────────────────────────────────────────────

class ModelManager:
    """Manages model loading, tokenizer, and inference."""

    SENTIMENT_LABELS = ["negative", "neutral", "positive"]
    SARCASM_LABELS   = ["literal", "sarcastic"]

    def __init__(self):
        self.model      : Optional[OfflineSentimentModel] = None
        self.tokenizer  : Optional[SimpleTokenizer]      = None
        self.device     : Optional[str]                  = None
        self.loaded_at  : Optional[str]                  = None
        self.checkpoint : Optional[str]                  = None
        self.prev_f1    : float = 0.0
        self.sarc_threshold: float = 0.5  # default; can be updated

    def load(self, checkpoint_path: str, csv_path: str = None):
        """Load model + tokenizer. Builds vocab from CSV if provided."""
        if not TORCH_OK:
            raise RuntimeError("PyTorch not installed")

        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

        # Build tokenizer vocab
        self.tokenizer = SimpleTokenizer()
        if csv_path and Path(csv_path).exists():
            texts = []
            with open(csv_path, encoding="utf-8", errors="replace") as f:
                for row in csv.DictReader(f):
                    if row.get("text", "").strip():
                        texts.append(row["text"])
            self.tokenizer.build_vocab(texts)
            logger.info(f"Vocab built from CSV: {len(self.tokenizer.vocab):,} tokens")
        else:
            logger.warning("No CSV provided — using empty vocab (UNK for all tokens)")

        # Build model + load weights
        saved = torch.load(ckpt, map_location="cpu", weights_only=False)
        saved_vocab = saved.get("vocab_size", 30004)
        self.model = OfflineSentimentModel(
            vocab_size=saved_vocab, embed_dim=128, hidden_dim=256,
            num_lid_langs=10, num_sentiment_classes=3,
        )
        self.model.load_state_dict(saved["model_state"])
        self.model.eval()
        self.prev_f1    = saved.get("combined_f1", 0.0)
        self.checkpoint = str(ckpt)

        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model  = self.model.to(self.device)
        self.loaded_at = datetime.now().isoformat()

        logger.info(
            f"Model loaded | device={self.device} | "
            f"checkpoint F1={self.prev_f1:.4f} | vocab={len(self.tokenizer.vocab):,}"
        )

    @torch.no_grad()
    def predict_one(self, text: str) -> dict:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")

        ids, mask = self.tokenizer.encode(text, max_length=128)
        input_ids      = torch.tensor([ids],  dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([mask], dtype=torch.long).to(self.device)
        lid_token_ids  = torch.zeros(1, 128, dtype=torch.long).to(self.device)

        out = self.model(input_ids, attention_mask, lid_token_ids)

        sent_probs  = torch.softmax(out["sentiment_logits"], dim=-1).cpu().numpy()[0]
        sarc_prob   = torch.sigmoid(out["sarcasm_logit"].squeeze()).cpu().item()

        sent_idx   = int(sent_probs.argmax())
        sarc_pred  = sarc_prob >= self.sarc_threshold

        return {
            "sentiment":            self.SENTIMENT_LABELS[sent_idx],
            "sentiment_confidence": float(sent_probs[sent_idx]),
            "sentiment_probs": {
                "negative": float(sent_probs[0]),
                "neutral":  float(sent_probs[1]),
                "positive": float(sent_probs[2]),
            },
            "sarcasm":           self.SARCASM_LABELS[1] if sarc_pred else self.SARCASM_LABELS[0],
            "sarcasm_prob":      float(sarc_prob),
            "sarcasm_threshold": float(self.sarc_threshold),
        }


# Global model manager (singleton)
manager = ModelManager()


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Multilingual Sentiment Analysis API",
    description=(
        "REST API for joint multilingual sentiment classification "
        "(negative/neutral/positive) and sarcasm detection. "
        "Supports English, Urdu, Roman Urdu, and code-switching text."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        examples=["This product is absolutely amazing! Totally worth every penny."],
        description="Input text for sentiment and sarcasm analysis",
    )

class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_items=100,
        examples=[["Great product!", "Yeah right, like that ever works 🙄"]],
        description="List of texts (max 100 per request)",
    )

class SentimentProbs(BaseModel):
    negative: float
    neutral:  float
    positive: float

class PredictResponse(BaseModel):
    text:                   str
    sentiment:              str
    sentiment_confidence:   float
    sentiment_probs:        SentimentProbs
    sarcasm:                str
    sarcasm_prob:           float
    sarcasm_threshold:      float
    latency_ms:             float

class BatchPredictResponse(BaseModel):
    results:    List[PredictResponse]
    count:      int
    latency_ms: float


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """Load model at server startup."""
    ckpt = ROOT / "checkpoints" / "best_model.pt"
    csv  = ROOT / "data" / "raw" / "combined_full.csv"
    if not csv.exists():
        csv = ROOT / "data" / "raw" / "sarcasm_news.csv"
    try:
        manager.load(str(ckpt), str(csv) if csv.exists() else None)
        logger.info("Server ready ✓")
    except Exception as e:
        logger.error(f"Failed to load model at startup: {e}")
        # Server still starts; /predict will return 503


@app.get("/health", summary="Health check", tags=["System"])
async def health():
    """Returns server + model status."""
    ok = manager.model is not None
    return {
        "status":       "ok"     if ok else "degraded",
        "model_loaded": ok,
        "checkpoint":   manager.checkpoint,
        "checkpoint_f1": manager.prev_f1,
        "device":       manager.device,
        "loaded_at":    manager.loaded_at,
        "sarcasm_threshold": manager.sarc_threshold,
        "server_time":  datetime.now().isoformat(),
    }


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Single text prediction",
    tags=["Inference"],
)
async def predict(request: PredictRequest):
    """
    Predict sentiment (negative / neutral / positive) and
    sarcasm (literal / sarcastic) for a single text.
    """
    if manager.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded — check server logs")

    t0 = time.perf_counter()
    try:
        result = manager.predict_one(request.text)
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    latency = (time.perf_counter() - t0) * 1000

    return PredictResponse(
        text=request.text,
        latency_ms=round(latency, 2),
        **result,
    )


@app.post(
    "/predict_batch",
    response_model=BatchPredictResponse,
    summary="Batch text prediction",
    tags=["Inference"],
)
async def predict_batch(request: BatchPredictRequest):
    """
    Predict sentiment and sarcasm for a list of texts (up to 100).
    Processes each text sequentially and returns results in order.
    """
    if manager.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded — check server logs")

    if len(request.texts) > 100:
        raise HTTPException(status_code=422, detail="Maximum 100 texts per batch request")

    t0 = time.perf_counter()
    results = []
    for text in request.texts:
        if not text.strip():
            raise HTTPException(status_code=422, detail=f"Empty text found in batch")
        t_start = time.perf_counter()
        result  = manager.predict_one(text)
        t_ms    = (time.perf_counter() - t_start) * 1000
        results.append(PredictResponse(text=text, latency_ms=round(t_ms, 2), **result))

    total_latency = (time.perf_counter() - t0) * 1000
    return BatchPredictResponse(
        results=results,
        count=len(results),
        latency_ms=round(total_latency, 2),
    )


@app.get("/", include_in_schema=False)
async def root():
    return JSONResponse({
        "service": "Multilingual Sentiment Analysis API",
        "docs":    "/docs",
        "health":  "/health",
    })


# ─────────────────────────────────────────────────────────────────────────────
# Dev server entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        import uvicorn
    except ImportError:
        print("ERROR: uvicorn not installed. Run: pip install uvicorn[standard]")
        sys.exit(1)

    import argparse
    p = argparse.ArgumentParser(description="Start the sentiment analysis API server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = p.parse_args()

    print(f"\n🚀 Starting Multilingual Sentiment Analysis API")
    print(f"   Host: {args.host}:{args.port}")
    print(f"   Docs: http://localhost:{args.port}/docs")
    print(f"   Health: http://localhost:{args.port}/health\n")

    uvicorn.run(
        "serve:app" if not __name__ == "__main__" else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
