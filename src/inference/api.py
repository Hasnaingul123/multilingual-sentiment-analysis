"""
FastAPI Server for Multilingual Sentiment Analysis

Production REST API with /predict, /predict_batch, /health endpoints.

Usage:
    uvicorn src.inference.api:app --host 0.0.0.0 --port 8000
"""

from pathlib import Path
from typing import List, Optional
from datetime import datetime
import time

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from fastapi import FastAPI, HTTPException, status
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field, validator
except ImportError:
    raise ImportError("Run: pip install fastapi uvicorn pydantic")

from utils.logger import get_logger

logger = get_logger("api")

_pipeline = None

# Request/Response models
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    return_probabilities: bool = True
    
    @validator("text")
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty")
        return v

class PredictResponse(BaseModel):
    text: str
    sentiment: str
    sentiment_confidence: float
    sarcasm: str
    sarcasm_confidence: float
    sentiment_probs: Optional[dict] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str

app = FastAPI(
    title="Multilingual Sentiment Analysis API",
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def load_model():
    global _pipeline
    try:
        from inference.pipeline import InferencePipeline
        logger.info("Loading model...")
        _pipeline = InferencePipeline.from_checkpoint(
            checkpoint_path="checkpoints/best_model.pt",
            config_dir="config",
        )
        logger.info("✓ Model loaded")
    except FileNotFoundError:
        logger.warning("Checkpoint not found — server will start but predictions will fail")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

@app.get("/")
async def root():
    return {
        "service": "Multilingual Sentiment Analysis API",
        "status": "operational" if _pipeline else "model_not_loaded",
        "endpoints": ["/predict", "/health", "/docs"],
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    return {
        "status": "healthy" if _pipeline else "model_not_loaded",
        "model_loaded": _pipeline is not None,
        "timestamp": datetime.now().isoformat(),
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if _pipeline is None:
        raise HTTPException(503, "Model not loaded")
    
    try:
        result = _pipeline.predict_single(
            text=request.text,
            return_probabilities=request.return_probabilities,
        )
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(500, f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("Multilingual Sentiment Analysis API")
    print("="*60)
    print("\nDocs: http://localhost:8000/docs\n")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
