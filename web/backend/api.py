"""
backend/api.py — FastAPI backend for the NIDS web application.

Backbone: FT-Transformer (Feature-Tokenized Transformer) — exp15
  Each of the 25 UNSW-NB15 features is projected to a 128-dim token; a learnable
  CLS token is prepended; 4 Transformer encoder layers (8 heads) process the
  26-token sequence; the CLS output is the 256-dim classification representation.

Endpoints:
    GET  /                        health check
    GET  /models                  list available models
    POST /predict                 single flow prediction
    POST /predict/batch           batch prediction (up to 1000 flows)
    GET  /features                list expected feature names

Run from project root:
    uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import os

from inference import NIDSPredictor


# ─── APP SETUP ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="NIDS API",
    description=(
        "Network Intrusion Detection using FT-Transformer + Supervised Contrastive Learning "
        "on UNSW-NB15 (10-class, macro F1 0.6504 — exp15)"
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your frontend domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── MODEL REGISTRY ───────────────────────────────────────────────────────────
# Discovers all experiment directories under results/ that contain model.pt +
# preprocessing.pkl.  Falls back to legacy *.pt / *_artifacts.pkl pairs for
# backwards compatibility with older training runs.
#
# Experiment naming convention (from trainer.py --run-name flag):
#   results/exp15/   ← FT-Transformer + SupCon  (default, macro F1 0.6504)
#   results/exp21/   ← FT-Transformer Direct     (macro F1 0.6497)
#   results/exp16/   ← BiLSTM + SupCon           (macro F1 0.6390)
#   results/exp19/   ← BiLSTM Direct             (macro F1 0.6432)
#   ... etc.
MODELS_DIR    = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
PREDICTORS:   Dict[str, NIDSPredictor] = {}
DEFAULT_MODEL = "exp15"


def _load_models():
    if not os.path.isdir(MODELS_DIR):
        print(f"WARNING: results/ directory not found at {MODELS_DIR}. Run trainer.py first.")
        return

    loaded = 0

    # ── Primary: experiment-directory layout (results/expXX/) ─────────────────
    for entry in sorted(os.scandir(MODELS_DIR), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        model_pt  = os.path.join(entry.path, "model.pt")
        preproc   = os.path.join(entry.path, "preprocessing.pkl")
        if not (os.path.exists(model_pt) and os.path.exists(preproc)):
            continue
        key = entry.name  # e.g. "exp15"
        try:
            PREDICTORS[key] = NIDSPredictor(entry.path)
            print(f"Registered model: '{key}'  ({entry.path})")
            loaded += 1
        except Exception as e:
            print(f"Failed to load {entry.path}: {e}")

    # ── Fallback: flat *.pt / *_artifacts.pkl layout (legacy BiLSTM runs) ─────
    if loaded == 0:
        import glob
        pt_files = glob.glob(os.path.join(MODELS_DIR, "*.pt"))
        for pt in pt_files:
            pkl = pt.replace(".pt", "_artifacts.pkl")
            if not os.path.exists(pkl):
                continue
            # Derive a short key: strip common prefixes so "nids_cnn_bilstm_80_20"
            # becomes "80_20", and "nids_fttransformer_exp15" becomes "exp15".
            key = (
                os.path.basename(pt)
                .replace("nids_cnn_bilstm_", "")
                .replace("nids_fttransformer_", "")
                .replace(".pt", "")
            )
            try:
                PREDICTORS[key] = NIDSPredictor(pt, pkl)
                print(f"Registered legacy model: '{key}'")
                loaded += 1
            except Exception as e:
                print(f"Failed to load {pt}: {e}")

    if loaded == 0:
        print("WARNING: No models loaded. Run trainer.py to generate experiment directories.")


_load_models()


# ─── SCHEMAS ──────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    features: Dict[str, Any] = Field(
        ...,
        description="Network flow feature dict. Keys are UNSW-NB15 column names.",
        example={
            "proto":   "tcp",
            "service": "http",
            "state":   "FIN",
            "dur":     0.121,
            "sbytes":  2048,
            "dbytes":  512,
            "sttl":    64,
            "dttl":    64,
            "sloss":   0,
            "dloss":   0,
            "sload":   135266.0,
            "dload":   33816.5,
            "spkts":   6,
            "dpkts":   4,
        }
    )
    model: Optional[str] = Field(
        DEFAULT_MODEL,
        description=(
            "Experiment key matching a results/expXX/ directory. "
            "Examples: 'exp15' (FT-Transformer+SupCon, default), "
            "'exp21' (FT-Transformer Direct), 'exp19' (BiLSTM Direct)."
        )
    )


class BatchPredictRequest(BaseModel):
    flows: List[Dict[str, Any]] = Field(..., description="List of feature dicts")
    model: Optional[str]        = Field(DEFAULT_MODEL)


class PredictionResponse(BaseModel):
    predicted_class:     str
    is_attack:           bool
    confidence:          float
    class_probabilities: Dict[str, float]
    top3:                List[Dict[str, Any]]
    model_used:          str


class BatchPredictionResponse(BaseModel):
    results:      List[PredictionResponse]
    total:        int
    attack_count: int
    model_used:   str


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def _get_predictor(model_key: Optional[str]):
    key = model_key or DEFAULT_MODEL
    if key not in PREDICTORS:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{key}' not found. Available: {list(PREDICTORS.keys())}"
        )
    return PREDICTORS[key], key


# ─── ENDPOINTS ────────────────────────────────────────────────────────────────
@app.get("/")
def health():
    return {
        "status":        "ok",
        "models_loaded": list(PREDICTORS.keys()),
        "default_model": DEFAULT_MODEL,
    }


@app.get("/models")
def list_models():
    return {
        "models": [
            {
                "key":        key,
                "backbone":   getattr(predictor, "backbone_name", "unknown"),
                "classes":    predictor.class_names,
                "n_features": len(predictor.feature_names),
                "is_default": key == DEFAULT_MODEL,
            }
            for key, predictor in PREDICTORS.items()
        ]
    }


@app.get("/features")
def get_features(model: Optional[str] = DEFAULT_MODEL):
    predictor, key = _get_predictor(model)
    return {
        "model":                key,
        "feature_names":        predictor.feature_names,
        "categorical_features": NIDSPredictor.CATEGORICAL_COLS,
        "n_features":           len(predictor.feature_names),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictRequest):
    predictor, key = _get_predictor(req.model)
    try:
        result = predictor.predict(req.features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return PredictionResponse(
        predicted_class     = result.predicted_class,
        is_attack           = result.is_attack,
        confidence          = result.confidence,
        class_probabilities = result.class_probabilities,
        top3                = result.top3,
        model_used          = key,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(req: BatchPredictRequest):
    if not req.flows:
        raise HTTPException(status_code=400, detail="flows list cannot be empty")
    if len(req.flows) > 1000:
        raise HTTPException(status_code=400, detail="Batch size limit is 1000")

    predictor, key = _get_predictor(req.model)
    try:
        results = predictor.predict_batch(req.flows)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {e}")

    responses = [
        PredictionResponse(
            predicted_class     = r.predicted_class,
            is_attack           = r.is_attack,
            confidence          = r.confidence,
            class_probabilities = r.class_probabilities,
            top3                = r.top3,
            model_used          = key,
        )
        for r in results
    ]

    return BatchPredictionResponse(
        results      = responses,
        total        = len(responses),
        attack_count = sum(1 for r in results if r.is_attack),
        model_used   = key,
    )
