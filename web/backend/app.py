"""
app.py -- FastAPI backend for the NIDS dashboard.

Backbone: FT-Transformer + Supervised Contrastive Learning (exp15, macro F1 0.6504)
  Each of 25 UNSW-NB15 flow features is projected to a 128-dim token; 4 Transformer
  encoder layers with 8 attention heads discover cross-feature interactions via
  permutation-invariant self-attention; the CLS token output is the 256-dim
  representation fed into the CB Focal Loss classification head.

Run from the PROJECT ROOT (NIDS 2.0/), not from web/backend/:
    uvicorn web.backend.app:app --reload --host 0.0.0.0 --port 8000

Requires admin/root for live packet capture (scapy raw sockets).
"""

import asyncio
import io
import json
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from inference import NIDSPredictor
from web.backend.capture import CaptureManager, EventLog
from web.backend.config import DEFAULT_EXP, EXP_DIR, RESULTS_DIR

# Rows per inference pass -- keeps GPU memory under 4 GB
CHUNK_SIZE = 512

SEVERITY = {
    "Normal":         "safe",
    "Analysis":       "low",
    "Reconnaissance": "low",
    "Fuzzers":        "medium",
    "Generic":        "medium",
    "Backdoor":       "high",
    "DoS":            "high",
    "Exploits":       "high",
    "Shellcode":      "critical",
    "Worms":          "critical",
}

# In-memory history store (capped at 500 analyses)
_history: deque = deque(maxlen=500)
_history_counter = 0


def _next_id() -> int:
    global _history_counter
    _history_counter += 1
    return _history_counter


def _format_predictions(raw, filename: str) -> dict:
    """
    Normalise NIDSPredictor output into the shape the frontend expects.

    NIDSPredictor (FT-Transformer) returns a list of dicts per row.
    Accepted key variants for the predicted class label:
        "predicted_class"  (new inference.py convention)
        "class"            (old convention)
        "label"            (capture path)
        "prediction"       (legacy api.py convention)
    Accepted key variants for per-class probabilities:
        "class_probabilities"  (new)
        "probabilities"        (old)
    """
    if isinstance(raw, dict):
        raw = [raw]

    results = []
    for row in raw:
        # ── Resolve predicted class label ─────────────────────────────────────
        label = (
            row.get("predicted_class")
            or row.get("class")
            or row.get("label")
            or row.get("prediction")
            or "Unknown"
        )

        # ── Resolve confidence ────────────────────────────────────────────────
        confidence = float(row.get("confidence", 0.0))

        # ── Resolve per-class probability dict ───────────────────────────────
        probs = row.get("class_probabilities") or row.get("probabilities") or {}

        is_mal = label != "Normal"
        results.append({
            "prediction":          label,
            "confidence":          confidence,
            "severity":            SEVERITY.get(label, "medium"),
            "is_malicious":        is_mal,
            "class_probabilities": probs,
        })

    malicious = sum(1 for r in results if r["is_malicious"])
    return {
        "filename": filename,
        "summary": {
            "total":     len(results),
            "malicious": malicious,
            "benign":    len(results) - malicious,
        },
        "results": results,
    }


# Singletons
predictor:   Optional[NIDSPredictor]   = None
capture_mgr: Optional[CaptureManager] = None
event_log:   EventLog                 = EventLog()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, capture_mgr
    if not EXP_DIR.exists():
        raise RuntimeError(
            f"Experiment directory not found: {EXP_DIR}. "
            f"Run training with --run-name {DEFAULT_EXP} or update DEFAULT_EXP in config.py."
        )
    predictor = NIDSPredictor(str(EXP_DIR))
    backbone  = getattr(predictor, "backbone_name", "fttransformer")
    print(f"[NIDS] Loaded predictor from {EXP_DIR}  (backbone: {backbone})")
    capture_mgr = CaptureManager(predictor, event_log)
    yield
    if capture_mgr and capture_mgr.running:
        capture_mgr.stop()


app = FastAPI(title="NIDS API — FT-Transformer", version="2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Batch prediction

@app.post("/api/predict/batch")
async def batch_predict(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")
    raw_bytes = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    # Process in chunks to avoid CUDA OOM on large CSVs (e.g. full test set)
    raw_preds = []
    for start in range(0, len(df), CHUNK_SIZE):
        chunk = df.iloc[start: start + CHUNK_SIZE]
        raw_preds.extend(predictor.predict(chunk))

    payload = _format_predictions(raw_preds, file.filename)

    _history.appendleft({
        "id":        _next_id(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "filename":  file.filename,
        "total":     payload["summary"]["total"],
        "malicious": payload["summary"]["malicious"],
        "benign":    payload["summary"]["benign"],
        "sample":    payload["results"][:10],
    })
    return payload


# History

@app.get("/api/history")
async def get_history():
    return {"history": list(_history)}


# Live capture

@app.post("/api/monitor/start")
async def start_monitor(interface: str = Query(default=None)):
    if capture_mgr is None:
        raise HTTPException(status_code=503, detail="Capture manager not initialised.")
    if capture_mgr.running:
        return {"status": "already_running"}
    capture_mgr.start(iface=interface or None)
    return {"status": "started", "interface": interface}


@app.post("/api/monitor/stop")
async def stop_monitor():
    if capture_mgr:
        capture_mgr.stop()
    return {"status": "stopped"}


@app.get("/api/monitor/status")
async def monitor_status():
    return {"running": bool(capture_mgr and capture_mgr.running)}


@app.get("/api/monitor/stream")
async def monitor_stream():
    """SSE stream -- each client has its own cursor, no missed events."""
    async def generator():
        cursor = 0
        try:
            while True:
                events, total = event_log.since(cursor)
                for item in events:
                    yield f"data: {json.dumps(item)}\n\n"
                cursor = total
                await asyncio.sleep(0.25)
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Model / experiment stats

@app.get("/api/experiments")
async def list_experiments():
    if not RESULTS_DIR.exists():
        return {"experiments": []}
    exps = sorted(d.name for d in RESULTS_DIR.iterdir() if d.is_dir())
    return {"experiments": exps}


@app.get("/api/model/stats")
async def model_stats(exp: str = DEFAULT_EXP):
    exp_dir = RESULTS_DIR / exp
    for fname in ("classification_report.json", "metrics.json"):
        path = exp_dir / fname
        if path.exists():
            with open(path) as f:
                return json.load(f)
    raise HTTPException(
        status_code=404,
        detail=f"No metrics file in results/{exp}/. Save classification_report.json from training.",
    )


# Serve frontend (mount last so API routes are not shadowed)
_FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
if _FRONTEND.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND), html=True), name="static")
    