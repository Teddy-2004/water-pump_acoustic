"""
api/main.py
-----------
FastAPI service for the acoustic pump diagnostics model.

Endpoints:
    GET  /health           -- liveness check
    GET  /uptime            -- how long this instance has been running
    GET  /metrics            -- last known evaluation metrics + class distribution (for the UI dashboard)
    POST /predict            -- upload ONE .wav file -> prediction
    POST /retrain             -- upload a ZIP of labeled .wav files -> trigger retraining
    GET  /model-info          -- current model version, training timestamp

Run locally:
    uvicorn api.main:app --reload --port 8000

Run in Docker: see api/Dockerfile
"""

import os
import sys
import time
import json
import shutil
import zipfile
import tempfile
from datetime import datetime, timezone

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.prediction import predict_file
from src.model import load_model, save_model
from api.retrain import run_retraining, RETRAIN_STATUS

APP_START_TIME = time.time()
MODEL_PATH = os.environ.get("MODEL_PATH", "models/pump_cnn_v1.h5")
METRICS_PATH = os.environ.get("METRICS_PATH", "models/latest_metrics.json")

app = FastAPI(title="Acoustic Pump Diagnostics API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None  # lazy-loaded, reloaded after retraining


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=503, detail=f"Model not found at {MODEL_PATH}")
        _model = load_model(MODEL_PATH)
    return _model


def reload_model():
    global _model
    _model = load_model(MODEL_PATH)


# ------------------------------------------------------------------
# Health / monitoring
# ------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/uptime")
def uptime():
    seconds = time.time() - APP_START_TIME
    return {
        "uptime_seconds": round(seconds, 1),
        "started_at": datetime.fromtimestamp(APP_START_TIME, tz=timezone.utc).isoformat(),
        "now": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/model-info")
def model_info():
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=503, detail="Model not found")
    mtime = os.path.getmtime(MODEL_PATH)
    return {
        "model_path": MODEL_PATH,
        "last_trained": datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat(),
        "retrain_status": RETRAIN_STATUS,
    }


@app.get("/metrics")
def metrics():
    if not os.path.exists(METRICS_PATH):
        return {"detail": "No evaluation metrics saved yet. Run the notebook or a retrain first."}
    with open(METRICS_PATH) as f:
        return json.load(f)


# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported.")

    model = get_model()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = predict_file(model, tmp_path)
    finally:
        os.remove(tmp_path)

    return result


# ------------------------------------------------------------------
# Retraining
# ------------------------------------------------------------------

class RetrainResponse(BaseModel):
    status: str
    message: str


@app.post("/retrain", response_model=RetrainResponse)
async def retrain(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Accepts a ZIP file with this structure:
        normal/*.wav
        abnormal/*.wav

    Bulk-uploaded clips are saved under data/incoming/, preprocessed, and used to
    fine-tune the currently deployed model in a background task. Poll /model-info
    for retrain_status while it runs.
    """
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Upload a .zip file containing normal/ and abnormal/ folders.")

    if RETRAIN_STATUS["state"] == "running":
        raise HTTPException(status_code=409, detail="A retraining job is already running.")

    upload_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    incoming_dir = os.path.join("data", "incoming", upload_id)
    os.makedirs(incoming_dir, exist_ok=True)

    zip_path = os.path.join(incoming_dir, "upload.zip")
    with open(zip_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(incoming_dir)
    os.remove(zip_path)

    if not (os.path.isdir(os.path.join(incoming_dir, "normal")) or
            os.path.isdir(os.path.join(incoming_dir, "abnormal"))):
        shutil.rmtree(incoming_dir, ignore_errors=True)
        raise HTTPException(
            status_code=400,
            detail="Zip must contain normal/ and/or abnormal/ folders of .wav files.",
        )

    RETRAIN_STATUS.update({"state": "running", "started_at": datetime.now(timezone.utc).isoformat()})
    background_tasks.add_task(run_retraining, incoming_dir, MODEL_PATH, METRICS_PATH, reload_model)

    return RetrainResponse(status="accepted", message=f"Retraining started on upload {upload_id}. Poll /model-info for progress.")
