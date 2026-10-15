"""FastAPI service.   Run from repo root:  uvicorn backend.app:app --reload --port 8000"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ml.config import DISCLAIMER
import numpy as np

from ml.data import load_raw
from ml.explain import TARGETS, explain_one, load_meta, predict_all, thresholds

app = FastAPI(title="Cardiac Risk API", version="0.1.0")
app.add_middleware(CORSMiddleware,
                   allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])


class Patient(BaseModel):
    # {"features": {"Age": 58, "Sex": "Male", ...}}. Missing keys are fine (they get imputed).
    features: dict[str, float | str | None] = {}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/schema")
def schema():
    """Lets the frontend build the input form automatically from the trained feature list."""
    meta = load_meta()
    enc = meta["encoders"]
    return {"features": [
        {"name": n, "type": "categorical", "options": list(enc[n].keys())} if n in enc
        else {"name": n, "type": "numeric"}
        for n in meta["feature_names"]]}


def _record(row, feature_names):
    """One dataset row -> {feature: value}, skipping missing cells and the label columns."""
    out = {}
    for n in feature_names:
        v = row.get(n)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        out[n] = v.item() if hasattr(v, "item") else v
    return out


@app.get("/examples")
def examples():
    """Three real patients from the training file, for demos. They were part of training, so
    their predictions are optimistic; the UI says so."""
    df, meta = load_raw(), load_meta()
    feats = meta["feature_names"]
    stenotic = lambda c: df[c].astype(str).str.strip().str.lower() == "stenotic"
    groups = [
        ("Recorded: all three arteries narrowed", stenotic("LAD") & stenotic("LCX") & stenotic("RCA")),
        ("Recorded: only LAD narrowed", stenotic("LAD") & ~stenotic("LCX") & ~stenotic("RCA")),
        ("Recorded: normal angiography", df["Cath"].astype(str).str.strip().str.lower() == "normal"),
    ]
    out = []
    for label, mask in groups:
        idx = df.index[mask]
        if len(idx) == 0:
            continue
        row = df.loc[idx[len(idx) // 2]]                        # a middle row, not the first
        out.append({"label": label,
                    "record": _record(row, feats),
                    "truth": {t: str(row[t]) for t in ("LAD", "LCX", "RCA")}})
    return {"examples": out,
            "note": "Sample patients from the training data; the model has already seen them."}


@app.get("/metrics")
def metrics():
    meta = load_meta()
    return {"report": meta["report"], "best_model": meta["best_model"], "cv": meta["cv"]}


@app.post("/predict")
def predict(p: Patient):
    probs = predict_all(p.features)
    thr = thresholds()
    return {"probabilities": probs,
            "thresholds": thr,                                         # colour midpoint for the 3D view
            "flagged": {t: probs[t] >= thr[t] for t in TARGETS},       # True = above decision threshold
            "disclaimer": DISCLAIMER}


@app.post("/explain/{target}")
def explain(target: str, p: Patient):
    target = target.upper()
    if target not in TARGETS:
        raise HTTPException(404, f"target must be one of {TARGETS}")
    return {**explain_one(p.features, target), "disclaimer": DISCLAIMER}
