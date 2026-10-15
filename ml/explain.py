"""Inference + SHAP explanations used by the API."""
import json
from functools import lru_cache

import joblib
import numpy as np
import shap

from .config import MODELS_DIR
from .data import encode_record

TARGETS = ["CAD", "LAD", "LCX", "RCA"]


@lru_cache(maxsize=1)
def load_meta() -> dict:
    return json.loads((MODELS_DIR / "meta.json").read_text())


@lru_cache(maxsize=None)
def _model(target: str):
    return joblib.load(MODELS_DIR / f"{target}.joblib")


@lru_cache(maxsize=None)
def _explainer(target: str):
    return shap.TreeExplainer(_model(target)[-1])          # [-1] = the tree model inside the Pipeline


def thresholds() -> dict:
    """Per-target decision thresholds chosen by train_tuned.py (0.5 if you only ran train.py)."""
    return load_meta().get("thresholds") or {t: 0.5 for t in TARGETS}


def predict_all(record: dict) -> dict:
    meta = load_meta()
    X = encode_record(record, meta["encoders"], meta["feature_names"])
    return {t: float(_model(t).predict_proba(X)[0, 1]) for t in TARGETS}


def explain_one(record: dict, target: str, top_k: int = 10) -> dict:
    meta = load_meta()
    raw = encode_record(record, meta["encoders"], meta["feature_names"])
    pipe = _model(target)
    Xi = pipe[0].transform(raw)                            # same imputation the model saw

    sv = _explainer(target).shap_values(Xi)
    # Different shap versions return different shapes. Normalise to 1-D (n_features,).
    if isinstance(sv, list):
        sv = sv[1]                                         # old API: [class0, class1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[..., 1]                                    # new API: (rows, features, classes)
    sv = sv[0]

    base = float(np.ravel(_explainer(target).expected_value)[-1])
    best = meta["best_model"][target]
    names = meta["feature_names"]
    items = [{"feature": n,
              "value": None if np.isnan(Xi.iloc[0][n]) else float(Xi.iloc[0][n]),
              "shap": float(sv[i]),
              "was_missing": bool(np.isnan(raw.iloc[0][n]))}   # True => value was imputed
             for i, n in enumerate(names)]
    items.sort(key=lambda d: abs(d["shap"]), reverse=True)

    return {"target": target, "model": best,
            # Random forest SHAP is in probability units, XGBoost in log-odds. Show it in the UI.
            "units": "probability" if best == "rf" else "log-odds",
            "base_value": base, "top_features": items[:top_k]}
