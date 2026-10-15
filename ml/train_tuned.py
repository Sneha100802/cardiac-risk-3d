"""Leak-free tuning, decision thresholds and calibration check.

Run:  python -m ml.train_tuned        (about 10-20 minutes; writes models/*.joblib + meta.json)

Why "nested"? If you tune hyper-parameters or the threshold on the same folds you report
results on, the numbers are inflated. Here, for every OUTER fold:
  1. grid-search hyper-parameters using only that fold's TRAINING part (inner CV),
  2. pick the decision threshold from out-of-fold predictions on the TRAINING part only,
  3. score the untouched OUTER test part.
The reported numbers therefore describe the whole procedure on unseen patients.
"""
import json
import time
import warnings

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict)

from .config import DISCLAIMER, EXPLAINABLE_MODELS, MODELS_DIR, SEED, TARGET_CANDIDATES
from .data import build_features, find_col, load_raw, to_binary
from .train import make_models

GRIDS = {
    "rf": {"clf__max_features": ["sqrt", 0.5], "clf__min_samples_leaf": [1, 3, 5]},
    "xgb": {"clf__max_depth": [2, 3, 4], "clf__learning_rate": [0.03, 0.08]},
}
OUTER = dict(n_splits=5, n_repeats=2)   # 10 outer test folds per model per target


def youden_threshold(y, p) -> float:
    """Threshold that maximises sensitivity + specificity - 1 (balanced; raises recall when
    the positive class is the minority, as for LCX / RCA)."""
    fpr, tpr, thr = roc_curve(y, p)
    k = int(np.argmax(tpr - fpr))
    return float(np.clip(thr[k], 0.05, 0.95))


def cls_metrics(y, pred) -> dict:
    return {"accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred),
            "f1": f1_score(y, pred)}


def inner_cv(k):
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)


def evaluate_nested(name, Xt, yt) -> dict:
    outer = RepeatedStratifiedKFold(random_state=SEED, **OUTER)
    rows = []
    for tr, te in outer.split(Xt, yt):
        Xtr, Xte, ytr, yte = Xt.iloc[tr], Xt.iloc[te], yt.iloc[tr], yt.iloc[te]
        gs = GridSearchCV(make_models()[name], GRIDS[name], cv=inner_cv(3),
                          scoring="roc_auc", n_jobs=-1).fit(Xtr, ytr)
        best = gs.best_estimator_
        oof = cross_val_predict(clone(best), Xtr, ytr, cv=inner_cv(5), method="predict_proba")[:, 1]
        thr = youden_threshold(ytr, oof)                       # chosen WITHOUT looking at Xte
        p = best.predict_proba(Xte)[:, 1]
        p_cal = CalibratedClassifierCV(clone(best), method="sigmoid", cv=3).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        row = {"auc": roc_auc_score(yte, p), "brier": brier_score_loss(yte, p),
               "brier_calibrated": brier_score_loss(yte, p_cal), "threshold": thr}
        row.update({f"{k}@0.5": v for k, v in cls_metrics(yte, p >= 0.5).items()})
        row.update({f"{k}@thr": v for k, v in cls_metrics(yte, p >= thr).items()})
        rows.append(row)
    return {k: {"mean": float(np.mean([r[k] for r in rows])), "std": float(np.std([r[k] for r in rows]))}
            for k in rows[0]}


def main():
    warnings.filterwarnings("ignore")
    df = load_raw()
    X, encoders = build_features(df)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    report, best_model, thresholds, best_params = {}, {}, {}, {}

    for target, candidates in TARGET_CANDIDATES.items():
        t0 = time.time()
        col = find_col(df, candidates)
        y_all = to_binary(df[col], col)
        keep = y_all.notna()
        Xt, yt = X[keep], y_all[keep].astype(int)
        print(f"\n=== {target} ('{col}')  n={len(yt)}  prevalence={yt.mean():.0%} ===", flush=True)

        report[target] = {"column": col, "n": int(len(yt)), "prevalence": float(yt.mean()), "models": {}}
        for name in sorted(EXPLAINABLE_MODELS):
            r = report[target]["models"][name] = evaluate_nested(name, Xt, yt)
            print(f"  {name:4s} AUC {r['auc']['mean']:.3f}±{r['auc']['std']:.3f} | "
                  f"@0.5  recall {r['recall@0.5']['mean']:.3f} prec {r['precision@0.5']['mean']:.3f} F1 {r['f1@0.5']['mean']:.3f} | "
                  f"@thr  recall {r['recall@thr']['mean']:.3f} prec {r['precision@thr']['mean']:.3f} F1 {r['f1@thr']['mean']:.3f} "
                  f"(thr~{r['threshold']['mean']:.2f}) | Brier {r['brier']['mean']:.3f} -> calibrated {r['brier_calibrated']['mean']:.3f}",
                  flush=True)

        # Choose the model family with the best nested AUC, then fit it on ALL data for serving.
        best = max(EXPLAINABLE_MODELS, key=lambda n: report[target]["models"][n]["auc"]["mean"])
        gs = GridSearchCV(make_models()[best], GRIDS[best], cv=inner_cv(5), scoring="roc_auc", n_jobs=-1).fit(Xt, yt)
        oof = cross_val_predict(clone(gs.best_estimator_), Xt, yt, cv=inner_cv(5), method="predict_proba")[:, 1]
        thresholds[target] = youden_threshold(yt, oof)
        best_model[target] = best
        best_params[target] = {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in gs.best_params_.items()}
        joblib.dump(gs.best_estimator_, MODELS_DIR / f"{target}.joblib")
        print(f"  -> serving '{best}' {best_params[target]}  threshold={thresholds[target]:.2f}  ({time.time()-t0:.0f}s)", flush=True)

    meta = {"feature_names": list(X.columns), "encoders": encoders, "best_model": best_model,
            "thresholds": thresholds, "best_params": best_params, "report": report, "tuned": True,
            "cv": {"outer": OUTER, "seed": SEED}, "disclaimer": DISCLAIMER}
    (MODELS_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved to {MODELS_DIR}", flush=True)


if __name__ == "__main__":
    main()
