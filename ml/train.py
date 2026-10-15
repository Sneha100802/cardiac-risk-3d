"""Train + cross-validate CAD / LAD / LCX / RCA classifiers.   Run:  python -m ml.train"""
import json
import warnings

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .config import (DISCLAIMER, EXPLAINABLE_MODELS, MODELS_DIR, N_REPEATS, N_SPLITS,
                     SEED, TARGET_CANDIDATES)
from .data import build_features, find_col, load_raw, to_binary

SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "brier": "neg_brier_score",        # sklearn negates it; we flip the sign back in summarise()
}


def _imputer():
    # Median fill, fitted ONLY on the training fold (it sits inside the Pipeline).
    return SimpleImputer(strategy="median").set_output(transform="pandas")


def make_models() -> dict:
    return {
        "logreg": Pipeline([("imp", _imputer()), ("scale", StandardScaler()),
                            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))]),
        "rf": Pipeline([("imp", _imputer()),
                        ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                       class_weight="balanced",
                                                       random_state=SEED, n_jobs=-1))]),
        "xgb": Pipeline([("imp", _imputer()),
                         ("clf", XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                               subsample=0.8, colsample_bytree=0.8,
                                               eval_metric="logloss", random_state=SEED,
                                               n_jobs=1))]),
    }


def summarise(res: dict) -> dict:
    out = {}
    for m in SCORING:
        vals = np.asarray(res[f"test_{m}"], dtype=float)
        if m == "brier":
            vals = -vals
        # mean +/- std over the 15 splits; ci95 is a rough 95% interval of the mean
        out[m] = {"mean": float(vals.mean()), "std": float(vals.std()),
                  "ci95": float(1.96 * vals.std() / np.sqrt(len(vals)))}
    return out


def main():
    warnings.filterwarnings("ignore")
    df = load_raw()
    X, encoders = build_features(df)
    feature_names = list(X.columns)
    print(f"{len(df)} patients, {len(feature_names)} input features (leaky columns removed)")

    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    report, best_per_target = {}, {}

    for target, candidates in TARGET_CANDIDATES.items():
        col = find_col(df, candidates)
        y_all = to_binary(df[col], col)
        keep = y_all.notna()                      # skip rows where this label is missing
        Xt, yt = X[keep], y_all[keep].astype(int)
        print(f"\n=== {target}  (column '{col}')  n={len(yt)}  positives={yt.sum()} ({yt.mean():.0%}) ===")

        report[target] = {"column": col, "n": int(len(yt)), "prevalence": float(yt.mean()), "models": {}}
        for name, model in make_models().items():
            res = cross_validate(clone(model), Xt, yt, cv=cv, scoring=SCORING, n_jobs=1)
            s = report[target]["models"][name] = summarise(res)
            print(f"  {name:7s} AUC {s['roc_auc']['mean']:.3f}±{s['roc_auc']['std']:.3f}  "
                  f"F1 {s['f1']['mean']:.3f}  recall {s['recall']['mean']:.3f}  "
                  f"precision {s['precision']['mean']:.3f}  acc {s['accuracy']['mean']:.3f}")

        # Pick the best *explainable* model by mean CV ROC-AUC.
        eligible = {n: report[target]["models"][n]["roc_auc"]["mean"] for n in EXPLAINABLE_MODELS}
        best = max(eligible, key=eligible.get)
        best_per_target[target] = best
        final = clone(make_models()[best]).fit(Xt, yt)       # final fit on ALL data, for serving
        joblib.dump(final, MODELS_DIR / f"{target}.joblib")
        print(f"  -> serving '{best}' for {target}")

    meta = {"feature_names": feature_names, "encoders": encoders, "best_model": best_per_target,
            "report": report, "cv": {"splits": N_SPLITS, "repeats": N_REPEATS, "seed": SEED},
            "disclaimer": DISCLAIMER}
    (MODELS_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved models + meta.json to {MODELS_DIR}")


if __name__ == "__main__":
    main()
