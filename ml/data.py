"""Loading, target binarising and (leak-free) feature encoding."""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_PATH, LEAKY_COLS

POSITIVE = {"1", "1.0", "cad", "stenotic", "stenosis", "y", "yes", "true", "abnormal"}
NEGATIVE = {"0", "0.0", "normal", "n", "no", "false"}


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. See README step 1.")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        # The UCI workbook has 2 sheets and the 2nd is an EMPTY template, so take the sheet with most rows.
        sheets = pd.read_excel(path, sheet_name=None)
        df = max(sheets.values(), key=len)
    else:
        df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]          # remove stray spaces in headers
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]      # drop index columns from exports
    return df


def find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    lookup = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lookup:
            return lookup[cand.lower()]
    raise KeyError(f"None of {candidates} found. Columns are: {list(df.columns)}")


def to_binary(series: pd.Series, name: str) -> pd.Series:
    """Turn a label column into 1 (disease / stenosis) or 0 (normal). NaN stays NaN."""
    def conv(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip().lower()
        if s in POSITIVE:
            return 1
        if s in NEGATIVE:
            return 0
        # Fail loudly instead of silently mislabelling patients.
        raise ValueError(
            f"Cannot binarise column {name!r}: unexpected value {v!r}. "
            f"Unique values: {sorted(map(str, series.dropna().unique()))}. "
            "Add the right words to POSITIVE/NEGATIVE in ml/data.py."
        )
    return series.map(conv)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return (numeric X without leaky columns, encoders for text columns).

    Encoding is a fixed lookup (text -> integer code). It uses no target information,
    so doing it once on the whole table cannot leak labels into the test folds.
    Anything that *learns from data* (imputer, scaler, model) lives inside the sklearn
    Pipeline in train.py, so it is re-fit inside every CV fold.
    """
    leaky = {c.lower() for c in LEAKY_COLS}
    X = df[[c for c in df.columns if c.lower() not in leaky]].copy()

    encoders: dict[str, dict[str, int]] = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            continue
        as_num = pd.to_numeric(X[col], errors="coerce")
        if as_num.notna().sum() == X[col].notna().sum():       # numbers that were stored as text
            X[col] = as_num
            continue
        clean = X[col].where(X[col].isna(), X[col].astype(str).str.strip())
        values = sorted(clean.dropna().unique())
        encoders[col] = {v: i for i, v in enumerate(values)}   # e.g. {"Fmale": 0, "Male": 1}
        X[col] = clean.map(encoders[col])

    # Hard guard: crash if a leaky column ever sneaks back in.
    assert not ({c.lower() for c in X.columns} & leaky), "Leaky column found in features!"
    return X.astype(float), encoders


def encode_record(record: dict, encoders: dict, feature_names: list[str]) -> pd.DataFrame:
    """Turn one patient (dict from the API) into a 1-row frame with training columns/order."""
    row = {}
    for col in feature_names:
        v = record.get(col)
        if v is None or v == "" or (isinstance(v, float) and np.isnan(v)):
            row[col] = np.nan                                   # imputed later by the pipeline
        elif col in encoders:
            row[col] = encoders[col].get(str(v).strip(), np.nan)
        else:
            try:
                row[col] = float(v)
            except (TypeError, ValueError):
                row[col] = np.nan
    return pd.DataFrame([row], columns=feature_names)
