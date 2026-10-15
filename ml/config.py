"""Single place for every setting. Change things here, not inside the other files."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Put the Z-Alizadeh Sani *extension* file here (.csv or .xlsx), then edit the name if needed.
DATA_PATH = ROOT / "data" / "raw" / "z_alizadeh_sani_extension.xlsx"
MODELS_DIR = ROOT / "models"

# The brief says: exclude LAD, LCX, RCA and Cath from the INPUT features.
# Matching is case-insensitive, so "lad" / "LAD" are both dropped.
LEAKY_COLS = ["LAD", "LCX", "RCA", "Cath"]

# target name -> candidate column names in the file (first match wins, case-insensitive).
# If your file has a dedicated overall-CAD column, add its name at the front of the "CAD" list.
TARGET_CANDIDATES = {
    "CAD": ["CAD", "Cath"],
    "LAD": ["LAD"],
    "LCX": ["LCX"],
    "RCA": ["RCA"],
}

# Cross-validation: 5 folds x 3 repeats = 15 train/test splits per model per target.
N_SPLITS = 5
N_REPEATS = 3
SEED = 42

# Only these models can be served, because SHAP TreeExplainer needs tree models.
# Logistic regression is trained as a baseline for the report, but never served.
EXPLAINABLE_MODELS = {"rf", "xgb"}

DISCLAIMER = (
    "Decision-support / educational use only. Not a substitute for formal "
    "diagnostic imaging or clinical judgement."
)
