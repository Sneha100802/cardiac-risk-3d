# Cardiac Risk 3D (Multimodal AI Hackathon 2026, Track A)

Predicts overall CAD and LAD / LCX / RCA stenosis from clinical data and colours the three coronary
arteries on an interactive 3D heart, with SHAP explanations.
**Decision support / education only. Not a substitute for diagnostic imaging.**

## Setup and run
```bash
# 1. Data: save the UCI "Extension of Z-Alizadeh Sani" workbook as
#    data/raw/z_alizadeh_sani_extension.xlsx

# 2. Python environment
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Train (pick ONE)
python -m ml.train            # fast baseline, about 1-2 min
python -m ml.train_tuned      # tuned + thresholds + calibration check, about 10-15 min (recommended)

# 4. API   (terminal 1, from the repo root)
uvicorn backend.app:app --reload --port 8000

# 5. Frontend   (terminal 2)
cd frontend && npm install && npm run dev               # open http://localhost:5173
```

## Method
- Inputs: demographic, exam, ECG, lab and echo features (55). `LAD`, `LCX`, `RCA`, `Cath` are removed
  from the inputs (target leakage); the code asserts this.
- Targets: overall CAD (`Cath`), and stenosis of LAD, LCX and RCA.
- Validation: repeated stratified CV. `train_tuned.py` uses **nested** CV: hyper-parameters and the
  decision threshold are chosen only on each outer fold's training part, then scored on the untouched
  test part.
- Metrics: accuracy, precision, recall, F1, ROC-AUC, Brier score.
- Explanations: SHAP TreeExplainer per prediction.
- 3D: a procedural (stylised) heart with the three arteries snapped onto its surface, so each model
  output maps to exactly one artery. Colour midpoint (yellow) = that vessel's decision threshold.

## Limitations (edit with your own numbers)
- 303 patients from a single centre; no external validation.
- LCX and RCA are hard from these features (ROC-AUC about 0.72); thresholds trade precision for recall.
- The 3D heart is stylised, not patient-specific anatomy; arteries are schematic.
- Not a diagnostic tool.

## Credits
- Dataset: Alizadehsani, Roshanzamir & Sani, *Extension of Z-Alizadeh Sani dataset*, UCI ML Repository,
  DOI 10.24432/C5461K (CC BY 4.0).
- 3D heart: generated from an implicit surface equation in `frontend/src/components/heartMath.js`
  (no third-party mesh used).
