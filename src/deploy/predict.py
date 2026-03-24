"""
Preprocessing + prediction + SHAP explanation pipeline.
"""
import os
import numpy as np
import pandas as pd
import joblib

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH     = os.path.join(BASE_DIR, "model", "model.pkl")
EXPLAINER_PATH = os.path.join(BASE_DIR, "model", "explainer.pkl")
FEATURES_PATH  = os.path.join(BASE_DIR, "model", "feature_names.pkl")

# ─── Lazy Load Artifacts ──────────────────────────────────────────────────────
_model = None
_explainer = None
_feature_names = None

def load_models():
    """Load models lazily to prevent import errors if they don't exist yet."""
    global _model, _explainer, _feature_names
    if _model is None:
        _model         = joblib.load(MODEL_PATH)
        _explainer     = joblib.load(EXPLAINER_PATH)
        _feature_names = joblib.load(FEATURES_PATH)
    return _model, _explainer, _feature_names

# ─── Encoding maps (must match train_and_save.py) ─────────────────────────────
SLEEP_MAP = {
    "Less than 5 hours": 0,
    "5-6 hours":         1,
    "7-8 hours":         2,
    "More than 8 hours": 3,
}

DEGREE_MAP = {
    "HSC": 0, "Class 12": 0,
    "BSc": 1, "B.Com": 1, "B.Ed": 1, "B.Pharm": 1,
    "B.Tech": 1, "BBA": 1, "BCA": 1, "BE": 1,
    "MSc": 2, "M.Com": 2, "M.Ed": 2, "M.Pharm": 2,
    "M.Tech": 2, "MBA": 2, "MCA": 2, "ME": 2, "MHM": 2, "LLM": 2,
    "PhD": 3,
    "Others": 1,
}

def preprocess(raw: dict, feature_names: list) -> tuple[pd.DataFrame, dict]:
    """
    Convert raw API input to a feature DataFrame matching training.
    """
    enc = {}
    enc["gender"]             = 1 if raw.get("gender", "Female").lower() in ("male", "nam") else 0
    enc["age"]                = float(raw.get("age", 22))
    enc["academic_pressure"]  = float(raw.get("academic_pressure", 3))
    enc["work_pressure"]      = float(raw.get("work_pressure", 1))
    enc["cgpa"]               = float(raw.get("cgpa", 7.0))
    enc["study_satisfaction"] = float(raw.get("study_satisfaction", 3))
    enc["job_satisfaction"]   = float(raw.get("job_satisfaction", 1))
    enc["sleep_duration"]     = SLEEP_MAP.get(raw.get("sleep_duration", "7-8 hours"), 2)
    enc["degree"]             = DEGREE_MAP.get(raw.get("degree", "B.Tech"), 1)
    enc["suicidal_thoughts"]  = 1 if raw.get("suicidal_thoughts", "No").lower() in ("yes", "có") else 0
    enc["work_study_hours"]   = float(raw.get("work_study_hours", 5))
    enc["financial_stress"]   = float(raw.get("financial_stress", 2))
    enc["family_history"]     = 1 if raw.get("family_history", "No").lower() in ("yes", "có") else 0

    diet = raw.get("dietary_habits", "Moderate")
    enc["dietary_habits_Healthy"]   = 1 if diet == "Healthy"   else 0
    enc["dietary_habits_Moderate"]  = 1 if diet == "Moderate"  else 0
    enc["dietary_habits_Others"]    = 1 if diet == "Others"    else 0
    enc["dietary_habits_Unhealthy"] = 1 if diet == "Unhealthy" else 0

    enc["stress_score"]             = enc["academic_pressure"] + enc["work_pressure"] + enc["financial_stress"]
    enc["satisfaction_score"]       = enc["study_satisfaction"] + enc["job_satisfaction"]
    enc["pressure_satisfaction_gap"]= enc["academic_pressure"] - enc["study_satisfaction"]
    enc["work_hours_per_sleep"]     = enc["work_study_hours"] / (enc["sleep_duration"] + 1)
    enc["cgpa_pressure_ratio"]      = enc["cgpa"] / (enc["academic_pressure"] + 1)

    row = {feat: enc.get(feat, 0) for feat in feature_names}
    X_df = pd.DataFrame([row], columns=feature_names)
    return X_df, enc

def predict_with_explanation(raw: dict) -> dict:
    """Full pipeline: preprocess → predict → SHAP → explanation."""
    from src.deploy.logic_translator import (
        translate_factors, build_explanation, get_risk_level
    )
    
    model, explainer, feature_names = load_models()
    X_df, enc = preprocess(raw, feature_names)

    prob      = float(model.predict_proba(X_df)[0, 1])
    pred      = int(prob >= 0.5)
    risk_lvl  = get_risk_level(prob)

    shap_vals_matrix = explainer.shap_values(X_df)
    
    # KernelExplainer on predict_proba might return a list of arrays or a 3D array (n_samples, n_features, n_classes)
    if isinstance(shap_vals_matrix, list):
        shap_vals = shap_vals_matrix[1][0] if shap_vals_matrix[1].ndim == 2 else shap_vals_matrix[1]
    elif getattr(shap_vals_matrix, "ndim", 0) == 3:
        shap_vals = shap_vals_matrix[0, :, 1]
    else:
        # Fallback for TreeExplainer
        shap_vals = shap_vals_matrix[0] if getattr(shap_vals_matrix, "ndim", 0) == 2 else shap_vals_matrix

    top_factors = translate_factors(shap_vals, feature_names, top_k=5)
    explanation, recommendations = build_explanation(top_factors, risk_lvl, prob, enc)

    return {
        "prediction":      pred,
        "probability":     round(prob, 4),
        "risk_level":      risk_lvl,
        "top_factors":     [f.model_dump() for f in top_factors],
        "explanation":     explanation,
        "recommendations": recommendations,
    }
