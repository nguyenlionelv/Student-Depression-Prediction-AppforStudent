"""
Train XGBoost model on student depression dataset and save:
  - model/model.pkl     : trained XGBClassifier
  - model/explainer.pkl : SHAP TreeExplainer
  - model/feature_names.pkl : ordered feature names
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import joblib
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "student_depression_dataset.csv")
MODEL_DIR  = os.path.dirname(__file__)
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
EXPLAINER_PATH = os.path.join(MODEL_DIR, "explainer.pkl")
FEATURES_PATH  = os.path.join(MODEL_DIR, "feature_names.pkl")

# ─── 1. Load data ─────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")
print(f"  Columns: {df.columns.tolist()}")

# ─── 2. Rename columns (normalize) ───────────────────────────────────────────
df.columns = df.columns.str.strip()
rename_map = {
    "Gender": "gender",
    "Age": "age",
    "Academic Pressure": "academic_pressure",
    "Work Pressure": "work_pressure",
    "CGPA": "cgpa",
    "Study Satisfaction": "study_satisfaction",
    "Job Satisfaction": "job_satisfaction",
    "Sleep Duration": "sleep_duration",
    "Dietary Habits": "dietary_habits",
    "Degree": "degree",
    "Have you ever had suicidal thoughts ?": "suicidal_thoughts",
    "Work/Study Hours": "work_study_hours",
    "Financial Stress": "financial_stress",
    "Family History of Mental Illness": "family_history",
    "Depression": "depression",
    "City": "city",
    "Profession": "profession",
    "id": "id",
}
df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

# ─── 3. Drop unused columns ───────────────────────────────────────────────────
drop_cols = [c for c in ["id", "city", "profession"] if c in df.columns]
df = df.drop(columns=drop_cols)

# ─── 4. Encode categorical features ───────────────────────────────────────────
# Gender
df["gender"] = df["gender"].map({"Male": 1, "Female": 0, "male": 1, "female": 0}).fillna(0).astype(int)

# Sleep Duration → ordinal
sleep_map = {
    "Less than 5 hours": 0,
    "5-6 hours": 1,
    "7-8 hours": 2,
    "More than 8 hours": 3,
    "Others": 1,          # treat as mid
}
df["sleep_duration"] = df["sleep_duration"].map(sleep_map).fillna(1).astype(int)

# Degree → ordinal
degree_map = {
    "HSC": 0,
    "BSc": 1, "B.Com": 1, "B.Ed": 1, "B.Pharm": 1, "B.Tech": 1, "BBA": 1,
    "BCA": 1, "BE": 1,
    "Class 12": 0,
    "MSc": 2, "M.Com": 2, "M.Ed": 2, "M.Pharm": 2, "M.Tech": 2, "MBA": 2,
    "MCA": 2, "ME": 2, "MHM": 2,
    "LLM": 2,
    "PhD": 3,
    "Others": 1,
}
df["degree"] = df["degree"].map(degree_map).fillna(1).astype(int)

# Suicidal thoughts
df["suicidal_thoughts"] = df["suicidal_thoughts"].map(
    {"Yes": 1, "No": 0, "yes": 1, "no": 0}
).fillna(0).astype(int)

# Family history
df["family_history"] = df["family_history"].map(
    {"Yes": 1, "No": 0, "yes": 1, "no": 0}
).fillna(0).astype(int)

# Dietary habits → one-hot
diet_dummies = pd.get_dummies(df["dietary_habits"], prefix="dietary_habits")
df = pd.concat([df.drop(columns=["dietary_habits"]), diet_dummies], axis=1)

# Ensure standard dietary columns exist
for col in ["dietary_habits_Healthy", "dietary_habits_Moderate",
            "dietary_habits_Others", "dietary_habits_Unhealthy"]:
    if col not in df.columns:
        df[col] = 0

# ─── 5b. Coerce any remaining columns that should be numeric ─────────────────
# Some columns like financial_stress, academic_pressure may have '?' or string values
NUMERIC_COLS_EXPECTED = [
    "age", "academic_pressure", "work_pressure", "cgpa",
    "study_satisfaction", "job_satisfaction", "work_study_hours", "financial_stress",
]
for col in NUMERIC_COLS_EXPECTED:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")  # '?' → NaN

# Target column
df["depression"] = pd.to_numeric(df["depression"], errors="coerce")
df = df.dropna(subset=["depression"])

# ─── 5c. Handle missing values (after coerce) ────────────────────────────────
all_num = df.select_dtypes(include=[np.number]).columns.tolist()
df[all_num] = df[all_num].fillna(df[all_num].median())
print(f"  Shape after cleaning: {df.shape}")

# ─── 6. Feature Engineering ───────────────────────────────────────────────────

df["stress_score"]             = df["academic_pressure"] + df["work_pressure"] + df["financial_stress"]
df["satisfaction_score"]       = df["study_satisfaction"] + df["job_satisfaction"]
df["pressure_satisfaction_gap"]= df["academic_pressure"] - df["study_satisfaction"]
df["work_hours_per_sleep"]     = df["work_study_hours"] / (df["sleep_duration"] + 1)
df["cgpa_pressure_ratio"]      = df["cgpa"] / (df["academic_pressure"] + 1)

# ─── 7. Prepare X, y ─────────────────────────────────────────────────────────
target = "depression"
X = df.drop(columns=[target])
y = df[target].astype(int)

# Keep only numeric columns
X = X.select_dtypes(include=[np.number])
feature_names = X.columns.tolist()
print(f"\nFeatures ({len(feature_names)}): {feature_names}")
print(f"Target distribution:\n{y.value_counts().to_dict()}")

# ─── 8. Train/test split ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─── 9. Train XGBoost ─────────────────────────────────────────────────────────
print("\nTraining XGBoost...")
model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
    verbosity=0,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred))

# ─── 10. SHAP TreeExplainer ───────────────────────────────────────────────────
print("Creating SHAP TreeExplainer...")
explainer = shap.TreeExplainer(model)

# Quick sanity check on first 5 samples
shap_vals = explainer.shap_values(X_test.iloc[:5])
print(f"  SHAP values shape: {shap_vals.shape}")

# ─── 11. Save artifacts ───────────────────────────────────────────────────────
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model,        MODEL_PATH)
joblib.dump(explainer,    EXPLAINER_PATH)
joblib.dump(feature_names, FEATURES_PATH)

print(f"\n✅ Saved model      → {MODEL_PATH}")
print(f"✅ Saved explainer  → {EXPLAINER_PATH}")
print(f"✅ Saved features   → {FEATURES_PATH}")
print("\nDone!")
