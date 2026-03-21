import pandas as pd
import numpy as np
import os, sys

DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "student_depression_dataset.csv")

def load_data():
    # ─── 1. Load data ─────────────────────────────────────────────────────────────
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    return df

def rename_columns(df):
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

def drop_columns(df):
    # ─── 3. Drop unused columns ───────────────────────────────────────────────────
    drop_cols = [c for c in ["id", "city", "profession"] if c in df.columns]
    df = df.drop(columns=drop_cols)

def encode_categorical_features(df):
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

def coerce_numeric_features(df):
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

def handle_missing_values(df):
    # ─── 5c. Handle missing values (after coerce) ────────────────────────────────
    all_num = df.select_dtypes(include=[np.number]).columns.tolist()
    df[all_num] = df[all_num].fillna(df[all_num].median())
    print(f"  Shape after cleaning: {df.shape}")

def feature_engineering(df):
    # ─── 6. Feature Engineering ───────────────────────────────────────────────────
    df["stress_score"]             = df["academic_pressure"] + df["work_pressure"] + df["financial_stress"]
    df["satisfaction_score"]       = df["study_satisfaction"] + df["job_satisfaction"]
    df["pressure_satisfaction_gap"]= df["academic_pressure"] - df["study_satisfaction"]
    df["work_hours_per_sleep"]     = df["work_study_hours"] / (df["sleep_duration"] + 1)
    df["cgpa_pressure_ratio"]      = df["cgpa"] / (df["academic_pressure"] + 1)

def preprocessing():
    df = load_data()
    rename_columns(df)
    drop_columns(df)
    encode_categorical_features(df)
    coerce_numeric_features(df)
    handle_missing_values(df)
    feature_engineering(df)
    return df