import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DEFAULT_RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "student_depression_dataset.csv")
DEFAULT_PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "student_depression_dataset_processed.csv")

def load_data(path: str) -> pd.DataFrame:
    """Load the raw dataset from a CSV file."""
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    return df

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and rename columns to standardize mapping."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df.rename(columns={'have_you_ever_had_suicidal_thoughts_?':'suicidal_thoughts'}, inplace=True)
    return df

def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are not used in modeling."""
    df = df.copy()
    drop_cols = [c for c in ["id", "city", "profession"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    return df

# def transform_columns(df: pd.DataFrame) -> pd.DataFrame:
#     df['dietary_habits'] = df['dietary_habits'].replace('Others',df['dietary_habits'].mode()[0])

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode string categoricals into ordinal/binary integers and dummy variables."""
    df = df.copy()
    
    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Male": 1, "Female": 0, "male": 1, "female": 0}).fillna(0).astype(int)

    if "sleep_duration" in df.columns:
        sleep_map = {
            "Less than 5 hours": 0,
            "5-6 hours": 1,
            "7-8 hours": 2,
            "More than 8 hours": 3,
            "Others": 1,
        }
        df["sleep_duration"] = df["sleep_duration"].map(sleep_map).fillna(1).astype(int)

    if "degree" in df.columns:
        degree_map = {
            "HSC": 0, "Class 12": 0,
            "BSc": 1, "B.Com": 1, "B.Ed": 1, "B.Pharm": 1, "B.Tech": 1, "BBA": 1, "BCA": 1, "BE": 1,
            "MSc": 2, "M.Com": 2, "M.Ed": 2, "M.Pharm": 2, "M.Tech": 2, "MBA": 2, "MCA": 2, "ME": 2, "MHM": 2, "LLM": 2,
            "PhD": 3,
            "Others": 1,
        }
        df["degree"] = df["degree"].map(degree_map).fillna(1).astype(int)

    if "suicidal_thoughts" in df.columns:
        df["suicidal_thoughts"] = df["suicidal_thoughts"].map({"Yes": 1, "No": 0, "yes": 1, "no": 0}).fillna(0).astype(int)

    if "family_history" in df.columns:
        df["family_history"] = df["family_history"].map({"Yes": 1, "No": 0, "yes": 1, "no": 0}).fillna(0).astype(int)

    if "dietary_habits" in df.columns:
        diet_dummies = pd.get_dummies(df["dietary_habits"], prefix="dietary_habits")
        df = pd.concat([df.drop(columns=["dietary_habits"]), diet_dummies], axis=1)

    for col in ["dietary_habits_Healthy", "dietary_habits_Moderate", "dietary_habits_Others", "dietary_habits_Unhealthy"]:
        if col not in df.columns:
            df[col] = 0

    return df

def coerce_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_expected = [
        "age", "academic_pressure", "work_pressure", "cgpa",
        "study_satisfaction", "job_satisfaction", "work_study_hours", "financial_stress",
    ]
    for col in numeric_expected:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "depression" in df.columns:
        df["depression"] = pd.to_numeric(df["depression"], errors="coerce")
        df = df.dropna(subset=["depression"])
        df["depression"] = df["depression"].astype(int)
    
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute NaNs with column medians for continuous features."""
    df = df.copy()
    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    if nums:
        df[nums] = df[nums].fillna(df[nums].median())
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create aggregated scores and ratios domain-relevant attributes."""
    df = df.copy()
    if all(c in df.columns for c in ["academic_pressure", "work_pressure", "financial_stress"]):
        df["stress_score"] = df["academic_pressure"] + df["work_pressure"] + df["financial_stress"]
        
    if all(c in df.columns for c in ["study_satisfaction", "job_satisfaction"]):
        df["satisfaction_score"] = df["study_satisfaction"] + df["job_satisfaction"]
        
    if all(c in df.columns for c in ["academic_pressure", "study_satisfaction"]):
        df["pressure_satisfaction_gap"] = df["academic_pressure"] - df["study_satisfaction"]
        
    if all(c in df.columns for c in ["work_study_hours", "sleep_duration"]):
        df["work_hours_per_sleep"] = df["work_study_hours"] / (df["sleep_duration"] + 1)
        
    if all(c in df.columns for c in ["cgpa", "academic_pressure"]):
        df["cgpa_pressure_ratio"] = df["cgpa"] / (df["academic_pressure"] + 1)
        
    return df

def run_pipeline(data_path: str = DEFAULT_RAW_DATA_PATH, processed_path: str = DEFAULT_PROCESSED_DATA_PATH, force_reprocess: bool = False) -> pd.DataFrame:
    """Run all preprocessing steps and return cleaned dataframe."""
    if not force_reprocess and os.path.exists(processed_path):
        print(f"Loading already processed data from {processed_path}...")
        return pd.read_csv(processed_path)
        
    df = load_data(data_path)
    df = rename_columns(df)
    df = drop_columns(df)
    df = encode_categorical_features(df)
    df = coerce_numeric_features(df)
    df = handle_missing_values(df)
    df = feature_engineering(df)
    
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"Saved processed data to {processed_path}")
    
    return df

# For backward compatibility if any script relies on preprocessing.preprocessing()
def preprocessing(data_path: str = DEFAULT_RAW_DATA_PATH) -> pd.DataFrame:
    """Legacy entrypoint mapping to run_pipeline()."""
    return run_pipeline(data_path)