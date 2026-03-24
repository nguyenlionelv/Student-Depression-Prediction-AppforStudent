"""
Train Stacking Ensemble model on student depression dataset and save artifacts.
"""
import os
import sys
import argparse
import joblib
import shap
import logging
import warnings
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")

# Ensure we can import preprocessing
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pipelines.preprocessing import run_pipeline

# Configure standard logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def get_args():
    """Parse command line arguments representing hyperparameters and paths."""
    parser = argparse.ArgumentParser(description="Train Stacking Ensemble model for Student Depression Prediction.")
    parser.add_argument("--data-path", type=str, default=None, help="Path to raw CSV data file")
    parser.add_argument("--model-dir", type=str, default=os.path.dirname(__file__), help="Directory to save model artifacts")
    return parser.parse_args()


def prepare_data(df: pd.DataFrame, target: str = "depression"):
    """Split dataframe into features (X) and target (y)."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in data.")
    
    y = df[target].astype(int)
    X = df.drop(columns=[target])
    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])
    return X, y


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> StackingClassifier:
    """Initialize and train Stacking Classifier (RF, XGB, LGBM)."""
    logging.info("Initializing Stacking Ensemble (RF, XGBoost, LightGBM)...")
    
    base_estimators = [
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=6, min_samples_split=10, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=500, max_depth=3, learning_rate=0.05, random_state=42, eval_metric="logloss", verbosity=0)),
        ('lgbm', LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.01, num_leaves=70, min_child_samples=20, random_state=42, verbose=-1))
    ]
    
    model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
        n_jobs=-1
    )
    
    logging.info("Training Stacking Model...")
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: StackingClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """Evaluate model and output classification metrics."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logging.info(f"Test Accuracy: {acc:.4f}")
    logging.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    return acc


def save_artifacts(model, explainer, feature_names: list, out_dir: str):
    """Save model, explainer, and feature names to the standard directory."""
    os.makedirs(out_dir, exist_ok=True)
    
    model_path = os.path.join(out_dir, "model.pkl")
    explainer_path = os.path.join(out_dir, "explainer.pkl")
    features_path = os.path.join(out_dir, "feature_names.pkl")
    
    joblib.dump(model, model_path)
    joblib.dump(explainer, explainer_path)
    joblib.dump(feature_names, features_path)
    
    logging.info(f"✅ Saved model to {model_path}")
    logging.info(f"✅ Saved explainer to {explainer_path}")
    logging.info(f"✅ Saved features to {features_path}")


def main():
    args = get_args()
    
    logging.info("Starting training pipeline...")
    # 1. Load data
    if args.data_path:
        df = run_pipeline(data_path=args.data_path)
    else:
        df = run_pipeline()
    
    # 2. Extract X/y
    X, y = prepare_data(df)
    feature_names = X.columns.tolist()
    logging.info(f"Using {len(feature_names)} features.")
    
    # 3. Data Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Train Model
    model = train_model(X_train, y_train)
    
    # 5. Evaluate Model
    evaluate_model(model, X_test, y_test)
    
    # 6. Interpretability (SHAP Explainer)
    logging.info("Creating SHAP KernelExplainer (with K-Means background)...")
    # Summarize background data for KernelExplainer speed
    background = shap.kmeans(X_train, 50)
    
    # Use KernelExplainer for non-tree meta-estimators
    logging.info("Initializing KernelExplainer with predict_proba...")
    explainer = shap.KernelExplainer(model.predict_proba, background)
    
    # 7. Save outputs
    save_artifacts(model, explainer, feature_names, args.model_dir)
    logging.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
