import logging
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

def prepare_data(df: pd.DataFrame, target: str = "depression"):
    """Split dataframe into features (X) and target (y)."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in data.")
    y = df[target].astype(int)
    X = df.drop(columns=[target]).select_dtypes(include=[np.number])
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
