import logging
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import StackingClassifier

def evaluate_model(model: StackingClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """Evaluate model and output classification metrics."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logging.info(f"Test Accuracy: {acc:.4f}")
    logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    return acc
