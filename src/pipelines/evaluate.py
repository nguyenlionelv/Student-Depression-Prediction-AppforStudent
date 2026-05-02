import logging
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_score

def evaluate_model(model: StackingClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """Evaluate model and output classification metrics."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logging.info(f"Test Accuracy: {acc:.4f}")
    logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    return acc

def evaluate_model_kfold(model: StackingClassifier, X: pd.DataFrame, y: pd.Series, n_splits: int = 5, shuffle: bool = True, random_state: int = 42) -> float:
    acc = cross_val_score(model, X, y, cv=n_splits, scoring="accuracy", n_jobs=-1)
    f1 = cross_val_score(model, X, y, cv=n_splits, scoring="f1", n_jobs=-1)
    rec = cross_val_score(model, X, y, cv=n_splits, scoring="recall", n_jobs=-1)
    prec = cross_val_score(model, X, y, cv=n_splits, scoring="precision", n_jobs=-1)

    logging.info("Result K-fold:")
    logging.info(f"Accuracy: {acc.mean():.4f} +/- {acc.std():.4f}")
    logging.info(f"Precision: {prec.mean():.4f} +/- {prec.std():.4f}")
    logging.info(f"Recall: {rec.mean():.4f} +/- {rec.std():.4f}")
    logging.info(f"F1-score: {f1.mean():.4f} +/- {f1.std():.4f}")
    return acc.mean(), f1.mean(), rec.mean(), prec.mean()
