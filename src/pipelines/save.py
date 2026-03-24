import os
import joblib
import logging

def save_artifacts(model, explainer, feature_names: list, out_dir: str):
    """Save model, explainer, and feature names to the target directory."""
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "model.pkl")
    explainer_path = os.path.join(out_dir, "explainer.pkl")
    features_path = os.path.join(out_dir, "feature_names.pkl")
    
    joblib.dump(model, model_path)
    joblib.dump(explainer, explainer_path)
    joblib.dump(feature_names, features_path)
    
    logging.info(f"Saved model to {model_path}")
    logging.info(f"Saved explainer to {explainer_path}")
    logging.info(f"Saved features to {features_path}")
