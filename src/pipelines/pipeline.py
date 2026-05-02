import os, sys
import argparse
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
import mlflow
# import warnings
import shap
from sklearn.model_selection import train_test_split

# warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.pipelines.preprocessing import run_pipeline
from src.pipelines.train import prepare_data, train_model
from src.pipelines.evaluate import evaluate_model, evaluate_model_kfold
from src.pipelines.save import save_artifacts

def get_args():
    parser = argparse.ArgumentParser(description="Student Depression Model Training Pipeline.")
    parser.add_argument("--data-path", type=str, default=None, help="Path to raw CSV data file")
    default_model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "model")
    parser.add_argument("--model-dir", type=str, default=default_model_dir, help="Directory to save model artifacts")
    return parser.parse_args()

def main():
    args = get_args()
    logging.info("Starting training pipeline...")
    
    df = run_pipeline(data_path=args.data_path) if args.data_path else run_pipeline()
    X, y = prepare_data(df)
    feature_names = X.columns.tolist()
    logging.info(f"Using {len(feature_names)} features.")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    exp = mlflow.set_experiment(experiment_name="Stacking Ensemble")
    with mlflow.start_run(experiment_id=exp.experiment_id,run_name="Stacking Ensemble"):

        model = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)
        acc, f1, rec, prec = evaluate_model_kfold(model, X, y)

        mlflow.log_metric("acc", acc)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("rec", rec)
        mlflow.log_metric("prec", prec)
    
    logging.info("Creating SHAP KernelExplainer with K-Means background")
    background = shap.kmeans(X_train, 50)
    explainer = shap.KernelExplainer(model.predict_proba, background)
    
    save_artifacts(model, explainer, feature_names, args.model_dir)
    logging.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
