import os
import pandas as pd

def test_processed_data_exists():
    # Processed path points to data/processed
    processed_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "student_depression_dataset_processed.csv")
    assert os.path.exists(processed_path), "Processed data should exist for tests."
    df = pd.read_csv(processed_path)
    assert not df.empty
    assert "depression" in df.columns

def test_model_artifact_exists():
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "model.pkl")
    assert os.path.exists(model_path), "Model artifact should exist. Run `make train` first."

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_predict_endpoint_success():
    payload = {
        "gender": "Female",
        "age": 21,
        "academic_pressure": 4,
        "work_pressure": 2,
        "cgpa": 7.5,
        "study_satisfaction": 2,
        "job_satisfaction": 1,
        "sleep_duration": "5-6 hours",
        "dietary_habits": "Unhealthy",
        "degree": "B.Tech",
        "suicidal_thoughts": "No",
        "work_study_hours": 8,
        "financial_stress": 4,
        "family_history": "No"
    }
    response = client.post("/predict", json=payload)
    if response.status_code == 503:
        pytest.skip("Models absent, skipping execution endpoint test.")
    
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "risk_level" in data
    assert "explanation" in data
    assert "recommendations" in data

