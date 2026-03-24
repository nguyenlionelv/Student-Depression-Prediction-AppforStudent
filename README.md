# Student Depression Prediction App

## Overview
The **Student Depression Prediction App** is an end-to-end Machine Learning project designed to predict the risk of depression among university students based on academic, psychological, and demographic factors. Built with **FastAPI** and a powerful **Stacking Ensemble**, this application provides a robust and scalable backend for real-time inference, coupled with **SHAP** (SHapley Additive exPlanations) for model interpretability.

## Key Features
- **Predictive Analytics**: High-accuracy Stacking Classifier combining Random Forest, XGBoost, and LightGBM for superior predictive performance.
- **Model Explainability**: Integration with SHAP to translate black-box model predictions into actionable, human-readable insights for each individual prediction.
- **RESTful API**: Fast and fully documented API via FastAPI and Swagger UI.
- **Modular Data Pipeline**: Highly reusable Pandas-based preprocessing steps (cleaning, imputation, standardization, feature engineering).
- **Extensible Architecture**: Clean code following modern Python standards (`argparse`, structured logging, type checking).

## Architecture & Tech Stack
- **Backend Framework**: FastAPI, Pydantic, Uvicorn (ASGI)
- **Machine Learning**: Scikit-Learn, XGBoost, LightGBM, RF, SHAP
- **Data Engineering**: Pandas, NumPy, Matplotlib
- **Persistence**: Joblib

## Directory Structure
```text
.
├── api                         # FastAPI application entry point (main.py)
├── data                        # Datasets (CSV files)
│   ├── raw                     # Raw source files
│   └── processed               # Auto-generated clean files for modeling
├── model                       # Serialized Stacking Classifier and SHAP explainer
├── src                         # Core logic and inference
│   ├── pipelines                   # Modular ML pipeline (train, evaluate, save, pipeline)
│   └── deploy                  # Inference code (predict, schemas, logic_translator)
├── tests                       # Unit testing suite
├── static                      # Static Web User Interface (HTML/CSS/JS)
├── Dockerfile                  # Production-ready API deployment descriptor
├── Makefile                    # Developer tool commands
├── docker-compose.yml          # Docker Compose configuration
└── requirements.txt            # Python dependencies
```

## Setup & Installation

### Prerequisites
- Python 3.9+

### 1. Clone the Repository
```bash
git clone https://github.com/nguyenlionelv/Student-Depression-Predictation.git
```

### 2. Install Dependencies
```bash
python -m pip install -r requirements.txt
```
or
```bash
make install
```

### 3. Model Training Pipeline
Before starting the API server, generate the serialized machine learning artifacts. The preprocessing pipeline automatically parses raw data, handles missing values, engineers new features, and trains the multi-algorithm Stacking Classifier.
```bash
make train
```
Upon successful completion, model artifacts (`model.pkl`, `explainer.pkl`, `feature_names.pkl`) will be stored within the `model/` directory.

### 4. Running the API Server
To start the server and automatically launch the frontend Web UI in your default browser, run:
```bash
make deploy
```
Alternatively, if you only want to start the API without opening the browser, use:
```bash
make run
```

- **Web Interface:** Access the frontend application at [http://localhost:8000](http://localhost:8000)
- **API Documentation Sandbox:** Visit the auto-generated Swagger UI at [http://localhost:8000/docs](http://localhost:8000/docs)
- **Production Web (Render):** https://student-depression-predictation.onrender.com

### 5. Running via Docker
If you do not want to install Python dependencies manually, you can run the entire application (including model training and the API server) inside a Docker container using Docker Compose.

*Note: The Docker build process is optimized. If you have already trained the model locally (`model/model.pkl` exists), Docker will seamlessly use it and skip retraining, saving significant build time.*

**Start the application:**
```bash
docker-compose up --build -d
```
The API and frontend will be immediately available at `http://localhost:8000`.

**Stop the application:**
```bash
docker-compose down
```


## REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the main landing page UI. |
| `GET` | `/survey` | Serves the interactive student survey frontend. |
| `GET` | `/health` | API health check endpoint. |
| `POST`| `/predict`| Accepts a student profile (JSON payload) and returns depression prediction, risk level, and SHAP-based interpretations. |

### Example Request (`POST /predict`)
```json
{
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
```

## Authors
- **Nguyen Bui Trong**
