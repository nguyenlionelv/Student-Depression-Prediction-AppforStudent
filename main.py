"""FastAPI backend for Student Depression Prediction."""
import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.schemas import StudentInput, PredictionResponse

from src.predict import *

# ─── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Student Depression Predictor API",
    description="Dự đoán nguy cơ trầm cảm của sinh viên dựa trên các yếu tố học tập và cuộc sống.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ─── Lazy-load predict module after startup ───────────────────────────────────
_predict_fn = None

def get_predict_fn():
    global _predict_fn
    if _predict_fn is None:
        _predict_fn = predict_with_explanation
    return _predict_fn


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/survey")
async def survey_page():
    return FileResponse(os.path.join(STATIC_DIR, "survey.html"))


@app.get("/health")
async def health():
    return {"status": "ok", "message": "API đang hoạt động bình thường"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(data: StudentInput):
    try:
        raw = data.model_dump()
        predict_fn = get_predict_fn()
        result = predict_fn(raw)
        return JSONResponse(content=result)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model chưa được train. Hãy chạy: python model/train_and_save.py. ({e})",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
