"""FastAPI backend for Student Depression Prediction."""
import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.deploy.schemas import StudentInput, PredictionResponse

from src.deploy.predict import *
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Dùng lifespan để nạp (preload) mô hình vào RAM ngay lúc khởi động (tránh Cold Start)
    try:
        load_models()
        print("Đã tải mô hình vào RAM thành công!")
    except FileNotFoundError:
        print("Cảnh báo: Không tìm thấy model. Hãy chạy lệnh `make train`.")
    yield

# Setup
app = FastAPI(
    title="Student Depression Predictor API",
    description="Dự đoán nguy cơ trầm cảm của sinh viên dựa trên các yếu tố học tập và cuộc sống.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# Routes
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
        result = predict_with_explanation(raw)
        return JSONResponse(content=result)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model chưa được train. Hãy chạy: make train hoặc python src/pipelines/pipeline.py. ({e})",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))