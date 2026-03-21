"""Pydantic schemas for request/response."""
from pydantic import BaseModel, Field
from typing import List, Optional


class StudentInput(BaseModel):
    gender: str = Field(..., description="Male hoặc Female")
    age: float = Field(..., ge=10, le=60)
    academic_pressure: float = Field(..., ge=0, le=5)
    work_pressure: float = Field(..., ge=0, le=5)
    cgpa: float = Field(..., ge=0.0, le=10.0)
    study_satisfaction: float = Field(..., ge=0, le=5)
    job_satisfaction: float = Field(..., ge=0, le=5)
    sleep_duration: str = Field(..., description="Less than 5 hours / 5-6 hours / 7-8 hours / More than 8 hours")
    degree: str = Field(..., description="HSC / BSc / B.Tech / MBA / PhD / Others ...")
    suicidal_thoughts: str = Field(..., description="Yes hoặc No")
    work_study_hours: float = Field(..., ge=0, le=24)
    financial_stress: float = Field(..., ge=0, le=5)
    family_history: str = Field(..., description="Yes hoặc No")
    dietary_habits: str = Field(..., description="Healthy / Moderate / Unhealthy / Others")

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Male",
                "age": 22,
                "academic_pressure": 4,
                "work_pressure": 1,
                "cgpa": 7.2,
                "study_satisfaction": 2,
                "job_satisfaction": 1,
                "sleep_duration": "5-6 hours",
                "degree": "B.Tech",
                "suicidal_thoughts": "No",
                "work_study_hours": 8,
                "financial_stress": 4,
                "family_history": "No",
                "dietary_habits": "Unhealthy",
            }
        }


class FactorDetail(BaseModel):
    feature: str
    label_vi: str
    direction: str        # "tăng nguy cơ" / "giảm nguy cơ"
    impact_level: str     # "cao" / "trung bình" / "thấp"
    shap_value: float


class PredictionResponse(BaseModel):
    prediction: int                    # 0 or 1
    probability: float                 # probability of depression
    risk_level: str                    # "Thấp" / "Trung bình" / "Cao" / "Rất cao"
    top_factors: List[FactorDetail]
    explanation: str                   # summary paragraph in Vietnamese
    recommendations: List[str]
