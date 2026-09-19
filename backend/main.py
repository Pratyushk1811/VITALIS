from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import math

from backend.heart_pipeline import predict_heart
from backend.breast_cancer_pipeline import predict_breast_cancer

from backend.benchmark_service import get_all_benchmarks

from backend.explainability_service import (
    explain_heart,
    explain_breast_cancer,
)


from backend.chatbot_service import explain_prediction


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="VITALIS API",
    description=(
        "Hybrid classical-quantum machine learning platform "
        "for early disease screening and research benchmarking."
    ),
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# INPUT MODELS
# ============================================================

class HeartPatient(BaseModel):
    cp: float
    thal: float
    thalach: float
    oldpeak: float
    ca: float
    age: float

    # --------------------------------------------------------
    # CATEGORICAL VALIDATION
    # --------------------------------------------------------

    @field_validator("cp")
    @classmethod
    def validate_cp(cls, value):
        if value not in {1, 2, 3, 4}:
            raise ValueError(
                "cp must be one of: 1, 2, 3, 4"
            )
        return value

    @field_validator("thal")
    @classmethod
    def validate_thal(cls, value):
        if value not in {3, 6, 7}:
            raise ValueError(
                "thal must be one of: 3, 6, 7"
            )
        return value

    @field_validator("ca")
    @classmethod
    def validate_ca(cls, value):
        if value not in {0, 1, 2, 3}:
            raise ValueError(
                "ca must be one of: 0, 1, 2, 3"
            )
        return value

    # --------------------------------------------------------
    # NUMERIC VALIDATION
    # --------------------------------------------------------

    @field_validator(
        "age",
        "thalach",
        "oldpeak",
    )
    @classmethod
    def validate_finite(cls, value):
        if not math.isfinite(value):
            raise ValueError(
                "Value must be finite"
            )
        return value


class BreastCancerPatient(BaseModel):
    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    smoothness_mean: float
    compactness_mean: float
    concavity_mean: float
    concave_points_mean: float
    symmetry_mean: float
    fractal_dimension_mean: float

    radius_se: float
    texture_se: float
    perimeter_se: float
    area_se: float
    smoothness_se: float
    compactness_se: float
    concavity_se: float
    concave_points_se: float
    symmetry_se: float
    fractal_dimension_se: float

    radius_worst: float
    texture_worst: float
    perimeter_worst: float
    area_worst: float
    smoothness_worst: float
    compactness_worst: float
    concavity_worst: float
    concave_points_worst: float
    symmetry_worst: float
    fractal_dimension_worst: float

    @field_validator("*")
    @classmethod
    def validate_finite(cls, value):
        if not math.isfinite(value):
            raise ValueError(
                "All feature values must be finite numbers"
            )
        return value


# ============================================================
# AI CHAT REQUEST
# ============================================================

class ChatRequest(BaseModel):
    disease: str
    patient_data: dict
    prediction_result: dict


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "name": "VITALIS",
        "description": (
            "Hybrid classical-quantum machine learning "
            "platform for disease screening."
        ),
        "status": "operational",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# ============================================================
# DISEASES
# ============================================================

@app.get("/diseases")
def diseases():
    return {
        "diseases": [
            {
                "id": "heart",
                "name": "Cardiovascular Disease",
                "status": "available",
            },
            {
                "id": "breast_cancer",
                "name": "Breast Cancer",
                "status": "available",
            },
        ]
    }


# ============================================================
# HEART PREDICTION
# ============================================================

@app.post("/predict/heart")
def predict_heart_endpoint(patient: HeartPatient):

    try:

        result = predict_heart(
            patient.model_dump()
        )

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# BREAST CANCER PREDICTION
# ============================================================

@app.post("/predict/breast-cancer")
def predict_breast_cancer_endpoint(
    patient: BreastCancerPatient
):

    try:

        result = predict_breast_cancer(
            patient.model_dump()
        )

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# BENCHMARK
# ============================================================

@app.get("/benchmark")
def benchmark():

    try:

        return get_all_benchmarks()

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# HEART EXPLAINABILITY
# ============================================================

@app.post("/explain/heart")
def explain_heart_endpoint(patient: HeartPatient):

    try:

        result = explain_heart(
            patient.model_dump()
        )

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# BREAST CANCER EXPLAINABILITY
# ============================================================

@app.post("/explain/breast-cancer")
def explain_breast_cancer_endpoint(
    patient: BreastCancerPatient
):

    try:

        result = explain_breast_cancer(
            patient.model_dump()
        )

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# AI EXPLANATION
# ============================================================

@app.post("/chat")
def chat_endpoint(request: ChatRequest):

    try:

        disease = request.disease.lower().strip()

        if disease not in {
            "heart",
            "breast_cancer",
        }:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Unsupported disease. "
                    "Use 'heart' or 'breast_cancer'."
                )
            )

        # ----------------------------------------------------
        # GENERATE VITALIS EXPLAINABILITY
        # ----------------------------------------------------

        if disease == "heart":

            explainability_result = explain_heart(
                request.patient_data
            )

        else:

            explainability_result = explain_breast_cancer(
                request.patient_data
            )

        # ----------------------------------------------------
        # SEND DATA TO OPENROUTER
        # ----------------------------------------------------

        explanation = explain_prediction(
            disease=disease,
            patient_data=request.patient_data,
            prediction_result=request.prediction_result,
            explainability_result=explainability_result,
        )

        return {
            "disease": disease,
            "model": "openai/gpt-oss-20b",
            "explanation": explanation,
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )