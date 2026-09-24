from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import math
import json
from functools import lru_cache

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

    @field_validator("cp")
    @classmethod
    def validate_cp(cls, value):
        if value not in {1, 2, 3, 4}:
            raise ValueError("cp must be one of: 1, 2, 3, 4")
        return value

    @field_validator("thal")
    @classmethod
    def validate_thal(cls, value):
        if value not in {3, 6, 7}:
            raise ValueError("thal must be one of: 3, 6, 7")
        return value

    @field_validator("ca")
    @classmethod
    def validate_ca(cls, value):
        if value not in {0, 1, 2, 3}:
            raise ValueError("ca must be one of: 0, 1, 2, 3")
        return value

    @field_validator("age", "thalach", "oldpeak")
    @classmethod
    def validate_finite(cls, value):
        if not math.isfinite(value):
            raise ValueError("Value must be finite")
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
            raise ValueError("All feature values must be finite numbers")
        return value


# ============================================================
# AI CHAT REQUEST
# ============================================================

class ChatRequest(BaseModel):
    disease: str
    question: str
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
    return {"status": "healthy"}


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
        return predict_heart(patient.model_dump())

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# BREAST CANCER PREDICTION
# ============================================================

@app.post("/predict/breast-cancer")
def predict_breast_cancer_endpoint(patient: BreastCancerPatient):

    try:
        return predict_breast_cancer(patient.model_dump())

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# BENCHMARK
# ============================================================

@app.get("/benchmark")
def benchmark():

    try:
        return get_all_benchmarks()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# HEART EXPLAINABILITY
# ============================================================

@app.post("/explain/heart")
def explain_heart_endpoint(patient: HeartPatient):

    try:
        return explain_heart(patient.model_dump())

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# BREAST CANCER EXPLAINABILITY
# ============================================================

@app.post("/explain/breast-cancer")
def explain_breast_cancer_endpoint(patient: BreastCancerPatient):

    try:
        return explain_breast_cancer(patient.model_dump())

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# FAST CHAT HELPERS
# ============================================================

def _q(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _direct_answer(disease, question, patient, result):
    q = _q(question)

    # Model comparison: answer locally from the actual screening outputs.
    # This avoids Groq latency and prevents truncated comparison answers.
    comparison_phrases = (
        "how did the classical and quantum models differ",
        "how do the classical and quantum models differ",
        "classical and quantum models differ",
        "compare classical and quantum models",
        "difference between classical and quantum models",
        "classical vs quantum models",
        "classical versus quantum models",
    )
    if any(p in q for p in comparison_phrases):
        models = result.get("models") or {}
        classical = [
            ("Logistic Regression", models.get("logistic_regression")),
            ("Random Forest", models.get("random_forest")),
            ("RBF SVM", models.get("rbf_svm")),
        ]
        quantum = [
            ("QSVM", models.get("qsvm")),
            ("VQC", models.get("vqc")),
        ]

        def summarize(group):
            available = [(name, data) for name, data in group if isinstance(data, dict)]
            counts = {0: 0, 1: 0}
            for _, data in available:
                pred = data.get("prediction")
                if pred in (0, 1):
                    counts[pred] += 1

            if not available:
                return "No model outputs were returned."

            parts = [
                f"{len(available)} models",
                f"{counts[0]} predicted class 0",
                f"{counts[1]} predicted class 1",
            ]

            probs = []
            for name, data in available:
                p0 = data.get("class_0_probability")
                p1 = data.get("class_1_probability")
                if isinstance(p0, (int, float)) and isinstance(p1, (int, float)):
                    probs.append(
                        f"{name}: {float(p0) * 100:.1f}% class 0 / "
                        f"{float(p1) * 100:.1f}% class 1"
                    )

            return "; ".join(parts) + ". " + (
                "Probability scores — " + "; ".join(probs) + "."
                if probs else ""
            )

        return (
            "The classical group contains Logistic Regression, Random Forest, "
            "and RBF SVM. The quantum group contains QSVM and VQC. "
            "For this screening: Classical — " + summarize(classical) +
            " Quantum — " + summarize(quantum) +
            " The models use different approaches, so their probability scores "
            "and predictions can differ; this comparison does not by itself show "
            "that one group is better."
        )

    # Result explanation: answer directly from the actual prediction.
    if any(
        phrase in q
        for phrase in (
            "what does the result mean",
            "what do the results mean",
            "explain the result",
            "explain the prediction",
            "what is the result",
        )
    ):
        label = result.get("prediction_label")

        if not label:
            prediction = result.get("prediction")
            if prediction == 1:
                label = "Higher likelihood of cardiovascular disease"
            elif prediction == 0:
                label = "Lower likelihood of cardiovascular disease"

        consensus = result.get("consensus") or {}
        agreeing = consensus.get("agreeing_models")
        total = consensus.get("total_models")
        percentage = consensus.get("percentage")

        if label:
            if agreeing is not None and total is not None and percentage is not None:
                return (
                    f"The models predict a {label.lower()}. "
                    f"{agreeing} of {total} models agree ({percentage:.0f}%). "
                    "This is a screening prediction, not a medical diagnosis."
                )

            return (
                f"The models predict a {label.lower()}. "
                "This is a screening prediction, not a medical diagnosis."
            )

    if q in {"hi", "hello", "hey", "hii"}:
        return "Hi. Ask me anything about this screening."

    if q in {"thanks", "thank you", "thx"}:
        return "You're welcome."

    if any(x in q for x in ("what is my result", "what was my result", "what is the prediction", "what did the model predict", "what did vitalis predict")):
        label = result.get("prediction_label")
        if label:
            return f"The screening result is {label}."
        pred = result.get("prediction")
        if pred is not None:
            return f"The screening prediction is {pred}."

    if "oldpeak" in q or "st depression" in q:
        if "my" in q or "value" in q:
            v = patient.get("oldpeak")
            if v is not None:
                return f"Your oldpeak value is {v}."
        return "Oldpeak measures ST-segment depression during exercise."

    if "qsvm" in q:
        return "QSVM is a quantum support vector machine used as one of VITALIS's research models."

    if "vqc" in q or "variational quantum classifier" in q:
        return "VQC is a trainable quantum classification model used in VITALIS."

    if disease == "heart":
        if "age" in q:
            v = patient.get("age")
            if v is not None: return f"Your age is {v:g} years."
        if any(x in q for x in ("heart rate", "thalach")):
            v = patient.get("thalach")
            if v is not None: return f"Your maximum heart rate is {v:g} bpm."
        if "chest pain" in q or "my cp" in q:
            v = patient.get("cp")
            names={1:"typical angina",2:"atypical angina",3:"non-anginal pain",4:"asymptomatic"}
            if v is not None:
                n=int(v); return f"Your chest pain type is {n} — {names.get(n,'unknown')}."
        if "major vessel" in q or "my ca" in q:
            v=patient.get("ca")
            if v is not None: return f"Your major-vessel value is {int(v)}."
        if "thalassemia" in q or "my thal" in q:
            v=patient.get("thal")
            names={3:"normal",6:"fixed defect",7:"reversible defect"}
            if v is not None:
                n=int(v); return f"Your thal value is {n} — {names.get(n,'unknown')}."

    if disease == "breast_cancer":
        normalized=q.replace(" ","_")
        for key,value in patient.items():
            if key.lower() in normalized:
                return f"Your {key.replace('_',' ')} value is {value}."

    return None


def _model_comparison_answer(result: dict) -> str | None:
    """Give a short deterministic comparison without calling Groq."""
    models = result.get("models") or {}
    if not all(name in models for name in ("logistic_regression", "random_forest", "rbf_svm", "qsvm", "vqc")):
        return None

    classical_names = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "rbf_svm": "RBF SVM",
    }
    quantum_names = {"qsvm": "QSVM", "vqc": "VQC"}

    def fmt_group(group):
        parts=[]
        for key,name in group.items():
            m=models[key]
            pred="class 1" if m.get("prediction") == 1 else "class 0"
            p=m.get("class_1_probability")
            if isinstance(p,(int,float)):
                parts.append(f"{name}: {pred} ({p*100:.1f}% class-1)")
            else:
                parts.append(f"{name}: {pred}")
        return "; ".join(parts)

    return (
        "The classical models were " + fmt_group(classical_names) + ". "
        "The quantum models were " + fmt_group(quantum_names) + ". "
        "So the main difference is that the first three use classical machine-learning methods, "
        "while QSVM and VQC use quantum-model approaches; their predictions can differ on the same input."
    )


def _top_features_answer(explainability: dict | None) -> str | None:
    if not explainability:
        return None
    groups = []
    for family in ("classical", "quantum"):
        for model, items in (explainability.get(family) or {}).items():
            if items:
                top = max(items, key=lambda x: float(x.get("importance", 0)))
                groups.append((float(top.get("importance", 0)), top.get("feature"), model))
    if not groups:
        return None
    groups.sort(reverse=True)
    names=[]
    seen=set()
    for _, feature, _ in groups:
        if feature and feature not in seen:
            names.append(feature)
            seen.add(feature)
        if len(names) == 3:
            break
    if not names:
        return None
    return "The strongest features in the supplied model explanations were " + ", ".join(names) + "."


def _deterministic_explanation(question: str, result: dict, explainability: dict | None) -> str | None:
    q=_q(question)

    if ("classical" in q and "quantum" in q and
            any(x in q for x in ("differ", "difference", "compare"))):
        return _model_comparison_answer(result)

    if any(x in q for x in (
        "what influenced",
        "influenced the prediction",
        "most important",
        "feature importance",
    )):
        return _top_features_answer(explainability)

    if ("why" in q and
            any(x in q for x in ("predict", "prediction", "result", "models"))):
        top = _top_features_answer(explainability)
        if top:
            return top + " These are model influences, not proof that any feature caused the condition."

    return None


def _cache_key(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


@lru_cache(maxsize=64)
def _cached_explainability(disease: str, data_json: str):
    data = json.loads(data_json)
    return explain_heart(data) if disease == "heart" else explain_breast_cancer(data)


# ============================================================
# AI CHAT
# ============================================================

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    try:
        disease=request.disease.lower().strip()
        question=request.question.strip()

        if disease not in {"heart", "breast_cancer"}:
            raise HTTPException(status_code=400, detail="Unsupported disease. Use 'heart' or 'breast_cancer'.")
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty.")

        # FAST PATH: no explainability and no Groq.
        direct=_direct_answer(disease, question, request.patient_data, request.prediction_result)
        if direct is not None:
            return {"disease": disease, "model": "vitalis-direct", "question": question, "explanation": direct}

        # Deterministic answers for common VITALIS questions.
        # These avoid both explainability work and Groq latency where possible.
        deterministic=_deterministic_explanation(
            question, request.prediction_result, None
        )
        if deterministic is not None:
            return {"disease": disease, "model": "vitalis-direct", "question": question, "explanation": deterministic}

        # Only compute explainability when the question actually needs it.
        explainability_result=None
        if any(x in _q(question) for x in (
            "why", "influenc", "feature importance", "important feature",
            "important factor", "reason for", "explain prediction"
        )):
            explainability_result=_cached_explainability(
                disease, _cache_key(request.patient_data)
            )

            deterministic=_deterministic_explanation(
                question, request.prediction_result, explainability_result
            )
            if deterministic is not None:
                return {"disease": disease, "model": "vitalis-direct", "question": question, "explanation": deterministic}

        explanation=explain_prediction(
            disease=disease,
            patient_data=request.patient_data,
            prediction_result=request.prediction_result,
            question=question,
            explainability_result=explainability_result,
        )

        return {"disease": disease, "model": "openai/gpt-oss-20b", "question": question, "explanation": explanation}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"{type(e).__name__}: {e}"
        )


