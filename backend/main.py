from __future__ import annotations

import io
import math
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import pennylane as qml

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.train_pipeline import run_training_pipeline
from backend.train_quantum import quantum_kernel

from backend.heart_pipeline import predict_heart
from backend.breast_cancer_pipeline import predict_breast_cancer

from backend.benchmark_service import (
    get_all_benchmarks,
    get_session_benchmark,
)

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
        "for disease screening, benchmarking, and research."
    ),
    version="2.0.0",
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
# TRAINING SESSIONS
# ============================================================

TRAINING_SESSIONS: dict[str, dict[str, Any]] = {}


# ============================================================
# REQUEST MODELS
# ============================================================


class DatasetTrainRequest(BaseModel):
    target_column: str = Field(
        ...,
        description=(
            "Name of the target/label column "
            "in the uploaded CSV."
        ),
    )

    quantum_components: int = Field(
        default=6,
        ge=1,
        le=20,
        description="Number of quantum dimensions/qubits.",
    )

    test_size: float = Field(
        default=0.20,
        gt=0.05,
        lt=0.50,
        description="Fraction of the dataset reserved for testing.",
    )

    random_state: int = Field(
        default=42,
        description="Random seed used for the shared train/test split.",
    )


class GenericPredictionRequest(BaseModel):
    session_id: str
    features: dict[str, Any]


class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None

    # Kept for backward compatibility with the old disease-specific frontend.
    disease: str | None = None
    patient_data: dict[str, Any] | None = None
    prediction_result: dict[str, Any] | None = None

# ============================================================
# BASIC HELPERS
# ============================================================


def _json_safe(value: Any) -> Any:

    if value is None:
        return None

    if isinstance(value, (str, int, bool)):
        return value

    if isinstance(value, float):

        if math.isnan(value) or math.isinf(value):
            return None

        return value

    if hasattr(value, "item"):

        try:
            return _json_safe(value.item())
        except Exception:
            pass

    if isinstance(value, dict):

        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):

        return [
            _json_safe(item)
            for item in value
        ]

    if isinstance(value, pd.Series):

        return [
            _json_safe(item)
            for item in value.tolist()
        ]

    if isinstance(value, pd.DataFrame):

        return [
            {
                str(key): _json_safe(item)
                for key, item in row.items()
            }
            for row in value.to_dict(
                orient="records"
            )
        ]

    if hasattr(value, "tolist"):

        try:
            return _json_safe(value.tolist())
        except Exception:
            pass

    return str(value)


def _read_csv_upload(raw: bytes) -> pd.DataFrame:

    if not raw:
        raise ValueError(
            "Uploaded file is empty."
        )

    try:
        text = raw.decode("utf-8-sig")

    except UnicodeDecodeError as exc:

        raise ValueError(
            "CSV must be UTF-8 encoded."
        ) from exc

    if not text.strip():

        raise ValueError(
            "CSV contains no data."
        )

    try:

        dataframe = pd.read_csv(
            io.StringIO(text)
        )

    except Exception as exc:

        raise ValueError(
            f"Could not parse CSV: {exc}"
        ) from exc

    if dataframe.empty:

        raise ValueError(
            "CSV contains no rows."
        )

    if len(dataframe.columns) == 0:

        raise ValueError(
            "CSV contains no columns."
        )

    return dataframe


def _validate_target(
    dataframe: pd.DataFrame,
    target_column: str,
) -> None:

    target_column = target_column.strip()

    if not target_column:

        raise ValueError(
            "target_column cannot be empty."
        )

    if target_column not in dataframe.columns:

        raise ValueError(
            f"Target column '{target_column}' "
            f"was not found. "
            f"Available columns: "
            f"{list(dataframe.columns)}"
        )

    target = dataframe[target_column]

    if target.isna().all():

        raise ValueError(
            "The target column contains no usable values."
        )

    non_null_classes = (
        target.dropna().nunique()
    )

    if non_null_classes < 2:

        raise ValueError(
            "Training requires at least two target classes."
        )


def _dataset_summary(
    dataframe: pd.DataFrame,
    target_column: str | None = None,
) -> dict[str, Any]:

    numeric_columns = (
        dataframe
        .select_dtypes(include=["number"])
        .columns
        .tolist()
    )

    categorical_columns = (
        dataframe
        .select_dtypes(
            include=[
                "object",
                "category",
                "bool",
            ]
        )
        .columns
        .tolist()
    )

    missing = dataframe.isna().sum()

    summary = {
        "rows": int(len(dataframe)),
        "columns": int(len(dataframe.columns)),
        "column_names": dataframe.columns.tolist(),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "missing_values": int(
            dataframe.isna().sum().sum()
        ),
        "duplicate_rows": int(
            dataframe.duplicated().sum()
        ),
        "target_column": target_column,
    }

    if (
        target_column
        and target_column in dataframe.columns
    ):

        target = dataframe[target_column]

        distribution = (
            target
            .value_counts(dropna=False)
            .to_dict()
        )

        summary["target"] = {
            "classes": int(
                target.dropna().nunique()
            ),
            "distribution": {
                str(key): int(value)
                for key, value in distribution.items()
            },
        }

    summary["missing_by_column"] = {
        str(column): int(value)
        for column, value in missing.items()
        if int(value) > 0
    }

    return summary


def _public_training_result(
    result: dict[str, Any],
    session_id: str,
    dataframe: pd.DataFrame,
    target_column: str,
) -> dict[str, Any]:

    public = {}

    for key, value in result.items():

        if key == "_artifacts":
            continue

        public[key] = _json_safe(value)

    artifacts = result.get(
        "_artifacts",
        {},
    )

    selected_features = (
        artifacts.get(
            "selected_features"
        )
    )

    if selected_features is None:

        selected_features = (
            result.get(
                "selected_features"
            )
        )

    quantum_features = (
        artifacts.get(
            "X_train_quantum"
        )
    )

    return {
        "session_id": session_id,
        "status": "trained",

        "dataset": {
            "rows": int(len(dataframe)),
            "columns": int(len(dataframe.columns)),
            "target_column": target_column,
        },

        "selected_features": _json_safe(
            selected_features
        ),

        "quantum_dimensions": (
            int(quantum_features.shape[1])
            if hasattr(
                quantum_features,
                "shape",
            )
            and len(quantum_features.shape) == 2
            else None
        ),

        "training_result": public,
    }


# ============================================================
# QUANTUM PREDICTION
# ============================================================


def _build_prediction_feature_map(
    n_qubits: int,
):

    dev = qml.device(
        "default.qubit",
        wires=n_qubits,
    )

    @qml.qnode(dev)
    def feature_map(x):

        for i in range(n_qubits):

            qml.RY(
                x[i],
                wires=i,
            )

        for i in range(n_qubits - 1):

            qml.CNOT(
                wires=[
                    i,
                    i + 1,
                ]
            )

        return qml.state()

    return feature_map


def _predict_qsvm(
    qsvm_result: dict[str, Any],
    quantum_features,
) -> dict[str, Any]:

    qsvm = qsvm_result.get(
        "qsvm"
    )

    scaler = qsvm_result.get(
        "scaler"
    )

    training_data = qsvm_result.get(
        "training_data"
    )

    if qsvm is None:

        raise ValueError(
            "QSVM model artifact is missing."
        )

    if scaler is None:

        raise ValueError(
            "QSVM scaler artifact is missing."
        )

    if training_data is None:

        raise ValueError(
            "QSVM training quantum data is missing."
        )

    quantum_features = pd.DataFrame(
        quantum_features
    )

    scaled_sample = scaler.transform(
        quantum_features
    )

    n_qubits = scaled_sample.shape[1]

    feature_map = (
        _build_prediction_feature_map(
            n_qubits
        )
    )

    kernel_row = []

    for train_row in training_data:

        kernel_value = quantum_kernel(
            scaled_sample[0],
            train_row,
            feature_map,
            n_qubits,
        )

        kernel_row.append(
            kernel_value
        )

    kernel_matrix = [
        kernel_row
    ]

    prediction = qsvm.predict(
        kernel_matrix
    )[0]

    probability = None

    if hasattr(
        qsvm,
        "predict_proba",
    ):

        probabilities = (
            qsvm.predict_proba(
                kernel_matrix
            )[0]
        )

        if len(probabilities) > 1:

            probability = float(
                probabilities[1]
            )

    return {
        "prediction": _json_safe(
            prediction
        ),
        "class_1_probability": probability,
    }


def _predict_vqc(
    vqc_result: dict[str, Any],
    quantum_features,
) -> dict[str, Any]:

    model = vqc_result.get(
        "model"
    )

    scaler = vqc_result.get(
        "scaler"
    )

    weights = vqc_result.get(
        "weights"
    )

    if model is None:

        raise ValueError(
            "VQC model artifact is missing."
        )

    if scaler is None:

        raise ValueError(
            "VQC scaler artifact is missing."
        )

    if weights is None:

        weights = getattr(
            model,
            "weights",
            None,
        )

    if weights is None:

        raise ValueError(
            "VQC trained weights are missing."
        )

    quantum_features = pd.DataFrame(
        quantum_features
    )

    scaled_sample = scaler.transform(
        quantum_features
    )

    probabilities = (
        model.predict_proba(
            scaled_sample,
            weights,
        )
    )

    probability = float(
        probabilities[0]
    )

    prediction = int(
        probability >= 0.5
    )

    return {
        "prediction": prediction,
        "class_1_probability": probability,
    }


# ============================================================
# REFERENCE DATASET DOWNLOADS
# ============================================================


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@app.get("/datasets/heart")
def download_heart_dataset():
    """Download the cleaned cardiovascular reference dataset."""

    dataset_path = DATA_DIR / "heart_cleaned.csv"

    if not dataset_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Heart reference dataset is not available on the server.",
        )

    return FileResponse(
        path=dataset_path,
        media_type="text/csv",
        filename="heart_cleaned.csv",
    )


@app.get("/datasets/breast-cancer")
def download_breast_cancer_dataset():
    """Download the headered breast-cancer reference dataset."""

    candidates = (
        DATA_DIR / "bcancer_fixed.csv",
        DATA_DIR / "breast_cancer.csv",
        DATA_DIR / "bcancer.csv",
    )

    dataset_path = next(
        (path for path in candidates if path.is_file()),
        None,
    )

    if dataset_path is None:
        raise HTTPException(
            status_code=404,
            detail="Breast-cancer reference dataset is not available on the server.",
        )

    return FileResponse(
        path=dataset_path,
        media_type="text/csv",
        filename="breast_cancer.csv",
    )


# ============================================================
# ROOT
# ============================================================


@app.get("/")
def root():

    return {
        "name": "VITALIS",
        "description": (
            "Hybrid classical-quantum machine learning "
            "platform for disease screening and research."
        ),
        "version": "2.0.0",
        "status": "operational",
    }


# ============================================================
# HEALTH
# ============================================================


@app.api_route(
    "/health",
    methods=["GET", "HEAD"],
)
def health():

    return {
        "status": "healthy",
        "active_training_sessions": len(
            TRAINING_SESSIONS
        ),
    }


# ============================================================
# DATASET PROFILING
# ============================================================


@app.post("/upload/dataset")
async def upload_dataset(
    file: UploadFile = File(...),
):

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No filename was provided.",
        )

    if not file.filename.lower().endswith(
        ".csv"
    ):

        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV dataset.",
        )

    raw = await file.read()

    if len(raw) > 50 * 1024 * 1024:

        raise HTTPException(
            status_code=413,
            detail=(
                "CSV file is too large. "
                "Maximum size is 50 MB."
            ),
        )

    try:

        dataframe = _read_csv_upload(
            raw
        )

        summary = _dataset_summary(
            dataframe
        )

        return {
            "filename": file.filename,
            "status": "validated",
            "summary": summary,
            "preview": _json_safe(
                dataframe.head(10)
            ),
        }

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Dataset profiling failed: {exc}"
            ),
        )


# ============================================================
# GENERIC TRAINING
# ============================================================


@app.post("/train/dataset")
async def train_dataset(
    file: UploadFile = File(...),
    target_column: str | None = None,
    quantum_components: int = 6,
    test_size: float = 0.20,
    random_state: int = 42,
):

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No filename was provided.",
        )

    if not file.filename.lower().endswith(
        ".csv"
    ):

        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV dataset.",
        )

    if not target_column:

        raise HTTPException(
            status_code=400,
            detail=(
                "target_column is required. "
                "Specify the column containing the class/label."
            ),
        )

    if (
        quantum_components < 1
        or quantum_components > 20
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "quantum_components must be between 1 and 20."
            ),
        )

    if not (
        0.05 < test_size < 0.50
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "test_size must be between 0.05 and 0.50."
            ),
        )

    raw = await file.read()

    if len(raw) > 50 * 1024 * 1024:

        raise HTTPException(
            status_code=413,
            detail=(
                "CSV file is too large. "
                "Maximum size is 50 MB."
            ),
        )

    try:

        dataframe = _read_csv_upload(
            raw
        )

        target_column = (
            target_column.strip()
        )

        _validate_target(
            dataframe,
            target_column,
        )

        session_id = uuid.uuid4().hex

        print()
        print("=" * 70)
        print("VITALIS API TRAINING")
        print("=" * 70)

        print(
            f"Dataset       : {file.filename}"
        )

        print(
            f"Rows          : {len(dataframe)}"
        )

        print(
            f"Columns       : {len(dataframe.columns)}"
        )

        print(
            f"Target        : {target_column}"
        )

        print(
            f"Quantum dims  : {quantum_components}"
        )

        print(
            f"Test size     : {test_size}"
        )

        print(
            f"Random state  : {random_state}"
        )

        print("=" * 70)

        result = run_training_pipeline(
            dataframe,
            target_column=target_column,
            quantum_components=quantum_components,
            test_size=test_size,
            random_state=random_state,
        )

        TRAINING_SESSIONS[
            session_id
        ] = {
            "filename": file.filename,
            "dataframe": dataframe,
            "target_column": target_column,
            "result": result,
        }

        return _public_training_result(
            result=result,
            session_id=session_id,
            dataframe=dataframe,
            target_column=target_column,
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Training failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        )


# ============================================================
# TRAINING SESSION
# ============================================================


@app.get(
    "/train/session/{session_id}"
)
def get_training_session(
    session_id: str,
):

    session = TRAINING_SESSIONS.get(
        session_id
    )

    if session is None:

        raise HTTPException(
            status_code=404,
            detail="Training session not found.",
        )

    return _public_training_result(
        result=session["result"],
        session_id=session_id,
        dataframe=session["dataframe"],
        target_column=session["target_column"],
    )


# ============================================================
# GENERIC PREDICTION
# ============================================================


@app.post("/predict/dataset")
def predict_dataset(
    request: GenericPredictionRequest,
):

    session = TRAINING_SESSIONS.get(
        request.session_id
    )

    if session is None:

        raise HTTPException(
            status_code=404,
            detail="Training session not found.",
        )

    result = session["result"]

    artifacts = result.get(
        "_artifacts",
        {},
    )

    selected_features = (
        artifacts.get(
            "selected_features"
        )
    )

    if selected_features is None:

        X_train_selected = (
            artifacts.get(
                "X_train_selected"
            )
        )

        if X_train_selected is not None:

            selected_features = (
                X_train_selected
                .columns
                .tolist()
            )

    if not selected_features:

        raise HTTPException(
            status_code=500,
            detail=(
                "Selected feature information is unavailable."
            ),
        )

    missing = [
        feature
        for feature in selected_features
        if feature not in request.features
    ]

    if missing:

        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    "Missing required features."
                ),
                "missing_features": missing,
            },
        )

    try:

        # ====================================================
        # BUILD SAMPLE
        # ====================================================

        sample = pd.DataFrame(
            [
                {
                    feature: request.features[feature]
                    for feature in selected_features
                }
            ]
        )

        # ====================================================
        # CLASSICAL MODELS
        # ====================================================

        classical_models = (
            artifacts.get(
                "classical_models"
            )
        )

        classical_scaler = (
            artifacts.get(
                "classical_scaler"
            )
        )

        if not classical_models:

            raise ValueError(
                "Classical models were not found."
            )

        sample_classical = (
            sample.apply(
                pd.to_numeric,
                errors="raise",
            )
        )

        if classical_scaler is not None:

            sample_classical_scaled = (
                classical_scaler.transform(
                    sample_classical
                )
            )

        else:

            sample_classical_scaled = (
                sample_classical
            )

        classical_predictions = {}

        for (
            model_name,
            model,
        ) in classical_models.items():

            prediction = model.predict(
                sample_classical_scaled
            )[0]

            probability = None

            if hasattr(
                model,
                "predict_proba",
            ):

                probabilities = (
                    model.predict_proba(
                        sample_classical_scaled
                    )[0]
                )

                if len(probabilities) > 1:

                    probability = float(
                        probabilities[1]
                    )

            classical_predictions[
                model_name
            ] = {
                "prediction": _json_safe(
                    prediction
                ),
                "class_1_probability": probability,
            }

        # ====================================================
        # QUANTUM REDUCTION
        # ====================================================

        reducer = artifacts.get(
            "reducer"
        )

        if reducer is None:

            raise ValueError(
                "Quantum feature reducer was not found."
            )

        quantum_reduced = (
            reducer.transform(
                sample
            )
        )

        quantum_feature_names = (
            reducer.get_feature_names()
        )

        quantum_sample = pd.DataFrame(
            quantum_reduced,
            columns=quantum_feature_names,
        )

        # ====================================================
        # QSVM
        # ====================================================

        # IMPORTANT:
        #
        # train_pipeline stores the fitted QSVM
        # result directly as quantum_result.
        #
        # Therefore:
        #
        # quantum_result = {
        #     "qsvm": SVC(...),
        #     "scaler": ...,
        #     "training_data": ...,
        #     ...
        # }
        #
        # We must NOT do:
        #
        # quantum_result["qsvm"]
        #
        # because that gives the raw SVC.

        qsvm_result = artifacts.get(
            "quantum_result"
        )

        if qsvm_result is None:

            raise ValueError(
                "QSVM training result was not found."
            )

        # Defensive validation: train_quantum() must return the complete
        # result dictionary, not the raw sklearn SVC object.
        # The prediction helper needs the QSVM, scaler, and training data.
        print(
            "\n[DEBUG] quantum_result type:",
            type(qsvm_result),
        )

        if not isinstance(qsvm_result, dict):
            raise TypeError(
                "Invalid QSVM artifact: expected the complete "
                "train_quantum() result dictionary, but received "
                f"{type(qsvm_result).__name__}. "
                "The artifact must contain 'qsvm', 'scaler', "
                "and 'training_data'."
            )

        print(
            "[DEBUG] quantum_result keys:",
            list(qsvm_result.keys()),
        )

        required_qsvm_keys = {
            "qsvm",
            "scaler",
            "training_data",
        }

        missing_qsvm_keys = sorted(
            required_qsvm_keys - set(qsvm_result.keys())
        )

        if missing_qsvm_keys:
            raise ValueError(
                "QSVM training result is incomplete. Missing artifact "
                f"keys: {missing_qsvm_keys}"
            )

        qsvm_prediction = (
            _predict_qsvm(
                qsvm_result,
                quantum_sample,
            )
        )

        # ====================================================
        # VQC
        # ====================================================

        vqc_result = artifacts.get(
            "vqc_result"
        )

        if vqc_result is None:

            raise ValueError(
                "VQC training result was not found."
            )

        vqc_prediction = (
            _predict_vqc(
                vqc_result,
                quantum_sample,
            )
        )

        # ====================================================
        # BUILD MODEL RESPONSE
        # ====================================================

        models = {}

        for (
            model_name,
            prediction,
        ) in classical_predictions.items():

            models[
                model_name
            ] = {
                **prediction,
                "family": "classical",
            }

        models[
            "Quantum Kernel SVM"
        ] = {
            **qsvm_prediction,
            "family": "quantum",
        }

        models[
            "Variational Quantum Classifier"
        ] = {
            **vqc_prediction,
            "family": "quantum",
        }

        # ====================================================
        # CONSENSUS
        # ====================================================

        prediction_values = []

        for model_result in models.values():

            prediction_value = (
                model_result.get(
                    "prediction"
                )
            )

            if prediction_value is not None:

                try:

                    prediction_values.append(
                        int(
                            prediction_value
                        )
                    )

                except Exception:
                    pass

        consensus = None

        if prediction_values:

            class_1_votes = sum(
                value == 1
                for value in prediction_values
            )

            class_0_votes = sum(
                value == 0
                for value in prediction_values
            )

            total_models = len(
                prediction_values
            )

            consensus_prediction = (
                1
                if class_1_votes > class_0_votes
                else 0
            )

            consensus = {
                "prediction": consensus_prediction,
                "class_0_votes": class_0_votes,
                "class_1_votes": class_1_votes,
                "total_models": total_models,
                "agreeing_models": max(
                    class_0_votes,
                    class_1_votes,
                ),
            }

        return {
            "session_id": request.session_id,

            "models": models,

            "consensus": consensus,

            "features_used": selected_features,

            "quantum_features": quantum_feature_names,
        }

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Prediction failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        )


# ============================================================
# LEGACY DISEASE ENDPOINTS
# ============================================================


@app.post("/predict/heart")
def predict_heart_endpoint(
    patient: dict,
):

    try:

        return predict_heart(
            patient
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


@app.post("/predict/breast-cancer")
def predict_breast_cancer_endpoint(
    patient: dict,
):

    try:

        return predict_breast_cancer(
            patient
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# BENCHMARK
# ============================================================


@app.get("/benchmark")
def benchmark():

    try:

        return get_all_benchmarks()

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


@app.get(
    "/benchmark/session/{session_id}"
)
def benchmark_session(
    session_id: str,
):

    session = TRAINING_SESSIONS.get(
        session_id
    )

    if session is None:

        raise HTTPException(
            status_code=404,
            detail="Training session not found.",
        )

    try:

        benchmark_result = (
            get_session_benchmark(
                session["result"]
            )
        )

        return {
            "session_id": session_id,
            "benchmark": _json_safe(
                benchmark_result
            ),
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Benchmark generation failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        )


# ============================================================
# GENERIC DATASET EXPLAINABILITY
# ============================================================


@app.post("/explain/dataset")
def explain_dataset(
    request: GenericPredictionRequest,
):
    """
    Generate model-agnostic local feature influence
    information for a trained dataset session.

    Uses the same fitted training artifacts as prediction.
    No model is retrained here.
    """

    session = TRAINING_SESSIONS.get(
        request.session_id
    )

    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Training session not found.",
        )

    result = session["result"]
    artifacts = result.get("_artifacts", {})

    selected_features = artifacts.get("selected_features")

    if selected_features is None:
        X_train_selected = artifacts.get("X_train_selected")
        if X_train_selected is not None:
            selected_features = X_train_selected.columns.tolist()

    if not selected_features:
        raise HTTPException(
            status_code=500,
            detail="Selected feature information is unavailable.",
        )

    missing = [
        feature
        for feature in selected_features
        if feature not in request.features
    ]

    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Missing required features.",
                "missing_features": missing,
            },
        )

    try:
        sample = pd.DataFrame([
            {
                feature: request.features[feature]
                for feature in selected_features
            }
        ])

        sample_numeric = sample.apply(
            pd.to_numeric,
            errors="raise",
        )

        classical_models = artifacts.get("classical_models")
        classical_scaler = artifacts.get("classical_scaler")

        if not classical_models:
            raise ValueError("Classical models were not found.")

        if classical_scaler is not None:
            sample_scaled = classical_scaler.transform(sample_numeric)
        else:
            sample_scaled = sample_numeric

        feature_influence = {}

        logistic_model = classical_models.get("Logistic Regression")
        if logistic_model is not None:
            coefficients = logistic_model.coef_[0]
            feature_influence["Logistic Regression"] = {
                feature: float(coefficients[index])
                for index, feature in enumerate(selected_features)
            }

        random_forest = classical_models.get("Random Forest")
        if random_forest is not None:
            importances = random_forest.feature_importances_
            feature_influence["Random Forest"] = {
                feature: float(importances[index])
                for index, feature in enumerate(selected_features)
            }

        perturbation = {}

        for index, feature in enumerate(selected_features):
            original_value = float(sample_numeric.iloc[0, index])
            delta = max(abs(original_value) * 0.05, 0.05)

            lower = sample_numeric.copy()
            upper = sample_numeric.copy()
            lower.iloc[0, index] = original_value - delta
            upper.iloc[0, index] = original_value + delta

            if classical_scaler is not None:
                lower_scaled = classical_scaler.transform(lower)
                upper_scaled = classical_scaler.transform(upper)
            else:
                lower_scaled = lower
                upper_scaled = upper

            model_changes = {}

            for model_name, model in classical_models.items():
                if not hasattr(model, "predict_proba"):
                    continue

                lower_probability = float(
                    model.predict_proba(lower_scaled)[0][1]
                )
                upper_probability = float(
                    model.predict_proba(upper_scaled)[0][1]
                )

                model_changes[model_name] = {
                    "lower_probability": lower_probability,
                    "upper_probability": upper_probability,
                    "change": upper_probability - lower_probability,
                }

            perturbation[feature] = model_changes

        feature_scores = {}

        for feature, models in perturbation.items():
            changes = [
                abs(model_result["change"])
                for model_result in models.values()
            ]
            feature_scores[feature] = (
                float(sum(changes) / len(changes))
                if changes
                else 0.0
            )

        ranked_features = sorted(
            feature_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return {
            "session_id": request.session_id,
            "method": (
                "Local perturbation sensitivity "
                "plus model-native feature importance"
            ),
            "features": selected_features,
            "model_native_importance": _json_safe(feature_influence),
            "local_perturbation": _json_safe(perturbation),
            "ranked_features": [
                {"feature": feature, "score": score}
                for feature, score in ranked_features
            ],
            "note": (
                "Feature influence indicates model sensitivity and "
                "association with the prediction. It does not establish "
                "causation or constitute a medical diagnosis."
            ),
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Explainability failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        )


# ============================================================
# LEGACY EXPLAINABILITY
# ============================================================


@app.post("/explain/heart")
def explain_heart_endpoint(
    patient: dict,
):

    try:

        return explain_heart(
            patient
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


@app.post("/explain/breast-cancer")
def explain_breast_cancer_endpoint(
    patient: dict,
):

    try:

        return explain_breast_cancer(
            patient
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# LEGACY CHAT
# ============================================================


def _q(text: str) -> str:

    return " ".join(
        text.lower()
        .strip()
        .split()
    )


def _direct_answer(
    disease: str,
    question: str,
    patient: dict,
    result: dict,
) -> str | None:

    q = _q(question)

    if q in {
        "hi",
        "hello",
        "hey",
        "hii",
        "yo",
    }:

        return (
            "Hi. Ask me about the screening result, "
            "model outputs, features, or how VITALIS works."
        )

    if q in {
        "thanks",
        "thank you",
        "thx",
        "thank u",
    }:

        return "You're welcome."

    if any(
        phrase in q
        for phrase in (
            "what is my result",
            "what was my result",
            "what is the prediction",
            "what did the model predict",
            "what did vitalis predict",
            "what is my prediction",
            "show my result",
        )
    ):

        label = result.get(
            "prediction_label"
        )

        if label:

            consensus = (
                result.get(
                    "consensus"
                )
                or {}
            )

            agreeing = (
                consensus.get(
                    "agreeing_models"
                )
            )

            total = (
                consensus.get(
                    "total_models"
                )
            )

            if (
                agreeing is not None
                and total is not None
            ):

                return (
                    f"The screening result is {label}. "
                    f"{agreeing} of {total} models agree."
                )

            return (
                f"The screening result is {label}."
            )

        prediction = result.get(
            "prediction"
        )

        if prediction is not None:

            return (
                f"The screening prediction is class {prediction}."
            )

    if any(
        phrase in q
        for phrase in (
            "classical vs quantum",
            "classical versus quantum",
            "difference between classical and quantum",
            "compare classical and quantum",
            "how did the classical and quantum models differ",
        )
    ):

        models = (
            result.get(
                "models"
            )
            or {}
        )

        classical = [
            "logistic_regression",
            "random_forest",
            "rbf_svm",
        ]

        quantum = [
            "qsvm",
            "vqc",
        ]

        classical_available = [
            name
            for name in classical
            if name in models
        ]

        quantum_available = [
            name
            for name in quantum
            if name in models
        ]

        return (
            "VITALIS uses Logistic Regression, Random Forest, "
            "and RBF SVM as classical models, while QSVM and "
            "VQC represent the quantum-model approaches. "
            f"The current result contains "
            f"{len(classical_available)} classical and "
            f"{len(quantum_available)} quantum model outputs. "
            "Differences in their predictions are measured "
            "from the actual model outputs; this comparison "
            "does not by itself establish that one approach "
            "is superior."
        )

    if disease == "heart":

        if "oldpeak" in q:

            value = patient.get(
                "oldpeak"
            )

            if (
                value is not None
                and any(
                    x in q
                    for x in (
                        "my",
                        "mine",
                        "value",
                        "what is",
                    )
                )
            ):

                return (
                    f"Your oldpeak value is {value}."
                )

            return (
                "Oldpeak is the coded ST-segment "
                "depression feature used by the heart "
                "screening dataset."
            )

        if (
            "heart rate" in q
            or "thalach" in q
            or "maximum heart rate" in q
        ):

            value = patient.get(
                "thalach"
            )

            if value is not None:

                return (
                    f"Your maximum heart rate value "
                    f"is {value} bpm."
                )

            return (
                "Maximum heart rate (thalach) is the "
                "peak heart rate recorded during the test."
            )

        if "chest pain" in q:

            value = patient.get(
                "cp"
            )

            names = {
                1: "typical angina",
                2: "atypical angina",
                3: "non-anginal pain",
                4: "asymptomatic",
            }

            if value is not None:

                number = int(value)

                return (
                    f"Your chest pain type is {number} — "
                    f"{names.get(number, 'unknown')}."
                )

        if "age" in q:

            value = patient.get(
                "age"
            )

            if value is not None:

                return (
                    f"Your age is {value:g} years."
                )

    if disease == "breast_cancer":

        normalized = q.replace(
            " ",
            "_",
        )

        for key, value in patient.items():

            if key.lower() in normalized:

                return (
                    f"Your {key.replace('_', ' ')} "
                    f"value is {value}."
                )

    return None


@app.post("/chat")
def chat_endpoint(
    request: ChatRequest,
):
    try:
        question = request.question.strip()

        if not question:
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty.",
            )

        # ========================================================
        # GENERIC SESSION CHAT
        # ========================================================

        if request.session_id:
            session = TRAINING_SESSIONS.get(
                request.session_id
            )

            if session is None:
                raise HTTPException(
                    status_code=404,
                    detail="Training session not found.",
                )

            result = session["result"]

            # Public training information
            dataset_context = _public_training_result(
                result=result,
                session_id=request.session_id,
                dataframe=session["dataframe"],
                target_column=session["target_column"],
            )

            # Benchmark information
            try:
                benchmark_context = get_session_benchmark(
                    result
                )
            except Exception:
                benchmark_context = None

            explanation = explain_prediction(
                disease=request.disease,
                patient_data=request.patient_data or {},
                prediction_result=request.prediction_result or {},
                question=question,
                explainability_result=None,
                dataset_context=dataset_context,
                benchmark_context=benchmark_context,
            )

            return {
                "session_id": request.session_id,
                "model": "openai/gpt-oss-20b",
                "question": question,
                "explanation": explanation,
            }

        # ========================================================
        # LEGACY DISEASE CHAT
        # ========================================================

        disease = (
            request.disease or "unknown"
        ).lower().strip()

        patient_data = request.patient_data or {}
        prediction_result = request.prediction_result or {}

        direct = _direct_answer(
            disease=disease,
            question=question,
            patient=patient_data,
            result=prediction_result,
        )

        if direct is not None:
            return {
                "disease": disease,
                "model": "vitalis-direct",
                "question": question,
                "explanation": direct,
            }

        explanation = explain_prediction(
            disease=disease,
            patient_data=patient_data,
            prediction_result=prediction_result,
            question=question,
            explainability_result=None,
        )

        return {
            "disease": disease,
            "model": "openai/gpt-oss-20b",
            "question": question,
            "explanation": explanation,
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"{type(exc).__name__}: {exc}"
            ),
        )

# ============================================================
# LOCAL DEVELOPMENT
# ============================================================


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )