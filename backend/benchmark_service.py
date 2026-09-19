from pathlib import Path
import json


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


# ============================================================
# HELPERS
# ============================================================

def load_json(filename):
    """
    Load a JSON benchmark file from the models directory.
    """

    path = MODELS_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark file not found: {path}"
        )

    with open(path, "r") as file:
        return json.load(file)


def normalize_model(
    model_name,
    family,
    data,
):
    """
    Convert a benchmark result into the common
    VITALIS benchmark format.
    """

    result = {
        "model": model_name,
        "family": family,

        "accuracy": round(
            float(data.get("accuracy", 0.0)) * 100,
            2
        ),

        "precision": round(
            float(data.get("precision", 0.0)) * 100,
            2
        ),

        "sensitivity": round(
            float(data.get("sensitivity", 0.0)) * 100,
            2
        ),

        "specificity": round(
            float(data.get("specificity", 0.0)) * 100,
            2
        ),

        "f1_score": round(
            float(data.get("f1_score", 0.0)) * 100,
            2
        ),

        "roc_auc": round(
            float(data.get("roc_auc", 0.0)) * 100,
            2
        ),
    }

    # --------------------------------------------------------
    # Optional quantum information
    # --------------------------------------------------------

    if "n_qubits" in data:
        result["n_qubits"] = int(
            data["n_qubits"]
        )

    if "reuploading_layers" in data:
        result["reuploading_layers"] = int(
            data["reuploading_layers"]
        )

    if "trainable_parameters" in data:
        result["trainable_parameters"] = int(
            data["trainable_parameters"]
        )

    # --------------------------------------------------------
    # Training time
    # --------------------------------------------------------

    if "training_time_seconds" in data:
        result["training_time_seconds"] = round(
            float(
                data["training_time_seconds"]
            ),
            4
        )

    # --------------------------------------------------------
    # QSVM timing information
    # --------------------------------------------------------

    if "train_state_time_seconds" in data:
        result["train_state_time_seconds"] = round(
            float(
                data["train_state_time_seconds"]
            ),
            4
        )

    if "test_state_time_seconds" in data:
        result["test_state_time_seconds"] = round(
            float(
                data["test_state_time_seconds"]
            ),
            4
        )

    if "train_kernel_time_seconds" in data:
        result["train_kernel_time_seconds"] = round(
            float(
                data["train_kernel_time_seconds"]
            ),
            4
        )

    if "test_kernel_time_seconds" in data:
        result["test_kernel_time_seconds"] = round(
            float(
                data["test_kernel_time_seconds"]
            ),
            4
        )

    return result


# ============================================================
# HEART BENCHMARK
# ============================================================

def get_heart_benchmark():

    classical_data = load_json(
        "benchmark.json"
    )

    quantum_data = load_json(
        "quantum_benchmark.json"
    )

    vqc_data = load_json(
        "heart_disease_vqc_benchmark.json"
    )

    models = []

    # --------------------------------------------------------
    # Classical models
    # --------------------------------------------------------

    classical_names = {
        "Logistic Regression":
            "Logistic Regression",

        "Random Forest":
            "Random Forest",

        "RBF SVM":
            "RBF SVM",
    }

    for source_name, display_name in classical_names.items():

        if source_name not in classical_data:
            continue

        models.append(
            normalize_model(
                model_name=display_name,
                family="classical",
                data=classical_data[source_name],
            )
        )

    # --------------------------------------------------------
    # Quantum Kernel SVM
    # --------------------------------------------------------

    models.append(
        normalize_model(
            model_name="Quantum Kernel SVM",
            family="quantum",
            data=quantum_data,
        )
    )

    # --------------------------------------------------------
    # VQC
    # --------------------------------------------------------

    models.append(
        normalize_model(
            model_name="Variational Quantum Classifier",
            family="quantum",
            data=vqc_data,
        )
    )

    return {
        "dataset": "UCI Cleveland Heart Disease",
        "models": models,
    }


# ============================================================
# BREAST CANCER BENCHMARK
# ============================================================

def get_breast_cancer_benchmark():

    classical_data = load_json(
        "breast_cancer_classical_benchmark.json"
    )

    quantum_data = load_json(
        "breast_cancer_quantum_benchmark.json"
    )

    vqc_data = load_json(
        "breast_cancer_vqc_benchmark.json"
    )

    models = []

    # --------------------------------------------------------
    # Classical models
    # --------------------------------------------------------

    classical_names = {
        "logistic_regression":
            "Logistic Regression",

        "random_forest":
            "Random Forest",

        "rbf_svm":
            "RBF SVM",
    }

    classical_models = classical_data.get(
        "models",
        {}
    )

    for source_name, display_name in classical_names.items():

        if source_name not in classical_models:
            continue

        models.append(
            normalize_model(
                model_name=display_name,
                family="classical",
                data=classical_models[source_name],
            )
        )

    # --------------------------------------------------------
    # Quantum Kernel SVM
    # --------------------------------------------------------

    models.append(
        normalize_model(
            model_name="Quantum Kernel SVM",
            family="quantum",
            data=quantum_data,
        )
    )

    # --------------------------------------------------------
    # VQC
    # --------------------------------------------------------

    models.append(
        normalize_model(
            model_name="Variational Quantum Classifier",
            family="quantum",
            data=vqc_data,
        )
    )

    return {
        "dataset":
            "Breast Cancer Wisconsin Diagnostic",

        "models": models,
    }


# ============================================================
# ALL BENCHMARKS
# ============================================================

def get_all_benchmarks():

    return {
        "heart": get_heart_benchmark(),
        "breast_cancer": get_breast_cancer_benchmark(),
    }