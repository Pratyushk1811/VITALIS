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


def _metric(data, *names):
    """
    Return the first available metric from the supplied names.
    """

    for name in names:
        if name in data:
            return float(data[name])

    return 0.0


def normalize_model(
    model_name,
    family,
    data,
):
    """
    Convert a benchmark result into the common
    VITALIS benchmark format.

    Input metrics are expected to be in the range 0-1.
    Output metrics are percentages.
    """

    result = {
        "model": model_name,
        "family": family,

        "accuracy": round(
            _metric(data, "accuracy") * 100,
            2
        ),

        "precision": round(
            _metric(data, "precision") * 100,
            2
        ),

        "sensitivity": round(
            _metric(
                data,
                "sensitivity",
                "recall"
            ) * 100,
            2
        ),

        "specificity": round(
            _metric(data, "specificity") * 100,
            2
        ),

        "f1_score": round(
            _metric(
                data,
                "f1_score",
                "f1"
            ) * 100,
            2
        ),

        "roc_auc": round(
            _metric(
                data,
                "roc_auc",
                "roc_auc_score"
            ) * 100,
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
    # Training / evaluation timing
    # --------------------------------------------------------

    if "training_time_seconds" in data:
        result["training_time_seconds"] = round(
            float(
                data["training_time_seconds"]
            ),
            4
        )

    if "evaluation_time_seconds" in data:
        result["evaluation_time_seconds"] = round(
            float(
                data["evaluation_time_seconds"]
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
# GENERIC SESSION BENCHMARK
# ============================================================

def get_session_benchmark(training_result):
    """
    Build a unified benchmark from the result of the
    generic VITALIS training pipeline.

    Expected pipeline structure:

        {
            "classical": {
                "metrics": {
                    ...
                }
            },

            "quantum": {
                "qsvm": {
                    "metrics": {...}
                },

                "vqc": {
                    "metrics": {...}
                }
            }
        }

    This function is dataset-independent.
    """

    if not isinstance(training_result, dict):
        raise ValueError(
            "training_result must be a dictionary."
        )

    models = []

    # ========================================================
    # CLASSICAL MODELS
    # ========================================================

    classical = training_result.get(
        "classical",
        {}
    )

    classical_metrics = classical.get(
        "metrics",
        {}
    )

    classical_display_names = {
        "logistic_regression":
            "Logistic Regression",

        "random_forest":
            "Random Forest",

        "rbf_svm":
            "RBF SVM",

        # Also support display-name keys.
        "Logistic Regression":
            "Logistic Regression",

        "Random Forest":
            "Random Forest",

        "RBF SVM":
            "RBF SVM",
    }

    for source_name, display_name in classical_display_names.items():

        if source_name not in classical_metrics:
            continue

        models.append(
            normalize_model(
                model_name=display_name,
                family="classical",
                data=classical_metrics[source_name],
            )
        )

    # ========================================================
    # QUANTUM KERNEL SVM
    # ========================================================

    quantum = training_result.get(
        "quantum",
        {}
    )

    qsvm = quantum.get(
        "qsvm",
        {}
    )

    qsvm_metrics = qsvm.get(
        "metrics",
        {}
    )

    if qsvm_metrics:
        models.append(
            normalize_model(
                model_name="Quantum Kernel SVM",
                family="quantum",
                data=qsvm_metrics,
            )
        )

    # ========================================================
    # VARIATIONAL QUANTUM CLASSIFIER
    # ========================================================

    vqc = quantum.get(
        "vqc",
        {}
    )

    vqc_metrics = vqc.get(
        "metrics",
        {}
    )

    if vqc_metrics:
        models.append(
            normalize_model(
                model_name="Variational Quantum Classifier",
                family="quantum",
                data=vqc_metrics,
            )
        )

    # ========================================================
    # DATASET INFORMATION
    # ========================================================

    dataset_info = training_result.get(
        "dataset",
        {}
    )

    if isinstance(dataset_info, dict):

        dataset_name = (
            dataset_info.get("name")
            or dataset_info.get("dataset_name")
            or "Uploaded Dataset"
        )

        target_column = dataset_info.get(
            "target_column"
        )

        n_samples = dataset_info.get(
            "rows"
        )

        n_features = dataset_info.get(
            "columns"
        )

    else:
        dataset_name = (
            str(dataset_info)
            if dataset_info
            else "Uploaded Dataset"
        )

        target_column = training_result.get(
            "target_column"
        )

        n_samples = None
        n_features = None

    # ========================================================
    # FALLBACK DATASET INFORMATION
    # ========================================================

    if target_column is None:
        target_column = training_result.get(
            "target_column"
        )

    # ========================================================
    # FINAL BENCHMARK
    # ========================================================

    benchmark = {
        "dataset": dataset_name,
        "target_column": target_column,
        "models": models,
    }

    if n_samples is not None:
        benchmark["samples"] = int(
            n_samples
        )

    if n_features is not None:
        benchmark["features"] = int(
            n_features
        )

    return benchmark


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
# ALL LEGACY BENCHMARKS
# ============================================================

def get_all_benchmarks():

    return {
        "heart": get_heart_benchmark(),
        "breast_cancer": get_breast_cancer_benchmark(),
    }