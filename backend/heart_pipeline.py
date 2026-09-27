from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import pennylane as qml


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


# ============================================================
# HEART DISEASE CONFIGURATION
# ============================================================

SELECTED_FEATURES = [
    "cp",
    "thal",
    "thalach",
    "oldpeak",
    "ca",
    "age",
]

N_QUBITS = 6
N_VQC_LAYERS = 3


# ============================================================
# LOAD CLASSICAL MODELS
# ============================================================

logistic_regression = joblib.load(
    MODELS_DIR / "logistic_regression.joblib"
)

random_forest = joblib.load(
    MODELS_DIR / "random_forest.joblib"
)

rbf_svm = joblib.load(
    MODELS_DIR / "rbf_svm.joblib"
)

classical_scaler = joblib.load(
    MODELS_DIR / "scaler.joblib"
)


# ============================================================
# LOAD QSVM ARTIFACTS
# ============================================================

qsvm = joblib.load(
    MODELS_DIR / "qsvm.joblib"
)

quantum_scaler = joblib.load(
    MODELS_DIR / "quantum_scaler.joblib"
)

quantum_training_data = np.asarray(
    joblib.load(
        MODELS_DIR / "quantum_training_data.joblib"
    )
)

quantum_training_labels = joblib.load(
    MODELS_DIR / "quantum_training_labels.joblib"
)


# ============================================================
# LOAD VQC ARTIFACTS
# ============================================================

vqc_scaler = joblib.load(
    MODELS_DIR / "heart_disease_vqc_scaler.joblib"
)

vqc_weights = joblib.load(
    MODELS_DIR / "heart_disease_vqc_weights.joblib"
)


# ============================================================
# QUANTUM DEVICE
# ============================================================

quantum_device = qml.device(
    "default.qubit",
    wires=N_QUBITS
)


# ============================================================
# QUANTUM FEATURE MAP
# ============================================================

def quantum_feature_map(x):
    """
    Encode six heart-disease features into six qubits.
    """

    for qubit in range(N_QUBITS):
        qml.RY(
            x[qubit],
            wires=qubit
        )

    for qubit in range(N_QUBITS - 1):
        qml.CNOT(
            wires=[qubit, qubit + 1]
        )


# ============================================================
# QUANTUM STATE
# ============================================================

@qml.qnode(quantum_device)
def quantum_state(x):
    quantum_feature_map(x)
    return qml.state()
# ============================================================
# QUANTUM KERNEL
# ============================================================

def quantum_kernel(x1, x2):
    """
    Fidelity-based quantum kernel.

    Kept for explainability/backward compatibility.
    QSVM prediction itself uses the optimized cached-state path.
    """

    state_1 = quantum_state(x1)
    state_2 = quantum_state(x2)

    overlap = np.vdot(
        state_1,
        state_2
    )

    return float(
        np.abs(overlap) ** 2
    )

# ============================================================
# CACHED TRAINING QUANTUM STATES
# ============================================================

_quantum_training_states = None


def get_quantum_training_states():
    """
    Compute the quantum states of the saved QSVM training
    samples only once, then reuse them for future predictions.
    """

    global _quantum_training_states

    if _quantum_training_states is None:

        print(
            "[VITALIS] Building cached quantum training states..."
        )

        start = time.perf_counter()

        states = [
            quantum_state(sample)
            for sample in quantum_training_data
        ]

        _quantum_training_states = np.asarray(
            states,
            dtype=complex
        )

        elapsed = (
            time.perf_counter() - start
        )

        print(
            "[VITALIS] Cached "
            f"{len(_quantum_training_states)} quantum states "
            f"in {elapsed:.2f}s"
        )

    return _quantum_training_states


# ============================================================
# QSVM PREDICTION
# ============================================================

def predict_qsvm(features):
    """
    Fast QSVM inference.

    The patient's quantum state is calculated once.
    Training quantum states are cached and reused.
    """

    scaled_features = quantum_scaler.transform(
        features
    )

    patient_features = scaled_features[0]

    # --------------------------------------------------------
    # Patient quantum state: ONLY ONE quantum simulation
    # --------------------------------------------------------

    patient_state = quantum_state(
        patient_features
    )

    # --------------------------------------------------------
    # Retrieve cached training states
    # --------------------------------------------------------

    training_states = get_quantum_training_states()

    # --------------------------------------------------------
    # Vectorized fidelity calculation
    #
    # K(x, xi) = |<phi(x) | phi(xi)>|^2
    # --------------------------------------------------------

    overlaps = (
        training_states.conj() @ patient_state
    )

    kernel_vector = (
        np.abs(overlaps) ** 2
    )

    kernel_vector = np.asarray(
        kernel_vector,
        dtype=float
    ).reshape(1, -1)

    # --------------------------------------------------------
    # QSVM prediction
    # --------------------------------------------------------

    prediction = int(
        qsvm.predict(
            kernel_vector
        )[0]
    )

    result = {
        "prediction": prediction
    }

    # --------------------------------------------------------
    # Probability / decision score
    # --------------------------------------------------------

    if hasattr(qsvm, "predict_proba"):

        probabilities = qsvm.predict_proba(
            kernel_vector
        )[0]

        result["class_0_probability"] = float(
            probabilities[0]
        )

        result["class_1_probability"] = float(
            probabilities[1]
        )

    elif hasattr(qsvm, "decision_function"):

        decision = qsvm.decision_function(
            kernel_vector
        )

        result["decision_score"] = float(
            np.asarray(decision).reshape(-1)[0]
        )

    return result


# ============================================================
# VQC CIRCUIT
# ============================================================

@qml.qnode(quantum_device)
def vqc_circuit(x, weights):

    for layer in range(N_VQC_LAYERS):

        # ----------------------------------------------------
        # Data re-uploading
        # ----------------------------------------------------

        for qubit in range(N_QUBITS):

            qml.RY(
                x[qubit],
                wires=qubit
            )

        # ----------------------------------------------------
        # Trainable rotations
        # ----------------------------------------------------

        for qubit in range(N_QUBITS):

            qml.RY(
                weights[layer, qubit],
                wires=qubit
            )

            qml.RZ(
                weights[layer, qubit + N_QUBITS],
                wires=qubit
            )

        # ----------------------------------------------------
        # Entanglement
        # ----------------------------------------------------

        for qubit in range(N_QUBITS - 1):

            qml.CNOT(
                wires=[qubit, qubit + 1]
            )

    return qml.expval(
        qml.PauliZ(0)
    )


# ============================================================
# VQC PREDICTION
# ============================================================

def predict_vqc(features):

    scaled_features = vqc_scaler.transform(
        features
    )

    expectation = float(
        vqc_circuit(
            scaled_features[0],
            vqc_weights
        )
    )

    class_1_probability = (
        expectation + 1.0
    ) / 2.0

    class_1_probability = float(
        np.clip(
            class_1_probability,
            0.0,
            1.0
        )
    )

    class_0_probability = (
        1.0 - class_1_probability
    )

    prediction = int(
        class_1_probability >= 0.5
    )

    return {
        "prediction": prediction,
        "class_0_probability": class_0_probability,
        "class_1_probability": class_1_probability,
    }


# ============================================================
# MAIN HEART PREDICTION FUNCTION
# ============================================================

def predict_heart(patient):

    total_start = time.perf_counter()

    # ========================================================
    # BUILD FEATURE DATAFRAME
    # ========================================================

    try:

        features = pd.DataFrame(
            [[
                float(patient[feature])
                for feature in SELECTED_FEATURES
            ]],
            columns=SELECTED_FEATURES,
        )

    except (KeyError, TypeError, ValueError) as error:

        raise ValueError(
            f"Invalid heart disease input: {error}"
        )

    # ========================================================
    # CLASSICAL PREPROCESSING
    # ========================================================

    scaled_features = classical_scaler.transform(
        features
    )

    # ========================================================
    # LOGISTIC REGRESSION
    # ========================================================

    start = time.perf_counter()

    lr_prediction = int(
        logistic_regression.predict(
            scaled_features
        )[0]
    )

    lr_probabilities = (
        logistic_regression.predict_proba(
            scaled_features
        )[0]
    )

    print(
        "[VITALIS] Logistic Regression: "
        f"{(time.perf_counter() - start) * 1000:.2f} ms"
    )

    # ========================================================
    # RANDOM FOREST
    # ========================================================

    start = time.perf_counter()

    rf_prediction = int(
        random_forest.predict(
            features
        )[0]
    )

    rf_probabilities = (
        random_forest.predict_proba(
            features
        )[0]
    )

    print(
        "[VITALIS] Random Forest: "
        f"{(time.perf_counter() - start) * 1000:.2f} ms"
    )

    # ========================================================
    # RBF SVM
    # ========================================================

    start = time.perf_counter()

    svm_prediction = int(
        rbf_svm.predict(
            scaled_features
        )[0]
    )

    svm_probabilities = (
        rbf_svm.predict_proba(
            scaled_features
        )[0]
    )

    print(
        "[VITALIS] RBF SVM: "
        f"{(time.perf_counter() - start) * 1000:.2f} ms"
    )

    # ========================================================
    # QSVM
    # ========================================================

    start = time.perf_counter()

    qsvm_result = predict_qsvm(
        features
    )

    print(
        "[VITALIS] QSVM: "
        f"{(time.perf_counter() - start) * 1000:.2f} ms"
    )

    # ========================================================
    # VQC
    # ========================================================

    start = time.perf_counter()

    vqc_result = predict_vqc(
        features
    )

    print(
        "[VITALIS] VQC: "
        f"{(time.perf_counter() - start) * 1000:.2f} ms"
    )

    # ========================================================
    # COLLECT MODEL RESULTS
    # ========================================================

    predictions = {

        "logistic_regression": {
            "prediction": lr_prediction,
            "class_0_probability": float(
                lr_probabilities[0]
            ),
            "class_1_probability": float(
                lr_probabilities[1]
            ),
        },

        "random_forest": {
            "prediction": rf_prediction,
            "class_0_probability": float(
                rf_probabilities[0]
            ),
            "class_1_probability": float(
                rf_probabilities[1]
            ),
        },

        "rbf_svm": {
            "prediction": svm_prediction,
            "class_0_probability": float(
                svm_probabilities[0]
            ),
            "class_1_probability": float(
                svm_probabilities[1]
            ),
        },

        "qsvm": qsvm_result,

        "vqc": vqc_result,
    }

    # ========================================================
    # MODEL CONSENSUS
    # ========================================================

    model_predictions = [
        result["prediction"]
        for result in predictions.values()
    ]

    class_0_count = model_predictions.count(0)
    class_1_count = model_predictions.count(1)

    if class_1_count > class_0_count:

        unified_prediction = 1
        agreeing_models = class_1_count

    else:

        unified_prediction = 0
        agreeing_models = class_0_count

    total_models = len(
        model_predictions
    )

    consensus_percentage = (
        agreeing_models / total_models
    ) * 100.0

    # ========================================================
    # CONSENSUS STATUS
    # ========================================================

    if agreeing_models == total_models:

        consensus_status = "unanimous"

    elif agreeing_models >= 3:

        consensus_status = "majority"

    else:

        consensus_status = "disagreement"

    # ========================================================
    # TOTAL TIMING
    # ========================================================

    total_time_ms = (
        time.perf_counter() - total_start
    ) * 1000

    print(
        "[VITALIS] TOTAL HEART INFERENCE: "
        f"{total_time_ms:.2f} ms"
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    return {

        "disease": "heart",

        "prediction": unified_prediction,

        "prediction_label": (
            "Higher likelihood of cardiovascular disease"
            if unified_prediction == 1
            else "Lower likelihood of cardiovascular disease"
        ),

        "consensus": {
            "agreeing_models": agreeing_models,
            "total_models": total_models,
            "percentage": round(
                consensus_percentage,
                2
            ),
            "status": consensus_status,
        },

        "models": predictions,

        "features_used": SELECTED_FEATURES,
    }