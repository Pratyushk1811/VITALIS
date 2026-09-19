from pathlib import Path

import joblib
import numpy as np
import pennylane as qml


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


# ============================================================
# BREAST CANCER FEATURES
# ============================================================

BREAST_CANCER_FEATURES = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",

    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",

    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]


N_FEATURES = 30
N_QUBITS = 6
N_LAYERS = 5


# ============================================================
# LOAD CLASSICAL MODELS
# ============================================================

logistic_regression = joblib.load(
    MODELS_DIR / "breast_cancer_logistic_regression.joblib"
)

random_forest = joblib.load(
    MODELS_DIR / "breast_cancer_random_forest.joblib"
)

rbf_svm = joblib.load(
    MODELS_DIR / "breast_cancer_rbf_svm.joblib"
)


# ============================================================
# LOAD QUANTUM MODEL ARTIFACTS
# ============================================================

qsvm = joblib.load(
    MODELS_DIR / "breast_cancer_qsvm.joblib"
)

quantum_scaler = joblib.load(
    MODELS_DIR / "breast_cancer_quantum_scaler.joblib"
)

quantum_training_states = joblib.load(
    MODELS_DIR / "breast_cancer_quantum_training_states.joblib"
)

vqc_scaler = joblib.load(
    MODELS_DIR / "breast_cancer_vqc_scaler.joblib"
)

vqc_weights = joblib.load(
    MODELS_DIR / "breast_cancer_vqc_weights.joblib"
)


# ============================================================
# QUANTUM DEVICE
# ============================================================

quantum_device = qml.device(
    "default.qubit",
    wires=N_QUBITS
)


# ============================================================
# BREAST CANCER QUANTUM FEATURE MAP
# ============================================================

@qml.qnode(quantum_device)
def breast_cancer_feature_map(x):
    """
    Five-layer, six-qubit quantum feature map.

    30 classical features are encoded as:

        Layer 0 -> features  0-5
        Layer 1 -> features  6-11
        Layer 2 -> features 12-17
        Layer 3 -> features 18-23
        Layer 4 -> features 24-29

    Each layer:
        1. Encodes six features using RY rotations.
        2. Applies nearest-neighbour CNOT entanglement.

    Returns the complete six-qubit state vector.
    """

    for layer in range(N_LAYERS):

        start = layer * N_QUBITS

        # ----------------------------------------------------
        # Data encoding
        # ----------------------------------------------------

        for qubit in range(N_QUBITS):

            qml.RY(
                x[start + qubit],
                wires=qubit
            )

        # ----------------------------------------------------
        # Entanglement
        # ----------------------------------------------------

        for qubit in range(N_QUBITS - 1):

            qml.CNOT(
                wires=[
                    qubit,
                    qubit + 1
                ]
            )

    return qml.state()


# ============================================================
# QUANTUM KERNEL BETWEEN TWO 30-FEATURE INPUTS
# ============================================================

def quantum_kernel(x1, x2):
    """
    Calculate:

        K(x1, x2) = |<phi(x1)|phi(x2)>|^2

    Both x1 and x2 must be 30-feature classical inputs.

    This function is useful when both inputs still need
    to pass through the quantum feature map.
    """

    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    state_1 = breast_cancer_feature_map(x1)
    state_2 = breast_cancer_feature_map(x2)

    overlap = np.vdot(
        state_1,
        state_2
    )

    return float(
        np.abs(overlap) ** 2
    )


# ============================================================
# QUANTUM KERNEL BETWEEN PATIENT AND SAVED TRAINING STATE
# ============================================================

def state_kernel(patient_state, training_state):
    """
    Calculate the quantum-kernel value directly between:

        patient_state
        training_state

    Both arguments must already be six-qubit state vectors.

    This is the correct operation for the saved breast-cancer
    training states, which are already 64-dimensional quantum
    state vectors.
    """

    patient_state = np.asarray(
        patient_state,
        dtype=complex
    )

    training_state = np.asarray(
        training_state,
        dtype=complex
    )

    overlap = np.vdot(
        patient_state,
        training_state
    )

    return float(
        np.abs(overlap) ** 2
    )


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_patient(patient):

    if not isinstance(patient, dict):

        raise TypeError(
            "Patient data must be a dictionary."
        )

    missing_features = [
        feature
        for feature in BREAST_CANCER_FEATURES
        if feature not in patient
    ]

    if missing_features:

        raise ValueError(
            "Missing breast cancer features: "
            f"{missing_features}"
        )

    for feature in BREAST_CANCER_FEATURES:

        value = patient[feature]

        if not isinstance(
            value,
            (int, float)
        ):

            raise TypeError(
                f"{feature} must be numeric."
            )

        if not np.isfinite(value):

            raise ValueError(
                f"{feature} must be a finite number."
            )


# ============================================================
# QSVM PREDICTION
# ============================================================

def predict_qsvm(features):
    """
    Generate a kernel vector between the new patient and
    every saved quantum training state.

    The saved training states are already six-qubit states,
    so they must NOT be passed through the feature map again.
    """

    # --------------------------------------------------------
    # Same scaler used during QSVM training
    # --------------------------------------------------------

    scaled_features = quantum_scaler.transform(
        features
    )

    patient_features = scaled_features[0]

    # --------------------------------------------------------
    # Generate patient's quantum state
    # --------------------------------------------------------

    patient_state = breast_cancer_feature_map(
        patient_features
    )

    # --------------------------------------------------------
    # Kernel against saved training states
    # --------------------------------------------------------

    kernel_values = []

    for training_state in quantum_training_states:

        kernel_value = state_kernel(
            patient_state,
            training_state
        )

        kernel_values.append(
            kernel_value
        )

    kernel_vector = np.asarray(
        kernel_values,
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

    probabilities = qsvm.predict_proba(
        kernel_vector
    )[0]

    return {
        "prediction": prediction,

        "class_0_probability": float(
            probabilities[0]
        ),

        "class_1_probability": float(
            probabilities[1]
        ),
    }


# ============================================================
# VQC CIRCUIT
# ============================================================

@qml.qnode(quantum_device)
def vqc_circuit(x, weights):
    """
    Five-layer variational quantum circuit.

    Each layer:

        1. Encodes six new input features.
        2. Applies trainable RY rotations.
        3. Applies trainable RZ rotations.
        4. Applies nearest-neighbour CNOT entanglement.

    30 features are therefore used across five layers.
    """

    for layer in range(N_LAYERS):

        start = layer * N_QUBITS

        # ----------------------------------------------------
        # Data encoding
        # ----------------------------------------------------

        for qubit in range(N_QUBITS):

            qml.RY(
                x[start + qubit],
                wires=qubit
            )

        # ----------------------------------------------------
        # Trainable rotations
        # ----------------------------------------------------

        for qubit in range(N_QUBITS):

            qml.RY(
                weights[
                    layer,
                    qubit
                ],
                wires=qubit
            )

            qml.RZ(
                weights[
                    layer,
                    qubit + N_QUBITS
                ],
                wires=qubit
            )

        # ----------------------------------------------------
        # Entanglement
        # ----------------------------------------------------

        for qubit in range(N_QUBITS - 1):

            qml.CNOT(
                wires=[
                    qubit,
                    qubit + 1
                ]
            )

    return qml.expval(
        qml.PauliZ(0)
    )


# ============================================================
# VQC PREDICTION
# ============================================================

def predict_vqc(features):

    # --------------------------------------------------------
    # Same scaler used during VQC training
    # --------------------------------------------------------

    scaled_features = vqc_scaler.transform(
        features
    )

    patient_features = scaled_features[0]

    # --------------------------------------------------------
    # Quantum circuit
    # --------------------------------------------------------

    expectation = float(
        vqc_circuit(
            patient_features,
            vqc_weights
        )
    )

    # --------------------------------------------------------
    # Convert expectation [-1, 1]
    # to class-1 probability [0, 1]
    # --------------------------------------------------------

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

        "class_0_probability":
            class_0_probability,

        "class_1_probability":
            class_1_probability,
    }


# ============================================================
# MAIN BREAST CANCER PREDICTION
# ============================================================

def predict_breast_cancer(patient):

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    validate_patient(
        patient
    )

    # --------------------------------------------------------
    # Create NumPy array in exact training order
    # --------------------------------------------------------

    features = np.asarray(
        [[
            float(
                patient[feature]
            )

            for feature
            in BREAST_CANCER_FEATURES
        ]],
        dtype=float
    )

    # ========================================================
    # CLASSICAL MODELS
    # ========================================================

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    lr_prediction = int(
        logistic_regression.predict(
            features
        )[0]
    )

    lr_probabilities = (
        logistic_regression.predict_proba(
            features
        )[0]
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # RBF SVM
    # --------------------------------------------------------

    svm_prediction = int(
        rbf_svm.predict(
            features
        )[0]
    )

    svm_probabilities = (
        rbf_svm.predict_proba(
            features
        )[0]
    )

    # ========================================================
    # QUANTUM MODELS
    # ========================================================

    qsvm_result = predict_qsvm(
        features
    )

    vqc_result = predict_vqc(
        features
    )

    # ========================================================
    # COLLECT MODEL RESULTS
    # ========================================================

    predictions = {

        "logistic_regression": {

            "prediction":
                lr_prediction,

            "class_0_probability":
                float(
                    lr_probabilities[0]
                ),

            "class_1_probability":
                float(
                    lr_probabilities[1]
                ),
        },

        "random_forest": {

            "prediction":
                rf_prediction,

            "class_0_probability":
                float(
                    rf_probabilities[0]
                ),

            "class_1_probability":
                float(
                    rf_probabilities[1]
                ),
        },

        "rbf_svm": {

            "prediction":
                svm_prediction,

            "class_0_probability":
                float(
                    svm_probabilities[0]
                ),

            "class_1_probability":
                float(
                    svm_probabilities[1]
                ),
        },

        "qsvm":
            qsvm_result,

        "vqc":
            vqc_result,
    }

    # ========================================================
    # CONSENSUS
    # ========================================================

    model_predictions = [
        result["prediction"]
        for result in predictions.values()
    ]

    class_0_count = (
        model_predictions.count(0)
    )

    class_1_count = (
        model_predictions.count(1)
    )

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
        agreeing_models
        / total_models
    ) * 100.0

    if agreeing_models == total_models:

        consensus_status = "unanimous"

    elif agreeing_models >= 3:

        consensus_status = "majority"

    else:

        consensus_status = "disagreement"

    # ========================================================
    # FINAL RESULT
    # ========================================================

    return {

        "disease":
            "breast_cancer",

        "prediction":
            unified_prediction,

        "prediction_label": (

            "Higher likelihood of malignant tumor"

            if unified_prediction == 1

            else

            "Lower likelihood of malignant tumor"
        ),

        "consensus": {

            "agreeing_models":
                agreeing_models,

            "total_models":
                total_models,

            "percentage":
                round(
                    consensus_percentage,
                    2
                ),

            "status":
                consensus_status,
        },

        "models":
            predictions,

        "features_used":
            BREAST_CANCER_FEATURES,
    }