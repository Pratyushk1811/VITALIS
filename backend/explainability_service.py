from pathlib import Path

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
# HEART FEATURES
# ============================================================

HEART_FEATURES = [
    "cp",
    "thal",
    "thalach",
    "oldpeak",
    "ca",
    "age",
]


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


# ============================================================
# LOAD HEART CLASSICAL MODELS
# ============================================================

heart_logistic_regression = joblib.load(
    MODELS_DIR / "logistic_regression.joblib"
)

heart_random_forest = joblib.load(
    MODELS_DIR / "random_forest.joblib"
)

heart_rbf_svm = joblib.load(
    MODELS_DIR / "rbf_svm.joblib"
)

heart_classical_scaler = joblib.load(
    MODELS_DIR / "scaler.joblib"
)


# ============================================================
# LOAD BREAST CANCER CLASSICAL MODELS
# ============================================================

breast_logistic_regression = joblib.load(
    MODELS_DIR / "breast_cancer_logistic_regression.joblib"
)

breast_random_forest = joblib.load(
    MODELS_DIR / "breast_cancer_random_forest.joblib"
)

breast_rbf_svm = joblib.load(
    MODELS_DIR / "breast_cancer_rbf_svm.joblib"
)


# ============================================================
# LOAD HEART QUANTUM ARTIFACTS
# ============================================================

heart_qsvm = joblib.load(
    MODELS_DIR / "qsvm.joblib"
)

heart_quantum_scaler = joblib.load(
    MODELS_DIR / "quantum_scaler.joblib"
)

heart_quantum_training_data = joblib.load(
    MODELS_DIR / "quantum_training_data.joblib"
)


# ============================================================
# LOAD HEART VQC ARTIFACTS
# ============================================================

heart_vqc_scaler = joblib.load(
    MODELS_DIR / "heart_disease_vqc_scaler.joblib"
)

heart_vqc_weights = joblib.load(
    MODELS_DIR / "heart_disease_vqc_weights.joblib"
)


# ============================================================
# LOAD BREAST CANCER QUANTUM ARTIFACTS
# ============================================================

breast_qsvm = joblib.load(
    MODELS_DIR / "breast_cancer_qsvm.joblib"
)

breast_quantum_scaler = joblib.load(
    MODELS_DIR / "breast_cancer_quantum_scaler.joblib"
)

breast_quantum_training_states = joblib.load(
    MODELS_DIR / "breast_cancer_quantum_training_states.joblib"
)


# ============================================================
# LOAD BREAST CANCER VQC ARTIFACTS
# ============================================================

breast_vqc_scaler = joblib.load(
    MODELS_DIR / "breast_cancer_vqc_scaler.joblib"
)

breast_vqc_weights = joblib.load(
    MODELS_DIR / "breast_cancer_vqc_weights.joblib"
)


# ============================================================
# IMPORT TESTED HEART QUANTUM FUNCTIONS
# ============================================================

from backend.heart_pipeline import (
    quantum_kernel,
    predict_vqc,
)


# ============================================================
# GENERIC HELPERS
# ============================================================

def build_dataframe(
    patient,
    feature_names,
):
    """
    Convert a patient dictionary into a one-row DataFrame.
    """

    try:

        return pd.DataFrame(
            [[
                float(
                    patient[feature]
                )
                for feature in feature_names
            ]],
            columns=feature_names,
        )

    except (
        KeyError,
        TypeError,
        ValueError
    ) as error:

        raise ValueError(
            f"Invalid patient input: {error}"
        )


def normalize_importance(
    values,
):
    """
    Normalize absolute importance values
    so that they sum to 1.
    """

    values = np.abs(
        np.asarray(
            values,
            dtype=float,
        )
    )

    total = np.sum(values)

    if total == 0:

        return np.zeros_like(
            values
        )

    return values / total


def ranked_features(
    feature_names,
    importance_values,
    directions=None,
):
    """
    Convert feature importance values into
    a ranked explanation list.
    """

    importance_values = np.asarray(
        importance_values,
        dtype=float,
    )

    normalized = normalize_importance(
        importance_values
    )

    order = np.argsort(
        normalized
    )[::-1]

    results = []

    for index in order:

        item = {

            "feature":
                feature_names[index],

            "importance":
                round(
                    float(
                        normalized[index]
                    ),
                    4,
                ),
        }

        if directions is not None:

            item["direction"] = (
                directions[index]
            )

        results.append(
            item
        )

    return results


def get_class_1_probability(
    model,
    features,
):
    """
    Return class-1 probability.
    """

    if hasattr(
        model,
        "predict_proba",
    ):

        probabilities = (
            model.predict_proba(
                features
            )[0]
        )

        return float(
            probabilities[1]
        )

    prediction = int(
        model.predict(
            features
        )[0]
    )

    return float(
        prediction
    )


# ============================================================
# LOGISTIC REGRESSION EXPLANATION
# ============================================================

def explain_logistic_regression(
    model,
    transformed_features,
    feature_names,
):
    """
    Local Logistic Regression explanation.

    Contribution:

        coefficient × transformed feature
    """

    coefficients = np.asarray(
        model.coef_[0],
        dtype=float,
    )

    values = np.asarray(
        transformed_features[0],
        dtype=float,
    )

    contributions = (
        coefficients * values
    )

    directions = [

        "risk_increasing"

        if contribution > 0

        else "risk_decreasing"

        if contribution < 0

        else "neutral"

        for contribution
        in contributions
    ]

    return ranked_features(
        feature_names=feature_names,

        importance_values=np.abs(
            contributions
        ),

        directions=directions,
    )


# ============================================================
# RANDOM FOREST EXPLANATION
# ============================================================

def explain_random_forest(
    model,
    feature_names,
):
    """
    Random Forest explanation using
    built-in feature_importances_.
    """

    return ranked_features(
        feature_names=feature_names,

        importance_values=(
            model.feature_importances_
        ),
    )


# ============================================================
# GENERIC PERTURBATION EXPLANATION
# ============================================================

def perturbation_explanation(
    predict_probability,
    features,
    feature_names,
    perturbation_fraction=0.05,
):
    """
    Estimate local feature sensitivity.

    Each feature is increased slightly.
    The change in class-1 probability is measured.

    This is a local sensitivity measure, not
    a causal explanation.
    """

    base_probability = (
        predict_probability(
            features
        )
    )

    base_values = (
        features.iloc[0]
        .to_numpy(
            dtype=float
        )
    )

    sensitivities = []
    directions = []

    for index in range(
        len(feature_names)
    ):

        value = base_values[index]

        delta = (
            abs(value)
            * perturbation_fraction
        )

        if delta == 0:

            delta = (
                perturbation_fraction
            )

        perturbed_values = (
            base_values.copy()
        )

        perturbed_values[index] = (
            value + delta
        )

        perturbed_features = pd.DataFrame(
            [perturbed_values],
            columns=feature_names,
        )

        perturbed_probability = (
            predict_probability(
                perturbed_features
            )
        )

        change = (
            perturbed_probability
            - base_probability
        )

        sensitivities.append(
            abs(change)
        )

        if change > 0:

            directions.append(
                "risk_increasing"
            )

        elif change < 0:

            directions.append(
                "risk_decreasing"
            )

        else:

            directions.append(
                "neutral"
            )

    return ranked_features(
        feature_names=feature_names,

        importance_values=sensitivities,

        directions=directions,
    )


# ============================================================
# HEART CLASSICAL EXPLANATIONS
# ============================================================

def explain_heart_classical(
    features,
):

    scaled_features = (
        heart_classical_scaler.transform(
            features
        )
    )

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    logistic = (
        explain_logistic_regression(
            model=heart_logistic_regression,

            transformed_features=(
                scaled_features
            ),

            feature_names=HEART_FEATURES,
        )
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    random_forest = (
        explain_random_forest(
            model=heart_random_forest,

            feature_names=HEART_FEATURES,
        )
    )

    # --------------------------------------------------------
    # RBF SVM
    # --------------------------------------------------------

    def svm_probability(
        input_features,
    ):

        scaled = (
            heart_classical_scaler.transform(
                input_features
            )
        )

        return get_class_1_probability(
            heart_rbf_svm,
            scaled,
        )

    svm = perturbation_explanation(
        predict_probability=svm_probability,

        features=features,

        feature_names=HEART_FEATURES,
    )

    return {

        "logistic_regression":
            logistic,

        "random_forest":
            random_forest,

        "rbf_svm":
            svm,
    }


# ============================================================
# BREAST CANCER CLASSICAL EXPLANATIONS
# ============================================================

def explain_breast_cancer_classical(
    features,
):

    # --------------------------------------------------------
    # Logistic Regression Pipeline
    # --------------------------------------------------------

    logistic_scaler = (
        breast_logistic_regression[
            "scaler"
        ]
    )

    logistic_model = (
        breast_logistic_regression[
            "model"
        ]
    )

    raw_values = features.to_numpy(
        dtype=float
    )

    transformed_features = (
        logistic_scaler.transform(
            raw_values
        )
    )

    logistic = (
        explain_logistic_regression(

            model=logistic_model,

            transformed_features=(
                transformed_features
            ),

            feature_names=(
                BREAST_CANCER_FEATURES
            ),
        )
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    random_forest = (
        explain_random_forest(

            model=breast_random_forest,

            feature_names=(
                BREAST_CANCER_FEATURES
            ),
        )
    )

    # --------------------------------------------------------
    # RBF SVM Pipeline
    # --------------------------------------------------------

    def svm_probability(
        input_features,
    ):

        raw_values = (
            input_features.to_numpy(
                dtype=float
            )
        )

        return get_class_1_probability(
            breast_rbf_svm,
            raw_values,
        )

    svm = perturbation_explanation(

        predict_probability=(
            svm_probability
        ),

        features=features,

        feature_names=(
            BREAST_CANCER_FEATURES
        ),
    )

    return {

        "logistic_regression":
            logistic,

        "random_forest":
            random_forest,

        "rbf_svm":
            svm,
    }


# ============================================================
# HEART QSVM PROBABILITY
# ============================================================

def heart_qsvm_probability(
    features,
):
    """
    Calculate heart QSVM class-1 probability.

    Uses the same quantum-kernel implementation
    as heart_pipeline.py.
    """

    scaled_features = (
        heart_quantum_scaler.transform(
            features
        )
    )

    patient_features = (
        scaled_features[0]
    )

    kernel_values = [

        quantum_kernel(
            patient_features,
            training_data,
        )

        for training_data
        in heart_quantum_training_data
    ]

    kernel_vector = np.asarray(
        kernel_values,
        dtype=float,
    ).reshape(
        1,
        -1,
    )

    if hasattr(
        heart_qsvm,
        "predict_proba",
    ):

        probabilities = (
            heart_qsvm.predict_proba(
                kernel_vector
            )[0]
        )

        return float(
            probabilities[1]
        )

    prediction = int(
        heart_qsvm.predict(
            kernel_vector
        )[0]
    )

    return float(
        prediction
    )


# ============================================================
# HEART VQC PROBABILITY
# ============================================================

def heart_vqc_probability(
    features,
):
    """
    Calculate heart VQC class-1 probability.
    """

    result = predict_vqc(
        features
    )

    return float(
        result[
            "class_1_probability"
        ]
    )


# ============================================================
# HEART QUANTUM EXPLANATIONS
# ============================================================

def explain_heart_quantum(
    features,
):

    qsvm = perturbation_explanation(

        predict_probability=(
            heart_qsvm_probability
        ),

        features=features,

        feature_names=HEART_FEATURES,
    )

    vqc = perturbation_explanation(

        predict_probability=(
            heart_vqc_probability
        ),

        features=features,

        feature_names=HEART_FEATURES,
    )

    return {

        "qsvm":
            qsvm,

        "vqc":
            vqc,
    }


# ============================================================
# BREAST CANCER QUANTUM CONSTANTS
# ============================================================

BREAST_N_FEATURES = 30
BREAST_N_QUBITS = 6
BREAST_N_LAYERS = 5


# ============================================================
# BREAST CANCER QUANTUM DEVICE
# ============================================================

breast_quantum_device = qml.device(
    "default.qubit",
    wires=BREAST_N_QUBITS,
)


# ============================================================
# BREAST CANCER QUANTUM FEATURE MAP
# ============================================================

@qml.qnode(
    breast_quantum_device
)
def breast_quantum_feature_map(
    x,
):
    """
    Same five-layer data-reuploading
    feature map used by the breast-cancer
    quantum model.

    30 features:

        Layer 0 -> x[0:6]
        Layer 1 -> x[6:12]
        Layer 2 -> x[12:18]
        Layer 3 -> x[18:24]
        Layer 4 -> x[24:30]
    """

    for layer in range(
        BREAST_N_LAYERS
    ):

        start = (
            layer
            * BREAST_N_QUBITS
        )

        # ----------------------------------------------------
        # Data encoding
        # ----------------------------------------------------

        for qubit in range(
            BREAST_N_QUBITS
        ):

            qml.RY(
                x[start + qubit],
                wires=qubit
            )

        # ----------------------------------------------------
        # Entanglement
        # ----------------------------------------------------

        for qubit in range(
            BREAST_N_QUBITS - 1
        ):

            qml.CNOT(
                wires=[
                    qubit,
                    qubit + 1
                ]
            )

    return qml.state()


# ============================================================
# BREAST CANCER QUANTUM STATE
# ============================================================

def breast_quantum_state(
    x,
):
    """
    Generate a six-qubit state from
    the complete 30-feature input.
    """

    return breast_quantum_feature_map(
        x
    )


# ============================================================
# BREAST CANCER QSVM PROBABILITY
# ============================================================

def breast_qsvm_probability(
    features,
):
    """
    Calculate breast-cancer QSVM class-1
    probability.

    The saved training states are already
    six-qubit state vectors.

    Therefore:

        patient features
             ↓
        quantum feature map
             ↓
        patient state
             ↓
        overlap with saved states
             ↓
        kernel vector
             ↓
        QSVM
    """

    raw_values = features.to_numpy(
        dtype=float
    )

    scaled_features = (
        breast_quantum_scaler.transform(
            raw_values
        )
    )

    patient_features = (
        scaled_features[0]
    )

    # --------------------------------------------------------
    # Generate patient state
    # --------------------------------------------------------

    patient_state = (
        breast_quantum_state(
            patient_features
        )
    )

    # --------------------------------------------------------
    # Compare patient state directly
    # against saved training states
    # --------------------------------------------------------

    kernel_values = []

    for training_state in (
        breast_quantum_training_states
    ):

        training_state = np.asarray(
            training_state,
            dtype=complex
        )

        overlap = np.vdot(
            patient_state,
            training_state,
        )

        kernel_value = float(
            np.abs(overlap) ** 2
        )

        kernel_values.append(
            kernel_value
        )

    kernel_vector = np.asarray(
        kernel_values,
        dtype=float,
    ).reshape(
        1,
        -1,
    )

    # --------------------------------------------------------
    # QSVM prediction
    # --------------------------------------------------------

    if hasattr(
        breast_qsvm,
        "predict_proba",
    ):

        probabilities = (
            breast_qsvm.predict_proba(
                kernel_vector
            )[0]
        )

        return float(
            probabilities[1]
        )

    prediction = int(
        breast_qsvm.predict(
            kernel_vector
        )[0]
    )

    return float(
        prediction
    )


# ============================================================
# BREAST CANCER VQC
# ============================================================

BREAST_VQC_N_LAYERS = 5
BREAST_VQC_N_QUBITS = 6


breast_vqc_device = qml.device(
    "default.qubit",
    wires=BREAST_VQC_N_QUBITS,
)


# ============================================================
# BREAST CANCER VQC CIRCUIT
# ============================================================

@qml.qnode(
    breast_vqc_device
)
def breast_vqc_circuit(
    x,
    weights,
):
    """
    Breast-cancer VQC.

    Architecture:

        5 layers × 12 trainable parameters

    Each layer:

        6 data-encoding RY gates
        6 trainable RY gates
        6 trainable RZ gates
        5 CNOT gates

    The data encoding uses all 30 features:

        Layer 0 -> x[0:6]
        Layer 1 -> x[6:12]
        Layer 2 -> x[12:18]
        Layer 3 -> x[18:24]
        Layer 4 -> x[24:30]
    """

    for layer in range(
        BREAST_VQC_N_LAYERS
    ):

        start = (
            layer
            * BREAST_VQC_N_QUBITS
        )

        # ----------------------------------------------------
        # Data encoding
        # ----------------------------------------------------

        for qubit in range(
            BREAST_VQC_N_QUBITS
        ):

            qml.RY(
                x[start + qubit],
                wires=qubit
            )

        # ----------------------------------------------------
        # Trainable rotations
        # ----------------------------------------------------

        for qubit in range(
            BREAST_VQC_N_QUBITS
        ):

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
                    qubit
                    + BREAST_VQC_N_QUBITS
                ],
                wires=qubit
            )

        # ----------------------------------------------------
        # Entanglement
        # ----------------------------------------------------

        for qubit in range(
            BREAST_VQC_N_QUBITS - 1
        ):

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
# BREAST CANCER VQC PROBABILITY
# ============================================================

def breast_vqc_probability(
    features,
):
    """
    Calculate breast-cancer VQC
    class-1 probability.
    """

    raw_values = features.to_numpy(
        dtype=float
    )

    scaled_features = (
        breast_vqc_scaler.transform(
            raw_values
        )
    )

    patient_features = (
        scaled_features[0]
    )

    expectation = float(
        breast_vqc_circuit(
            patient_features,
            breast_vqc_weights,
        )
    )

    class_1_probability = (
        expectation + 1.0
    ) / 2.0

    return float(
        np.clip(
            class_1_probability,
            0.0,
            1.0,
        )
    )


# ============================================================
# BREAST CANCER QUANTUM EXPLANATIONS
# ============================================================

def explain_breast_cancer_quantum(
    features,
):

    qsvm = perturbation_explanation(

        predict_probability=(
            breast_qsvm_probability
        ),

        features=features,

        feature_names=(
            BREAST_CANCER_FEATURES
        ),
    )

    vqc = perturbation_explanation(

        predict_probability=(
            breast_vqc_probability
        ),

        features=features,

        feature_names=(
            BREAST_CANCER_FEATURES
        ),
    )

    return {

        "qsvm":
            qsvm,

        "vqc":
            vqc,
    }


# ============================================================
# PUBLIC HEART EXPLANATION
# ============================================================

def explain_heart(
    patient,
):
    """
    Generate complete heart-disease
    explainability response.
    """

    features = build_dataframe(
        patient,
        HEART_FEATURES,
    )

    classical = (
        explain_heart_classical(
            features
        )
    )

    quantum = (
        explain_heart_quantum(
            features
        )
    )

    return {

        "disease":
            "heart",

        "feature_count":
            len(
                HEART_FEATURES
            ),

        "classical":
            classical,

        "quantum":
            quantum,

        "methodology": {

            "logistic_regression":
                "Coefficient-based local contribution",

            "random_forest":
                "Built-in feature importance",

            "rbf_svm":
                "Local perturbation sensitivity",

            "qsvm":
                "Local quantum-kernel perturbation sensitivity",

            "vqc":
                "Local variational-circuit perturbation sensitivity",
        },
    }


# ============================================================
# PUBLIC BREAST CANCER EXPLANATION
# ============================================================

def explain_breast_cancer(
    patient,
):
    """
    Generate complete breast-cancer
    explainability response.
    """

    features = build_dataframe(
        patient,
        BREAST_CANCER_FEATURES,
    )

    classical = (
        explain_breast_cancer_classical(
            features
        )
    )

    quantum = (
        explain_breast_cancer_quantum(
            features
        )
    )

    return {

        "disease":
            "breast_cancer",

        "feature_count":
            len(
                BREAST_CANCER_FEATURES
            ),

        "classical":
            classical,

        "quantum":
            quantum,

        "methodology": {

            "logistic_regression":
                "Coefficient-based local contribution",

            "random_forest":
                "Built-in feature importance",

            "rbf_svm":
                "Local perturbation sensitivity",

            "qsvm":
                "Local quantum-kernel perturbation sensitivity",

            "vqc":
                "Local variational-circuit perturbation sensitivity",
        },
    }