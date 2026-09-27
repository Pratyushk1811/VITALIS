"""
VITALIS - Quantum Kernel SVM Training Engine

Supports:
    - Binary classification
    - Multiclass classification
    - Numeric quantum features
    - Quantum feature scaling
    - Precomputed quantum kernels
    - PennyLane state-based quantum feature map
    - Reusable QSVM artifacts

IMPORTANT:
    This module uses STANDARD NumPy for all sklearn-facing data.

    PennyLane is used only for the quantum circuit/kernel.
    This prevents PennyLane tensors from leaking into sklearn
    metrics such as precision_score().
"""

import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import pennylane as qml

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# -------------------------------------------------------------------
# Default configuration
# -------------------------------------------------------------------

DEFAULT_QUANTUM_FEATURES = [
    "cp",
    "thal",
    "thalach",
    "oldpeak",
    "ca",
    "age",
]

# Backward-compatible alias
SELECTED_FEATURES = DEFAULT_QUANTUM_FEATURES


# -------------------------------------------------------------------
# Conversion helper
# -------------------------------------------------------------------

def _to_numpy(value, dtype=None):
    """
    Convert PennyLane tensors / NumPy arrays / lists into
    ordinary NumPy arrays before sklearn sees them.
    """

    try:
        value = qml.math.toarray(value)
    except Exception:
        pass

    result = np.asarray(value)

    if dtype is not None:
        result = result.astype(dtype)

    return result


# -------------------------------------------------------------------
# Quantum kernel
# -------------------------------------------------------------------

def quantum_kernel(
    x1,
    x2,
    feature_map,
    n_qubits,
):
    """
    Compute the quantum kernel between two feature vectors.

    K(x1, x2) = |<psi(x1)|psi(x2)>|^2

    Public function kept for compatibility with backend.main.
    """

    x1 = _to_numpy(
        x1,
        dtype=float,
    )

    x2 = _to_numpy(
        x2,
        dtype=float,
    )

    state1 = feature_map(x1)
    state2 = feature_map(x2)

    # Convert quantum states to ordinary NumPy arrays
    state1 = _to_numpy(
        state1,
        dtype=complex,
    ).reshape(-1)

    state2 = _to_numpy(
        state2,
        dtype=complex,
    ).reshape(-1)

    # Quantum-state overlap
    overlap = np.vdot(
        state1,
        state2,
    )

    # Kernel value: |<psi(x1)|psi(x2)>|^2
    kernel_value = np.abs(
        overlap
    ) ** 2

    # Force a genuine Python scalar
    return float(
        np.asarray(
            kernel_value
        ).reshape(-1)[0]
    )



# -------------------------------------------------------------------
# Kernel matrix
# -------------------------------------------------------------------

def build_kernel_matrix(
    X1,
    X2,
    feature_map,
    n_qubits,
):
    """
    Build a quantum kernel matrix.

    K[i,j] = quantum_kernel(X1[i], X2[j])
    """

    X1 = _to_numpy(
        X1,
        dtype=float,
    )

    X2 = _to_numpy(
        X2,
        dtype=float,
    )

    matrix = np.zeros(
        (
            X1.shape[0],
            X2.shape[0],
        ),
        dtype=float,
    )

    total = (
        X1.shape[0]
        * X2.shape[0]
    )

    completed = 0
    start_time = time.time()

    for i in range(
        X1.shape[0]
    ):

        for j in range(
            X2.shape[0]
        ):

            matrix[i, j] = quantum_kernel(
                X1[i],
                X2[j],
                feature_map,
                n_qubits,
            )

            completed += 1

        elapsed = (
            time.time()
            - start_time
        )

        print(
            f"Completed {completed}/{total} "
            f"kernel evaluations "
            f"({elapsed:.1f}s)"
        )

    return matrix


# -------------------------------------------------------------------
# Binary specificity
# -------------------------------------------------------------------

def _calculate_specificity_binary(
    y_true,
    y_pred,
):
    """
    Calculate binary specificity.
    """

    y_true = _to_numpy(
        y_true,
        dtype=int,
    ).reshape(-1)

    y_pred = _to_numpy(
        y_pred,
        dtype=int,
    ).reshape(-1)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    denominator = tn + fp

    if denominator == 0:
        return 0.0

    return float(
        tn / denominator
    )


# -------------------------------------------------------------------
# Multiclass specificity
# -------------------------------------------------------------------

def _calculate_specificity_multiclass(
    y_true,
    y_pred,
    classes,
):
    """
    Calculate macro one-vs-rest specificity.
    """

    y_true = _to_numpy(
        y_true,
        dtype=int,
    ).reshape(-1)

    y_pred = _to_numpy(
        y_pred,
        dtype=int,
    ).reshape(-1)

    classes = _to_numpy(
        classes,
        dtype=int,
    ).reshape(-1)

    specificities = []

    for class_value in classes:

        true_binary = (
            y_true == class_value
        ).astype(int)

        pred_binary = (
            y_pred == class_value
        ).astype(int)

        cm = confusion_matrix(
            true_binary,
            pred_binary,
            labels=[0, 1],
        )

        tn, fp, fn, tp = cm.ravel()

        denominator = tn + fp

        if denominator == 0:
            continue

        specificities.append(
            float(
                tn / denominator
            )
        )

    if not specificities:
        return 0.0

    return float(
        np.mean(
            specificities
        )
    )


# -------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------

def _metrics(
    y_test,
    y_pred,
    y_prob,
    classes,
):
    """
    Calculate VITALIS benchmark metrics.

    Everything passed into sklearn is converted to ordinary
    NumPy arrays / Python integers first.

    This is the critical fix for:

        pos_label=tensor(1, requires_grad=True)
    """

    # ---------------------------------------------------------------
    # Convert everything AWAY from PennyLane tensors
    # ---------------------------------------------------------------

    y_test = _to_numpy(
        y_test,
        dtype=int,
    ).reshape(-1)

    y_pred = _to_numpy(
        y_pred,
        dtype=int,
    ).reshape(-1)

    classes = _to_numpy(
        classes,
        dtype=int,
    ).reshape(-1)

    if y_prob is not None:

        y_prob = _to_numpy(
            y_prob,
            dtype=float,
        )

    is_binary = (
        len(classes) == 2
    )

    # ---------------------------------------------------------------
    # Accuracy
    # ---------------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    # ---------------------------------------------------------------
    # Binary classification
    # ---------------------------------------------------------------

    if is_binary:

        # IMPORTANT:
        # Python int, NOT PennyLane tensor.
        positive_class = int(
            classes[1]
        )

        precision = precision_score(
            y_test,
            y_pred,
            pos_label=positive_class,
            zero_division=0,
        )

        sensitivity = recall_score(
            y_test,
            y_pred,
            pos_label=positive_class,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            y_pred,
            pos_label=positive_class,
            zero_division=0,
        )

        specificity = (
            _calculate_specificity_binary(
                y_test,
                y_pred,
            )
        )

    # ---------------------------------------------------------------
    # Multiclass classification
    # ---------------------------------------------------------------

    else:

        precision = precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )

        sensitivity = recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )

        specificity = (
            _calculate_specificity_multiclass(
                y_test,
                y_pred,
                classes,
            )
        )

    # ---------------------------------------------------------------
    # ROC-AUC
    # ---------------------------------------------------------------

    auc = None

    try:

        if y_prob is not None:

            probability_array = _to_numpy(
                y_prob,
                dtype=float,
            )

            if is_binary:

                if (
                    probability_array.ndim == 2
                    and probability_array.shape[1] >= 2
                ):

                    positive_probability = (
                        probability_array[:, 1]
                    )

                else:

                    positive_probability = (
                        probability_array.reshape(-1)
                    )

                auc = roc_auc_score(
                    y_test,
                    positive_probability,
                )

            else:

                if (
                    probability_array.ndim == 2
                    and probability_array.shape[1]
                    == len(classes)
                ):

                    auc = roc_auc_score(
                        y_test,
                        probability_array,
                        multi_class="ovr",
                        average="macro",
                    )

    except (
        ValueError,
        TypeError,
    ):

        auc = None

    # ---------------------------------------------------------------
    # Return
    # ---------------------------------------------------------------

    return {
        "accuracy": float(
            accuracy
        ),

        "precision": float(
            precision
        ),

        "sensitivity": float(
            sensitivity
        ),

        "specificity": float(
            specificity
        ),

        "f1": float(
            f1
        ),

        "roc_auc": (
            float(auc)
            if auc is not None
            else None
        ),

        "classification_type": (
            "binary"
            if is_binary
            else "multiclass"
        ),

        "num_classes": int(
            len(classes)
        ),
    }


# -------------------------------------------------------------------
# Quantum Kernel SVM training
# -------------------------------------------------------------------

def train_quantum(
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    quantum_features=None,
    n_qubits=None,
    test_size=0.20,
    random_state=42,
):
    """
    Train a Quantum Kernel SVM.

    Preferred integrated usage:

        train_quantum(
            X_train,
            y_train,
            X_test,
            y_test,
            quantum_features=...,
            n_qubits=...
        )

    Backward-compatible standalone usage:

        train_quantum(
            X,
            y,
            quantum_features=...,
            n_qubits=...
        )

    When an explicit X_test/y_test is supplied, NO additional
    train/test split is performed.
    """

    total_start_time = (
        time.time()
    )

    # ===============================================================
    # Convert training data
    # ===============================================================

    if not isinstance(
        X_train,
        pd.DataFrame,
    ):

        X_train = pd.DataFrame(
            X_train
        )

    X_train = X_train.copy()

    # ===============================================================
    # Explicit train/test mode
    # ===============================================================

    if (
        X_test is not None
        and y_test is not None
    ):

        if not isinstance(
            X_test,
            pd.DataFrame,
        ):

            X_test = pd.DataFrame(
                X_test
            )

        X_test = X_test.copy()

        y_train = pd.Series(
            y_train
        ).reset_index(
            drop=True
        )

        y_test = pd.Series(
            y_test
        ).reset_index(
            drop=True
        )

        X_train = (
            X_train.reset_index(
                drop=True
            )
        )

        X_test = (
            X_test.reset_index(
                drop=True
            )
        )

    # ===============================================================
    # Standalone mode
    # ===============================================================

    else:

        print(
            "\nNo explicit test set supplied."
        )

        print(
            "Creating standalone train/test split."
        )

        y_train = pd.Series(
            y_train
        ).reset_index(
            drop=True
        )

        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = train_test_split(
            X_train,
            y_train,
            test_size=test_size,
            random_state=random_state,
            stratify=y_train,
        )

        X_train = (
            X_train.reset_index(
                drop=True
            )
        )

        X_test = (
            X_test.reset_index(
                drop=True
            )
        )

        y_train = (
            y_train.reset_index(
                drop=True
            )
        )

        y_test = (
            y_test.reset_index(
                drop=True
            )
        )

    # ===============================================================
    # Quantum feature selection
    # ===============================================================

    if quantum_features is None:

        quantum_features = list(
            DEFAULT_QUANTUM_FEATURES
        )

    else:

        quantum_features = list(
            quantum_features
        )

    if len(
        quantum_features
    ) == 0:

        raise ValueError(
            "At least one quantum feature is required."
        )

    # ===============================================================
    # Validate features
    # ===============================================================

    missing_train = [
        feature
        for feature in quantum_features
        if feature not in X_train.columns
    ]

    missing_test = [
        feature
        for feature in quantum_features
        if feature not in X_test.columns
    ]

    if missing_train:

        raise ValueError(
            "Training data is missing quantum features: "
            f"{missing_train}"
        )

    if missing_test:

        raise ValueError(
            "Test data is missing quantum features: "
            f"{missing_test}"
        )

    X_train_quantum = (
        X_train[
            quantum_features
        ].copy()
    )

    X_test_quantum = (
        X_test[
            quantum_features
        ].copy()
    )

    # ===============================================================
    # Convert features to numeric
    # ===============================================================

    for feature in quantum_features:

        X_train_quantum[
            feature
        ] = pd.to_numeric(
            X_train_quantum[
                feature
            ],
            errors="coerce",
        )

        X_test_quantum[
            feature
        ] = pd.to_numeric(
            X_test_quantum[
                feature
            ],
            errors="coerce",
        )

    # ===============================================================
    # Training-only median imputation
    # ===============================================================

    training_medians = (
        X_train_quantum.median(
            numeric_only=True
        )
    )

    X_train_quantum = (
        X_train_quantum.fillna(
            training_medians
        )
    )

    X_test_quantum = (
        X_test_quantum.fillna(
            training_medians
        )
    )

    if X_train_quantum.isna().any().any():

        bad_columns = (
            X_train_quantum
            .columns[
                X_train_quantum
                .isna()
                .any()
            ]
            .tolist()
        )

        raise ValueError(
            "Quantum training features contain "
            f"unusable columns: {bad_columns}"
        )

    if X_test_quantum.isna().any().any():

        bad_columns = (
            X_test_quantum
            .columns[
                X_test_quantum
                .isna()
                .any()
            ]
            .tolist()
        )

        raise ValueError(
            "Quantum test features contain "
            f"unusable columns: {bad_columns}"
        )

    # ===============================================================
    # Validate dimensions
    # ===============================================================

    if len(X_train_quantum) != len(y_train):

        raise ValueError(
            "Training feature and target lengths do not match."
        )

    if len(X_test_quantum) != len(y_test):

        raise ValueError(
            "Test feature and target lengths do not match."
        )

    # ===============================================================
    # Encode targets
    # ===============================================================

    target_encoder = LabelEncoder()

    combined_targets = pd.concat(
        [
            pd.Series(y_train),
            pd.Series(y_test),
        ],
        ignore_index=True,
    )

    if combined_targets.isna().any():

        raise ValueError(
            "Quantum target contains missing values."
        )

    target_encoder.fit(
        combined_targets.astype(str)
    )

    y_train_encoded = (
        target_encoder.transform(
            pd.Series(
                y_train
            ).astype(str)
        )
        .astype(int)
    )

    y_test_encoded = (
        target_encoder.transform(
            pd.Series(
                y_test
            ).astype(str)
        )
        .astype(int)
    )

    classes = target_encoder.classes_

    num_classes = len(
        classes
    )

    if num_classes < 2:

        raise ValueError(
            "Quantum training requires at least two target classes."
        )

    print(
        "\nQuantum target encoding:"
    )

    for index, class_name in enumerate(
        classes
    ):

        print(
            f"  {class_name} -> {index}"
        )

    print(
        f"  Number of classes: {num_classes}"
    )

    # ===============================================================
    # Determine number of qubits
    # ===============================================================

    if n_qubits is None:

        n_qubits = len(
            quantum_features
        )

    else:

        n_qubits = int(
            n_qubits
        )

    if n_qubits <= 0:

        raise ValueError(
            "n_qubits must be greater than zero."
        )

    if n_qubits != len(
        quantum_features
    ):

        raise ValueError(
            "n_qubits must equal the number "
            "of quantum features for the current "
            "feature-map implementation."
        )

    # ===============================================================
    # Scale features to [0, pi]
    # ===============================================================

    scaler = MinMaxScaler(
        feature_range=(
            0.0,
            np.pi,
        )
    )

    X_train_scaled = (
        scaler.fit_transform(
            X_train_quantum
        )
    )

    X_test_scaled = (
        scaler.transform(
            X_test_quantum
        )
    )

    # ===============================================================
    # Quantum device
    # ===============================================================

    dev = qml.device(
        "default.qubit",
        wires=n_qubits,
    )

    # ===============================================================
    # Quantum feature map
    # ===============================================================

    @qml.qnode(dev)
    def feature_map(x):

        for i in range(
            n_qubits
        ):

            qml.RY(
                x[i],
                wires=i,
            )

        for i in range(
            n_qubits - 1
        ):

            qml.CNOT(
                wires=[
                    i,
                    i + 1,
                ]
            )

        return qml.state()

    # ===============================================================
    # TRAINING QUANTUM KERNEL
    # ===============================================================

    print(
        "\n"
        + "=" * 55
    )

    print(
        "BUILDING TRAINING QUANTUM KERNEL"
    )

    print(
        "=" * 55
    )

    kernel_train_start = (
        time.time()
    )

    K_train = build_kernel_matrix(
        X_train_scaled,
        X_train_scaled,
        feature_map,
        n_qubits,
    )

    training_kernel_time = (
        time.time()
        - kernel_train_start
    )

    # ===============================================================
    # TEST QUANTUM KERNEL
    # ===============================================================

    print(
        "\n"
        + "=" * 55
    )

    print(
        "BUILDING TEST QUANTUM KERNEL"
    )

    print(
        "=" * 55
    )

    kernel_test_start = (
        time.time()
    )

    K_test = build_kernel_matrix(
        X_test_scaled,
        X_train_scaled,
        feature_map,
        n_qubits,
    )

    test_kernel_time = (
        time.time()
        - kernel_test_start
    )

    # ===============================================================
    # TRAIN QSVM
    # ===============================================================

    print(
        "\n"
        + "=" * 55
    )

    print(
        "TRAINING QUANTUM KERNEL SVM"
    )

    print(
        "=" * 55
    )

    training_start = (
        time.time()
    )

    qsvm = SVC(
        kernel="precomputed",
        probability=True,
        random_state=random_state,
    )

    qsvm.fit(
        K_train,
        y_train_encoded,
    )

    training_time = (
        time.time()
        - training_start
    )

    # ===============================================================
    # EVALUATION
    # ===============================================================

    print(
        "\n"
        + "=" * 55
    )

    print(
        "EVALUATING QUANTUM KERNEL SVM"
    )

    print(
        "=" * 55
    )

    evaluation_start = (
        time.time()
    )

    y_pred_encoded = qsvm.predict(
        K_test
    )

    try:

        y_prob = qsvm.predict_proba(
            K_test
        )

    except Exception:

        y_prob = None

    evaluation_time = (
        time.time()
        - evaluation_start
    )

    # ===============================================================
    # FORCE ORDINARY NUMPY TYPES
    # ===============================================================

    y_pred_encoded = _to_numpy(
        y_pred_encoded,
        dtype=int,
    ).reshape(-1)

    y_test_encoded_array = _to_numpy(
        y_test_encoded,
        dtype=int,
    ).reshape(-1)

    y_train_encoded_array = _to_numpy(
        y_train_encoded,
        dtype=int,
    ).reshape(-1)

    if y_prob is not None:

        y_prob = _to_numpy(
            y_prob,
            dtype=float,
        )

    # ===============================================================
    # METRICS
    # ===============================================================

    metrics = _metrics(
        y_test_encoded_array,
        y_pred_encoded,
        y_prob,
        classes=np.arange(
            num_classes,
            dtype=int,
        ),
    )

    total_time = (
        time.time()
        - total_start_time
    )

    # ===============================================================
    # Decode predictions
    # ===============================================================

    y_pred_labels = (
        target_encoder.inverse_transform(
            y_pred_encoded
        )
    )

    y_test_labels = (
        target_encoder.inverse_transform(
            y_test_encoded_array
        )
    )

    # ===============================================================
    # RESULTS
    # ===============================================================

    print(
        "\n"
        + "=" * 55
    )

    print(
        "QUANTUM KERNEL SVM RESULTS"
    )

    print(
        "=" * 55
    )

    print(
        f"Training data:       "
        f"{len(X_train_quantum)}"
    )

    print(
        f"Test data:           "
        f"{len(X_test_quantum)}"
    )

    print(
        f"Quantum features:    "
        f"{len(quantum_features)}"
    )

    print(
        f"Qubits:              "
        f"{n_qubits}"
    )

    print(
        f"Classes:             "
        f"{classes.tolist()}"
    )

    print(
        f"Classification:      "
        f"{'binary' if num_classes == 2 else 'multiclass'}"
    )

    print(
        f"Accuracy:            "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision:           "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Sensitivity:         "
        f"{metrics['sensitivity']:.4f}"
    )

    print(
        f"Specificity:         "
        f"{metrics['specificity']:.4f}"
    )

    print(
        f"F1 Score:            "
        f"{metrics['f1']:.4f}"
    )

    if metrics["roc_auc"] is not None:

        print(
            f"ROC-AUC:             "
            f"{metrics['roc_auc']:.4f}"
        )

    print(
        f"Training kernel:     "
        f"{training_kernel_time:.2f}s"
    )

    print(
        f"Test kernel:         "
        f"{test_kernel_time:.2f}s"
    )

    print(
        f"QSVM training:       "
        f"{training_time:.2f}s"
    )

    print(
        f"Evaluation:          "
        f"{evaluation_time:.2f}s"
    )

    print(
        f"Total time:          "
        f"{total_time:.2f}s"
    )

    # ===============================================================
    # RETURN
    #
    # These keys intentionally match the existing VITALIS pipeline.
    # ===============================================================

    return {

        "model": "Quantum Kernel SVM",

        "features": quantum_features,

        "n_qubits": n_qubits,

        "metrics": metrics,

        "qsvm": qsvm,

        "scaler": scaler,

        "target_encoder": target_encoder,

        "classes": classes.tolist(),

        "classification_type": (
            "binary"
            if num_classes == 2
            else "multiclass"
        ),

        "training_data": (
            X_train_scaled
        ),

        "training_labels": (
            y_train_encoded_array
        ),

        "training_labels_original": (
            pd.Series(
                y_train
            )
            .astype(str)
            .to_numpy()
        ),

        "test_data": (
            X_test_scaled
        ),

        "test_labels": (
            y_test_encoded_array
        ),

        "test_labels_original": (
            y_test_labels
        ),

        "predictions": (
            y_pred_encoded
        ),

        "prediction_labels": (
            y_pred_labels
        ),

        "probabilities": (
            y_prob
        ),

        "kernel_train": (
            K_train
        ),

        "kernel_test": (
            K_test
        ),

        "training_kernel_time": (
            training_kernel_time
        ),

        "test_kernel_time": (
            test_kernel_time
        ),

        "training_time": (
            training_time
        ),

        "evaluation_time": (
            evaluation_time
        ),

        "total_time": (
            total_time
        ),
    }


# -------------------------------------------------------------------
# Heart-disease demo loader
# -------------------------------------------------------------------

def load_dataset(
    path="data/heart_cleaned.csv"
):

    df = pd.read_csv(
        path
    )

    missing = [
        feature
        for feature in DEFAULT_QUANTUM_FEATURES
        if feature not in df.columns
    ]

    if missing:

        raise ValueError(
            "Dataset is missing required "
            f"heart quantum features: {missing}"
        )

    if "target" not in df.columns:

        raise ValueError(
            "Dataset must contain a 'target' column."
        )

    return (
        df[
            DEFAULT_QUANTUM_FEATURES
        ],
        df["target"],
    )


# -------------------------------------------------------------------
# Artifact saving
# -------------------------------------------------------------------

def save_artifacts(
    result,
    output_dir="models",
):
    """
    Save QSVM artifacts in the same structure expected
    by the existing VITALIS project.
    """

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    metrics = result[
        "metrics"
    ]

    benchmark = {

        "model": result[
            "model"
        ],

        "features": result[
            "features"
        ],

        "n_qubits": result[
            "n_qubits"
        ],

        "classes": result.get(
            "classes",
            [],
        ),

        "classification_type": result.get(
            "classification_type",
            "binary",
        ),

        "accuracy": round(
            metrics[
                "accuracy"
            ],
            4,
        ),

        "precision": round(
            metrics[
                "precision"
            ],
            4,
        ),

        "sensitivity": round(
            metrics[
                "sensitivity"
            ],
            4,
        ),

        "specificity": round(
            metrics[
                "specificity"
            ],
            4,
        ),

        "f1_score": round(
            metrics[
                "f1"
            ],
            4,
        ),

        "roc_auc": (
            round(
                metrics[
                    "roc_auc"
                ],
                4,
            )
            if metrics[
                "roc_auc"
            ] is not None
            else None
        ),

        "training_kernel_time": round(
            result[
                "training_kernel_time"
            ],
            4,
        ),

        "test_kernel_time": round(
            result[
                "test_kernel_time"
            ],
            4,
        ),

        "training_time": round(
            result[
                "training_time"
            ],
            4,
        ),

        "evaluation_time": round(
            result[
                "evaluation_time"
            ],
            4,
        ),

        "total_time": round(
            result[
                "total_time"
            ],
            4,
        ),
    }

    with open(
        os.path.join(
            output_dir,
            "quantum_benchmark.json",
        ),
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            benchmark,
            f,
            indent=4,
        )

    joblib.dump(
        result[
            "scaler"
        ],
        os.path.join(
            output_dir,
            "quantum_scaler.joblib",
        ),
    )

    joblib.dump(
        result[
            "training_data"
        ],
        os.path.join(
            output_dir,
            "quantum_training_data.joblib",
        ),
    )

    joblib.dump(
        result[
            "training_labels"
        ],
        os.path.join(
            output_dir,
            "quantum_training_labels.joblib",
        ),
    )

    joblib.dump(
        result[
            "qsvm"
        ],
        os.path.join(
            output_dir,
            "qsvm.joblib",
        ),
    )

    if (
        "target_encoder"
        in result
    ):

        joblib.dump(
            result[
                "target_encoder"
            ],
            os.path.join(
                output_dir,
                "quantum_target_encoder.joblib",
            ),
        )

    print(
        "\nSaved:"
    )

    print(
        os.path.join(
            output_dir,
            "quantum_benchmark.json",
        )
    )

    print(
        os.path.join(
            output_dir,
            "quantum_scaler.joblib",
        )
    )

    print(
        os.path.join(
            output_dir,
            "quantum_training_data.joblib",
        )
    )

    print(
        os.path.join(
            output_dir,
            "quantum_training_labels.joblib",
        )
    )

    print(
        os.path.join(
            output_dir,
            "qsvm.joblib",
        )
    )


# -------------------------------------------------------------------
# Standalone execution
# -------------------------------------------------------------------

if __name__ == "__main__":

    X, y = load_dataset()

    print(
        "Dataset shape:",
        (
            len(X),
            len(X.columns) + 1,
        ),
    )

    print(
        "Quantum features:",
        DEFAULT_QUANTUM_FEATURES,
    )

    result = train_quantum(
        X,
        y,
        quantum_features=(
            DEFAULT_QUANTUM_FEATURES
        ),
        n_qubits=len(
            DEFAULT_QUANTUM_FEATURES
        ),
    )

    save_artifacts(
        result
    )