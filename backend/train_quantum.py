import json
import os
import time

import joblib
import pandas as pd
import pennylane as qml

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from pennylane import numpy as np


# -------------------------------------------------------------------
# Default configuration
# -------------------------------------------------------------------
# Kept only for backward-compatible standalone heart-disease testing.
#
# The integrated VITALIS pipeline will provide quantum_features
# dynamically after feature selection + quantum reduction.
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
# Quantum kernel
# -------------------------------------------------------------------

def quantum_kernel(x1, x2, feature_map, n_qubits):
    """
    Compute the quantum kernel between two feature vectors.

    The kernel is the squared magnitude of the overlap between
    the quantum states produced by the feature map.
    """

    state1 = feature_map(x1)
    state2 = feature_map(x2)

    overlap = np.vdot(state1, state2)

    return float(np.abs(overlap) ** 2)


# -------------------------------------------------------------------
# Kernel matrix
# -------------------------------------------------------------------

def build_kernel_matrix(X1, X2, feature_map, n_qubits):
    """
    Build a quantum kernel matrix between two datasets.

    X1:
        First dataset.

    X2:
        Second dataset.

    The resulting matrix contains:

        K[i,j] = quantum_kernel(X1[i], X2[j])
    """

    matrix = np.zeros((len(X1), len(X2)))

    total = len(X1) * len(X2)
    completed = 0

    start_time = time.time()

    for i in range(len(X1)):

        for j in range(len(X2)):

            matrix[i, j] = quantum_kernel(
                X1[i],
                X2[j],
                feature_map,
                n_qubits,
            )

            completed += 1

        elapsed = time.time() - start_time

        print(
            f"Completed {completed}/{total} kernel evaluations "
            f"({elapsed:.1f}s)"
        )

    return matrix


# -------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------

def _metrics(y_test, y_pred, y_prob):
    """
    Calculate the standard binary-classification metrics
    used by the VITALIS benchmark.
    """

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    sensitivity = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    auc = roc_auc_score(
        y_test,
        y_prob,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1),
        "roc_auc": float(auc),
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
            quantum_features=...
        )

    This allows the VITALIS pipeline to create ONE common
    train/test split and pass the exact same test set to
    every model.

    Backward-compatible standalone usage:

        train_quantum(X, y)

    In standalone mode, this function creates its own split.
    """

    total_start_time = time.time()

    # ---------------------------------------------------------------
    # Convert training data to DataFrame
    # ---------------------------------------------------------------

    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)

    X_train = X_train.copy()

    # ---------------------------------------------------------------
    # Backward-compatible standalone mode
    # ---------------------------------------------------------------

    if X_test is None or y_test is None:

        print("\nNo explicit test set supplied.")
        print("Creating standalone train/test split.")

        y_train = pd.Series(
            y_train
        ).reset_index(drop=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X_train,
            y_train,
            test_size=test_size,
            random_state=random_state,
            stratify=y_train,
        )

    else:

        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test)

        X_test = X_test.copy()

        y_train = pd.Series(
            y_train
        ).reset_index(drop=True)

        y_test = pd.Series(
            y_test
        ).reset_index(drop=True)

    # ---------------------------------------------------------------
    # Select quantum features
    # ---------------------------------------------------------------

    if quantum_features is None:

        quantum_features = list(
            DEFAULT_QUANTUM_FEATURES
        )

    else:

        quantum_features = list(
            quantum_features
        )

    if len(quantum_features) == 0:

        raise ValueError(
            "At least one quantum feature is required."
        )

    # ---------------------------------------------------------------
    # Validate feature availability
    # ---------------------------------------------------------------

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

    X_train_quantum = X_train[
        quantum_features
    ].copy()

    X_test_quantum = X_test[
        quantum_features
    ].copy()

    # ---------------------------------------------------------------
    # Validate numeric data
    # ---------------------------------------------------------------

    for feature in quantum_features:

        X_train_quantum[feature] = pd.to_numeric(
            X_train_quantum[feature],
            errors="coerce",
        )

        X_test_quantum[feature] = pd.to_numeric(
            X_test_quantum[feature],
            errors="coerce",
        )

    if X_train_quantum.isna().any().any():

        missing_counts = (
            X_train_quantum
            .isna()
            .sum()
            .loc[lambda s: s > 0]
            .to_dict()
        )

        raise ValueError(
            "Quantum training features contain "
            f"missing/non-numeric values: {missing_counts}"
        )

    if X_test_quantum.isna().any().any():

        missing_counts = (
            X_test_quantum
            .isna()
            .sum()
            .loc[lambda s: s > 0]
            .to_dict()
        )

        raise ValueError(
            "Quantum test features contain "
            f"missing/non-numeric values: {missing_counts}"
        )

    # ---------------------------------------------------------------
    # Validate target lengths
    # ---------------------------------------------------------------

    if len(X_train_quantum) != len(y_train):

        raise ValueError(
            "Training feature and target lengths do not match."
        )

    if len(X_test_quantum) != len(y_test):

        raise ValueError(
            "Test feature and target lengths do not match."
        )

    # ---------------------------------------------------------------
    # Validate targets
    # ---------------------------------------------------------------

    y_train = pd.to_numeric(
        y_train,
        errors="coerce",
    )

    y_test = pd.to_numeric(
        y_test,
        errors="coerce",
    )

    if y_train.isna().any():

        raise ValueError(
            "Quantum training target contains "
            "missing or non-numeric values."
        )

    if y_test.isna().any():

        raise ValueError(
            "Quantum test target contains "
            "missing or non-numeric values."
        )

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    if y_train.nunique() != 2:

        raise ValueError(
            "Quantum training requires exactly two target classes."
        )

    if y_test.nunique() < 1:

        raise ValueError(
            "Quantum test set contains no valid target classes."
        )

    # ---------------------------------------------------------------
    # Reset indices
    # ---------------------------------------------------------------

    X_train_quantum = (
        X_train_quantum
        .reset_index(drop=True)
    )

    X_test_quantum = (
        X_test_quantum
        .reset_index(drop=True)
    )

    y_train = (
        y_train
        .reset_index(drop=True)
    )

    y_test = (
        y_test
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------------
    # Determine number of qubits
    # ---------------------------------------------------------------

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

    if n_qubits != len(quantum_features):

        raise ValueError(
            "For the current feature-map implementation, "
            "n_qubits must equal the number of quantum features."
        )

    # ---------------------------------------------------------------
    # Scale features into quantum rotation range
    # ---------------------------------------------------------------

    # IMPORTANT:
    #
    # The scaler is fitted ONLY on training data.
    #
    # The test set is transformed using the already-fitted scaler.
    #
    # This prevents preprocessing leakage.

    scaler = MinMaxScaler(
        feature_range=(0, np.pi)
    )

    X_train_scaled = scaler.fit_transform(
        X_train_quantum
    )

    X_test_scaled = scaler.transform(
        X_test_quantum
    )

    # ---------------------------------------------------------------
    # Quantum device
    # ---------------------------------------------------------------

    dev = qml.device(
        "default.qubit",
        wires=n_qubits,
    )

    # ---------------------------------------------------------------
    # Generic quantum feature map
    # ---------------------------------------------------------------

    @qml.qnode(dev)
    def feature_map(x):
        """
        Angle-encoding feature map.

        Each feature is encoded into one qubit using
        an RY rotation.

        A linear CNOT chain introduces entanglement.
        """

        for i in range(n_qubits):

            qml.RY(
                x[i],
                wires=i,
            )

        for i in range(n_qubits - 1):

            qml.CNOT(
                wires=[i, i + 1]
            )

        return qml.state()

    # ---------------------------------------------------------------
    # Build training quantum kernel
    # ---------------------------------------------------------------

    print("\n" + "=" * 55)
    print("BUILDING TRAINING QUANTUM KERNEL")
    print("=" * 55)

    kernel_train_start = time.time()

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

    # ---------------------------------------------------------------
    # Build test quantum kernel
    # ---------------------------------------------------------------

    print("\n" + "=" * 55)
    print("BUILDING TEST QUANTUM KERNEL")
    print("=" * 55)

    kernel_test_start = time.time()

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

    # ---------------------------------------------------------------
    # Train QSVM
    # ---------------------------------------------------------------

    print("\n" + "=" * 55)
    print("TRAINING QUANTUM KERNEL SVM")
    print("=" * 55)

    training_start = time.time()

    qsvm = SVC(
        kernel="precomputed",
        probability=True,
        random_state=random_state,
    )

    qsvm.fit(
        K_train,
        y_train,
    )

    training_time = (
        time.time()
        - training_start
    )

    # ---------------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------------

    print("\n" + "=" * 55)
    print("EVALUATING QUANTUM KERNEL SVM")
    print("=" * 55)

    evaluation_start = time.time()

    y_pred = qsvm.predict(
        K_test
    )

    y_prob = qsvm.predict_proba(
        K_test
    )[:, 1]

    evaluation_time = (
        time.time()
        - evaluation_start
    )

    # ---------------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------------

    metrics = _metrics(
        y_test,
        y_pred,
        y_prob,
    )

    total_time = (
        time.time()
        - total_start_time
    )

    # ---------------------------------------------------------------
    # Print results
    # ---------------------------------------------------------------

    print("\n" + "=" * 55)
    print("QUANTUM KERNEL SVM RESULTS")
    print("=" * 55)

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

    # ---------------------------------------------------------------
    # Return complete result
    # ---------------------------------------------------------------

    return {

        "model": "Quantum Kernel SVM",

        "features": quantum_features,

        "n_qubits": n_qubits,

        "metrics": metrics,

        "qsvm": qsvm,

        "scaler": scaler,

        "training_data": X_train_scaled,

        "training_labels": y_train.to_numpy(),

        "test_data": X_test_scaled,

        "test_labels": y_test.to_numpy(),

        "predictions": y_pred,

        "probabilities": y_prob,

        "kernel_train": K_train,

        "kernel_test": K_test,

        "training_kernel_time": (
            training_kernel_time
        ),

        "test_kernel_time": (
            test_kernel_time
        ),

        "training_time": training_time,

        "evaluation_time": evaluation_time,

        "total_time": total_time,
    }


# -------------------------------------------------------------------
# Heart-disease demo loader
# -------------------------------------------------------------------
# This exists only for standalone backward compatibility.
#
# The future VITALIS pipeline will NOT depend on this loader.
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
            "Dataset is missing required heart "
            f"quantum features: {missing}"
        )

    if "target" not in df.columns:

        raise ValueError(
            "Dataset must contain a 'target' column."
        )

    return (
        df[DEFAULT_QUANTUM_FEATURES],
        df["target"],
    )


# -------------------------------------------------------------------
# Artifact saving
# -------------------------------------------------------------------

def save_artifacts(
    result,
    output_dir="models",
):

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    metrics = result["metrics"]

    benchmark = {

        "model": result["model"],

        "features": result["features"],

        "n_qubits": result["n_qubits"],

        "accuracy": round(
            metrics["accuracy"],
            4,
        ),

        "precision": round(
            metrics["precision"],
            4,
        ),

        "sensitivity": round(
            metrics["sensitivity"],
            4,
        ),

        "specificity": round(
            metrics["specificity"],
            4,
        ),

        "f1_score": round(
            metrics["f1"],
            4,
        ),

        "roc_auc": round(
            metrics["roc_auc"],
            4,
        ),

        "training_kernel_time": round(
            result["training_kernel_time"],
            4,
        ),

        "test_kernel_time": round(
            result["test_kernel_time"],
            4,
        ),

        "training_time": round(
            result["training_time"],
            4,
        ),

        "evaluation_time": round(
            result["evaluation_time"],
            4,
        ),

        "total_time": round(
            result["total_time"],
            4,
        ),
    }

    with open(
        f"{output_dir}/quantum_benchmark.json",
        "w",
    ) as f:

        json.dump(
            benchmark,
            f,
            indent=4,
        )

    joblib.dump(
        result["scaler"],
        f"{output_dir}/quantum_scaler.joblib",
    )

    joblib.dump(
        result["training_data"],
        f"{output_dir}/quantum_training_data.joblib",
    )

    joblib.dump(
        result["training_labels"],
        f"{output_dir}/quantum_training_labels.joblib",
    )

    joblib.dump(
        result["qsvm"],
        f"{output_dir}/qsvm.joblib",
    )

    print("\nSaved:")

    print(
        f"{output_dir}/quantum_benchmark.json"
    )

    print(
        f"{output_dir}/quantum_scaler.joblib"
    )

    print(
        f"{output_dir}/quantum_training_data.joblib"
    )

    print(
        f"{output_dir}/quantum_training_labels.joblib"
    )

    print(
        f"{output_dir}/qsvm.joblib"
    )


# -------------------------------------------------------------------
# Standalone heart-disease experiment
# -------------------------------------------------------------------

if __name__ == "__main__":

    X, y = load_dataset()

    print(
        "Dataset shape:",
        (len(X), len(X.columns) + 1),
    )

    print(
        "Quantum features:",
        DEFAULT_QUANTUM_FEATURES,
    )

    # This standalone execution intentionally uses
    # backward-compatible automatic splitting.
    #
    # The integrated VITALIS pipeline will instead provide
    # X_train, y_train, X_test, y_test explicitly.

    result = train_quantum(
        X,
        y,
        quantum_features=DEFAULT_QUANTUM_FEATURES,
        n_qubits=len(DEFAULT_QUANTUM_FEATURES),
    )

    save_artifacts(
        result
    )