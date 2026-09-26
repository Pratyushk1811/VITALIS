"""
Generic Variational Quantum Classifier (VQC)

Pipeline:

    Classical features
        ↓
    Quantum dimensionality reduction
        ↓
    Angle encoding
        ↓
    Variational quantum circuit
        ↓
    Pauli-Z expectation
        ↓
    Probability of class 1
        ↓
    Binary classification metrics

The VQC currently supports binary classification.

Important:
For a Pauli-Z measurement:

    <Z> = P(0) - P(1)

Therefore:

    P(1) = (1 - <Z>) / 2

The integrated VITALIS pipeline supplies one shared
train/test split to this model.

For backward compatibility, train_vqc() can still create
its own split when X_test/y_test are not supplied.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pennylane as qml

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

DEFAULT_TEST_SIZE = 0.20
DEFAULT_RANDOM_STATE = 42

DEFAULT_LEARNING_RATE = 0.05
DEFAULT_EPOCHS = 30
DEFAULT_N_LAYERS = 2


# ============================================================
# VQC MODEL
# ============================================================

class VQCModel:
    """
    Variational Quantum Classifier.

    Each input feature is encoded into one qubit using RY.

    Each variational layer contains:

        RY
        RZ
        CNOT chain

    The circuit measures Pauli-Z on the first qubit.
    """

    def __init__(
        self,
        n_qubits,
        n_layers=DEFAULT_N_LAYERS,
        learning_rate=DEFAULT_LEARNING_RATE,
        random_state=DEFAULT_RANDOM_STATE,
    ):

        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.learning_rate = float(learning_rate)
        self.random_state = int(random_state)

        self.dev = qml.device(
            "default.qubit",
            wires=self.n_qubits,
        )

        rng = np.random.default_rng(
            self.random_state
        )

        initial_weights = rng.normal(
            loc=0.0,
            scale=0.05,
            size=(
                self.n_layers,
                self.n_qubits,
                2,
            ),
        )

        self.weights = qml.numpy.array(
            initial_weights,
            requires_grad=True,
        )

        self.qnode = qml.QNode(
            self._circuit,
            self.dev,
            interface="autograd",
            diff_method="backprop",
        )

    # --------------------------------------------------------
    # Quantum circuit
    # --------------------------------------------------------

    def _circuit(
        self,
        features,
        weights,
    ):

        # ----------------------------------------------------
        # Feature encoding
        # ----------------------------------------------------

        for qubit in range(
            self.n_qubits
        ):

            qml.RY(
                features[qubit],
                wires=qubit,
            )

        # ----------------------------------------------------
        # Variational layers
        # ----------------------------------------------------

        for layer in range(
            self.n_layers
        ):

            for qubit in range(
                self.n_qubits
            ):

                qml.RY(
                    weights[
                        layer,
                        qubit,
                        0,
                    ],
                    wires=qubit,
                )

                qml.RZ(
                    weights[
                        layer,
                        qubit,
                        1,
                    ],
                    wires=qubit,
                )

            # CNOT entanglement chain
            for qubit in range(
                self.n_qubits - 1
            ):

                qml.CNOT(
                    wires=[
                        qubit,
                        qubit + 1,
                    ]
                )

        # ----------------------------------------------------
        # Measurement
        # ----------------------------------------------------

        return qml.expval(
            qml.PauliZ(0)
        )

    # --------------------------------------------------------
    # Raw expectation
    # --------------------------------------------------------

    def expectation(
        self,
        features,
        weights=None,
    ):

        if weights is None:
            weights = self.weights

        return self.qnode(
            features,
            weights,
        )

    # --------------------------------------------------------
    # Probability of class 1
    # --------------------------------------------------------

    def probability_class_1(
        self,
        features,
        weights=None,
    ):
        """
        Convert Pauli-Z expectation into P(class=1).

        <Z> = P(0) - P(1)

        Since:

            P(0) + P(1) = 1

        we get:

            P(1) = (1 - <Z>) / 2
        """

        expectation = self.expectation(
            features,
            weights,
        )

        probability = (
            1.0 - expectation
        ) / 2.0

        return probability

    # --------------------------------------------------------
    # Batch probability prediction
    # --------------------------------------------------------

    def predict_proba(
        self,
        X,
        weights=None,
    ):

        probabilities = []

        for row in X:

            probability = (
                self.probability_class_1(
                    row,
                    weights,
                )
            )

            probabilities.append(
                probability
            )

        return np.asarray(
            probabilities,
            dtype=float,
        )

    # --------------------------------------------------------
    # Class prediction
    # --------------------------------------------------------

    def predict(
        self,
        X,
        weights=None,
        threshold=0.5,
    ):

        probabilities = (
            self.predict_proba(
                X,
                weights,
            )
        )

        return (
            probabilities >= threshold
        ).astype(int)


# ============================================================
# BINARY CROSS ENTROPY
# ============================================================

def binary_cross_entropy(
    model,
    X,
    y,
    weights,
):
    """
    Binary cross entropy.

    Only the quantum weights are differentiable.

    Labels are ordinary Python/NumPy scalar values.
    """

    losses = []

    for features, label in zip(
        X,
        y,
    ):

        # Quantum output.
        expectation = model.expectation(
            features,
            weights,
        )

        # Convert Pauli-Z expectation
        # to probability of class 1.
        probability = (
            1.0 - expectation
        ) / 2.0

        # Numerical stability.
        probability = qml.math.clip(
            probability,
            1e-7,
            1.0 - 1e-7,
        )

        # Label is a constant.
        #
        # It must NOT be a differentiable
        # PennyLane parameter.
        label = float(label)

        loss = (
            -label
            * qml.math.log(
                probability
            )
            - (
                1.0 - label
            )
            * qml.math.log(
                1.0 - probability
            )
        )

        losses.append(loss)

    return qml.math.mean(
        qml.math.stack(losses)
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold=0.5,
):
    """
    Calculate binary classification metrics.
    """

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    # --------------------------------------------------------
    # Sensitivity / Recall
    # --------------------------------------------------------

    sensitivity = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    # --------------------------------------------------------
    # Confusion matrix
    #
    # [[TN, FP],
    #  [FN, TP]]
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    # --------------------------------------------------------
    # Specificity
    # --------------------------------------------------------

    if (tn + fp) > 0:

        specificity = (
            tn / (tn + fp)
        )

    else:

        specificity = 0.0

    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    try:

        roc_auc = roc_auc_score(
            y_true,
            probabilities,
        )

    except ValueError:

        roc_auc = None

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
            float(roc_auc)
            if roc_auc is not None
            else None
        ),
    }


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_inputs(
    X,
    y,
):
    """
    Validate VQC input data.
    """

    if not isinstance(
        X,
        pd.DataFrame,
    ):

        raise TypeError(
            "X must be a pandas DataFrame."
        )

    if len(X) == 0:

        raise ValueError(
            "X is empty."
        )

    if len(X) != len(y):

        raise ValueError(
            "X and y must have the same "
            "number of samples."
        )

    # --------------------------------------------------------
    # Numeric feature validation
    # --------------------------------------------------------

    for column in X.columns:

        if not pd.api.types.is_numeric_dtype(
            X[column]
        ):

            raise ValueError(
                "Quantum features must be numeric. "
                f"Column '{column}' is not numeric."
            )

    # --------------------------------------------------------
    # Target validation
    # --------------------------------------------------------

    y_array = np.asarray(y)

    unique_classes = np.unique(
        y_array
    )

    if len(unique_classes) != 2:

        raise ValueError(
            "VQC currently supports "
            "binary classification only."
        )

    if not set(
        unique_classes
    ).issubset({0, 1}):

        raise ValueError(
            "Target labels must be encoded "
            "as 0 and 1."
        )


# ============================================================
# TRAIN VQC
# ============================================================

def train_vqc(
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    test_size=DEFAULT_TEST_SIZE,
    random_state=DEFAULT_RANDOM_STATE,
    learning_rate=DEFAULT_LEARNING_RATE,
    epochs=DEFAULT_EPOCHS,
    n_layers=DEFAULT_N_LAYERS,
):
    """
    Train a Variational Quantum Classifier.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training quantum-ready features.

    y_train : array-like
        Training binary labels encoded as 0 and 1.

    X_test : pandas.DataFrame, optional
        Held-out test quantum-ready features.

        If omitted, a standalone train/test split is created
        for backward compatibility.

    y_test : array-like, optional
        Held-out test labels.

    test_size : float
        Fraction reserved for testing when an automatic
        standalone split is required.

    random_state : int
        Random seed.

    learning_rate : float
        Adam optimizer learning rate.

    epochs : int
        Number of optimization epochs.

    n_layers : int
        Number of variational layers.

    Returns
    -------
    dict
        VQC model, metrics, scaler and training artifacts.

    Important
    ---------
    In the integrated VITALIS pipeline, X_train/y_train and
    X_test/y_test are supplied by train_pipeline.py.

    Therefore this model does NOT create another split.
    """

    # --------------------------------------------------------
    # Validate training input
    # --------------------------------------------------------

    validate_inputs(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Determine whether this is an integrated run
    # or standalone backward-compatible run.
    # --------------------------------------------------------

    if (
        X_test is None
        and y_test is None
    ):

        print()
        print(
            "No explicit test set supplied. "
            "Creating standalone train/test split."
        )

        X = X_train.copy()

        y = np.asarray(
            y_train,
            dtype=int,
        )

        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

    elif (
        X_test is None
        or y_test is None
    ):

        raise ValueError(
            "X_test and y_test must either both be supplied "
            "or both be omitted."
        )

    else:

        print()
        print(
            "Explicit train/test split supplied. "
            "Using it without creating another split."
        )

        validate_inputs(
            X_test,
            y_test,
        )

    # --------------------------------------------------------
    # Copy inputs
    # --------------------------------------------------------

    X_train = X_train.copy()
    X_test = X_test.copy()

    y_train = np.asarray(
        y_train,
        dtype=int,
    )

    y_test = np.asarray(
        y_test,
        dtype=int,
    )

    # --------------------------------------------------------
    # Validate train/test feature compatibility
    # --------------------------------------------------------

    if list(X_train.columns) != list(
        X_test.columns
    ):

        raise ValueError(
            "X_train and X_test must contain "
            "the same features in the same order."
        )

    if len(X_train) == 0:

        raise ValueError(
            "X_train is empty."
        )

    if len(X_test) == 0:

        raise ValueError(
            "X_test is empty."
        )

    feature_names = (
        X_train.columns.tolist()
    )

    n_qubits = X_train.shape[1]

    if n_qubits < 1:

        raise ValueError(
            "At least one quantum feature "
            "is required."
        )

    # --------------------------------------------------------
    # Print configuration
    # --------------------------------------------------------

    print()
    print("=" * 50)
    print(
        "TRAINING VARIATIONAL QUANTUM CLASSIFIER"
    )
    print("=" * 50)

    print(
        f"Qubits:        {n_qubits}"
    )

    print(
        f"Layers:        {n_layers}"
    )

    print(
        f"Training data: {len(X_train)}"
    )

    print(
        f"Test data:     {len(X_test)}"
    )

    print(
        f"Epochs:        {epochs}"
    )

    print(
        f"Learning rate: {learning_rate}"
    )

    # --------------------------------------------------------
    # Scale features to [0, pi]
    #
    # IMPORTANT:
    # The scaler is fitted ONLY on training data.
    # --------------------------------------------------------

    scaler = MinMaxScaler(
        feature_range=(
            0.0,
            np.pi,
        )
    )

    X_train_scaled = (
        scaler.fit_transform(
            X_train
        )
    )

    X_test_scaled = (
        scaler.transform(
            X_test
        )
    )

    # --------------------------------------------------------
    # Convert training features to
    # PennyLane-compatible arrays.
    #
    # requires_grad=False is correct because
    # features are not trainable parameters.
    # --------------------------------------------------------

    X_train_q = qml.numpy.array(
        X_train_scaled,
        requires_grad=False,
    )

    # IMPORTANT:
    #
    # y_train is deliberately NOT converted
    # into a PennyLane tensor.
    #
    # Labels are constants and do not need
    # gradients.
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Create VQC model
    # --------------------------------------------------------

    model = VQCModel(
        n_qubits=n_qubits,
        n_layers=n_layers,
        learning_rate=learning_rate,
        random_state=random_state,
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = qml.AdamOptimizer(
        stepsize=learning_rate
    )

    weights = model.weights

    loss_history = []

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    training_start = time.perf_counter()

    for epoch in range(
        epochs
    ):

        def objective(
            current_weights
        ):

            return binary_cross_entropy(
                model,
                X_train_q,
                y_train,
                current_weights,
            )

        (
            weights,
            loss,
        ) = optimizer.step_and_cost(
            objective,
            weights,
        )

        loss_value = float(
            loss
        )

        loss_history.append(
            loss_value
        )

        print(
            f"Epoch {epoch + 1:02d}/{epochs} "
            f"- Loss: {loss_value:.6f}"
        )

    training_time = (
        time.perf_counter()
        - training_start
    )

    # --------------------------------------------------------
    # Save final weights
    # --------------------------------------------------------

    model.weights = weights

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    evaluation_start = (
        time.perf_counter()
    )

    # Test probabilities.
    test_probabilities = (
        model.predict_proba(
            X_test_scaled,
            weights,
        )
    )

    # Test predictions.
    test_predictions = (
        test_probabilities >= 0.5
    ).astype(int)

    # Test metrics.
    test_metrics = calculate_metrics(
        y_test,
        test_probabilities,
    )

    # --------------------------------------------------------
    # Training-set probabilities
    # --------------------------------------------------------

    train_probabilities = (
        model.predict_proba(
            X_train_scaled,
            weights,
        )
    )

    train_predictions = (
        train_probabilities >= 0.5
    ).astype(int)

    train_metrics = calculate_metrics(
        y_train,
        train_probabilities,
    )

    evaluation_time = (
        time.perf_counter()
        - evaluation_start
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("=" * 50)
    print("VQC RESULTS")
    print("=" * 50)

    print(
        f"Accuracy:     "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision:    "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Sensitivity:  "
        f"{test_metrics['sensitivity']:.4f}"
    )

    print(
        f"Specificity:  "
        f"{test_metrics['specificity']:.4f}"
    )

    print(
        f"F1 Score:     "
        f"{test_metrics['f1']:.4f}"
    )

    if (
        test_metrics["roc_auc"]
        is not None
    ):

        print(
            f"ROC-AUC:      "
            f"{test_metrics['roc_auc']:.4f}"
        )

    else:

        print(
            "ROC-AUC:      N/A"
        )

    print(
        f"Training time:   "
        f"{training_time:.2f}s"
    )

    print(
        f"Evaluation time: "
        f"{evaluation_time:.2f}s"
    )

    # --------------------------------------------------------
    # Return artifacts
    # --------------------------------------------------------

    return {

        "model": model,

        "model_name": (
            "Variational Quantum Classifier"
        ),

        "features": feature_names,

        "n_qubits": n_qubits,

        "n_layers": n_layers,

        "learning_rate": (
            learning_rate
        ),

        "epochs": epochs,

        "metrics": test_metrics,

        "train_metrics": (
            train_metrics
        ),

        "training_time": float(
            training_time
        ),

        "evaluation_time": float(
            evaluation_time
        ),

        "loss_history": (
            loss_history
        ),

        "scaler": scaler,

        "weights": weights,

        "X_train": X_train,

        "X_test": X_test,

        "y_train": y_train,

        "y_test": y_test,

        "train_probabilities": (
            train_probabilities
        ),

        "test_probabilities": (
            test_probabilities
        ),

        "train_predictions": (
            train_predictions
        ),

        "test_predictions": (
            test_predictions
        ),
    }


# ============================================================
# STANDALONE HEART DATASET TEST
# ============================================================

def standalone_test():

    from quantum_reducer import (
        QuantumFeatureReducer,
    )

    dataset_path = Path(
        "data/heart_cleaned.csv"
    )

    print(
        f"Loading: {dataset_path}"
    )

    if not dataset_path.exists():

        raise FileNotFoundError(
            f"Dataset not found: "
            f"{dataset_path}"
        )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        dataset_path
    )

    if "target" not in df.columns:

        raise ValueError(
            "Dataset must contain "
            "a 'target' column."
        )

    X = df.drop(
        columns=["target"]
    )

    y = df["target"]

    # --------------------------------------------------------
    # Quantum dimensionality reduction
    #
    # Standalone demonstration only.
    #
    # The integrated platform pipeline will fit
    # preprocessing only on the training set.
    # --------------------------------------------------------

    reducer = QuantumFeatureReducer(
        n_components=6
    )

    X_reduced = (
        reducer.fit_transform(
            X
        )
    )

    quantum_feature_names = (
        reducer.get_feature_names()
    )

    X_quantum = pd.DataFrame(
        X_reduced,
        columns=quantum_feature_names,
        index=X.index,
    )

    # --------------------------------------------------------
    # Train VQC
    # --------------------------------------------------------

    result = train_vqc(
        X_quantum,
        y,
        test_size=DEFAULT_TEST_SIZE,
        random_state=DEFAULT_RANDOM_STATE,
        learning_rate=DEFAULT_LEARNING_RATE,
        epochs=DEFAULT_EPOCHS,
        n_layers=DEFAULT_N_LAYERS,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 50)
    print(
        "STANDALONE VQC TEST COMPLETE"
    )
    print("=" * 50)

    print(
        f"Quantum features: "
        f"{quantum_feature_names}"
    )

    print(
        f"Qubits: "
        f"{result['n_qubits']}"
    )

    print(
        f"Layers: "
        f"{result['n_layers']}"
    )

    print(
        f"Test accuracy: "
        f"{result['metrics']['accuracy']:.4f}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    standalone_test()