"""
VITALIS - Variational Quantum Classifier (VQC)

Supports:
    - Binary classification
    - Multiclass classification

Pipeline:

    Classical features
            ↓
    Quantum dimensionality reduction
            ↓
    Angle encoding
            ↓
    Variational quantum circuit
            ↓
    Pauli-Z expectation values
            ↓
    Class logits
            ↓
    Differentiable softmax
            ↓
    Prediction

The integrated VITALIS pipeline supplies the shared
train/test split.

The model does NOT create another split when
X_test/y_test are explicitly supplied.
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
# DIFFERENTIABLE SOFTMAX
# ============================================================

def differentiable_softmax(logits):
    """
    Numerically stable softmax implemented only with
    basic PennyLane math operations.

    This intentionally avoids qml.math.softmax because
    the installed PennyLane/Autoray stack is failing to
    resolve softmax for the Autograd backend.

    The operations below remain differentiable with
    respect to the VQC weights.
    """

    shifted = (
        logits
        - qml.math.max(logits)
    )

    exponentials = qml.math.exp(
        shifted
    )

    denominator = qml.math.sum(
        exponentials
    )

    return (
        exponentials
        / denominator
    )


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

    For binary classification:
        two measured expectation values are used as logits.

    For multiclass classification:
        one measured expectation value is used per class.
    """

    def __init__(
        self,
        n_qubits,
        n_classes=2,
        n_layers=DEFAULT_N_LAYERS,
        learning_rate=DEFAULT_LEARNING_RATE,
        random_state=DEFAULT_RANDOM_STATE,
    ):
        self.n_qubits = int(n_qubits)
        self.n_classes = int(n_classes)
        self.n_layers = int(n_layers)
        self.learning_rate = float(
            learning_rate
        )
        self.random_state = int(
            random_state
        )

        if self.n_qubits < 1:
            raise ValueError(
                "At least one qubit is required."
            )

        if self.n_classes < 2:
            raise ValueError(
                "At least two classes are required."
            )

        if self.n_classes > self.n_qubits:
            raise ValueError(
                f"VQC requires at least one measured "
                f"qubit per class. Received "
                f"{self.n_classes} classes but only "
                f"{self.n_qubits} qubits."
            )

        # ----------------------------------------------------
        # Quantum device
        # ----------------------------------------------------

        self.dev = qml.device(
            "default.qubit",
            wires=self.n_qubits,
        )

        # ----------------------------------------------------
        # Deterministic initialization
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # QNode
        # ----------------------------------------------------

        self.qnode = qml.QNode(
            self._circuit,
            self.dev,
            interface="autograd",
            diff_method="backprop",
        )

    # ========================================================
    # QUANTUM CIRCUIT
    # ========================================================

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

            # ------------------------------------------------
            # CNOT entanglement chain
            # ------------------------------------------------

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
        # Measurements
        #
        # One Pauli-Z expectation per class.
        # ----------------------------------------------------

        return tuple(
            qml.expval(
                qml.PauliZ(
                    qubit
                )
            )
            for qubit in range(
                self.n_classes
            )
        )

    # ========================================================
    # LOGITS
    # ========================================================

    def logits(
        self,
        features,
        weights=None,
    ):
        if weights is None:
            weights = self.weights

        values = self.qnode(
            features,
            weights,
        )

        return qml.math.stack(
            values
        )

    # ========================================================
    # PROBABILITY PREDICTION
    # ========================================================

    def predict_proba(
        self,
        X,
        weights=None,
    ):
        if weights is None:
            weights = self.weights

        probabilities = []

        for row in X:

            logits = self.logits(
                row,
                weights,
            )

            # IMPORTANT:
            # Do NOT use qml.math.softmax().
            probs = differentiable_softmax(
                logits
            )

            probabilities.append(
                probs
            )

        return np.asarray(
            probabilities,
            dtype=float,
        )

    # ========================================================
    # CLASS PREDICTION
    # ========================================================

    def predict(
        self,
        X,
        weights=None,
    ):
        probabilities = (
            self.predict_proba(
                X,
                weights,
            )
        )

        return np.argmax(
            probabilities,
            axis=1,
        ).astype(int)


# ============================================================
# MULTICLASS CROSS ENTROPY
# ============================================================

def multiclass_cross_entropy(
    model,
    X,
    y,
    weights,
):
    """
    Multiclass cross-entropy.

    The quantum circuit produces class logits.

    A manually implemented differentiable softmax converts
    them to probabilities.

    This avoids the broken qml.math.softmax -> Autoray ->
    Autograd path in the current environment.
    """

    losses = []

    for features, label in zip(
        X,
        y,
    ):

        logits = model.logits(
            features,
            weights,
        )

        # ----------------------------------------------------
        # Differentiable softmax
        # ----------------------------------------------------

        probabilities = (
            differentiable_softmax(
                logits
            )
        )

        # ----------------------------------------------------
        # Numerical protection
        # ----------------------------------------------------

        probabilities = qml.math.clip(
            probabilities,
            1e-7,
            1.0 - 1e-7,
        )

        # IMPORTANT:
        # label is ordinary NumPy/Python data.
        # It is NOT a differentiable parameter.
        label = int(label)

        loss = -qml.math.log(
            probabilities[label]
        )

        losses.append(
            loss
        )

    return qml.math.mean(
        qml.math.stack(
            losses
        )
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
):
    """
    Calculate binary or multiclass metrics.

    Multiclass:
        precision      = macro
        sensitivity    = macro recall
        specificity    = macro one-vs-rest
        F1             = macro
        ROC-AUC        = OVR macro
    """

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if probabilities.ndim == 1:
        probabilities = np.column_stack(
            [
                1.0 - probabilities,
                probabilities,
            ]
        )

    predictions = np.argmax(
        probabilities,
        axis=1,
    ).astype(int)

    num_classes = int(
        probabilities.shape[1]
    )

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
        average="macro",
        zero_division=0,
    )

    # --------------------------------------------------------
    # Sensitivity / Recall
    # --------------------------------------------------------

    sensitivity = recall_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0,
    )

    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    f1 = f1_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0,
    )

    # --------------------------------------------------------
    # Specificity
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=list(
            range(num_classes)
        ),
    )

    specificities = []

    for class_index in range(
        num_classes
    ):

        tp = cm[
            class_index,
            class_index,
        ]

        fn = (
            cm[class_index, :].sum()
            - tp
        )

        fp = (
            cm[:, class_index].sum()
            - tp
        )

        tn = (
            cm.sum()
            - tp
            - fn
            - fp
        )

        denominator = (
            tn + fp
        )

        if denominator > 0:
            class_specificity = (
                tn / denominator
            )
        else:
            class_specificity = 0.0

        specificities.append(
            float(
                class_specificity
            )
        )

    specificity = float(
        np.mean(
            specificities
        )
    )

    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    roc_auc = None

    try:

        if num_classes == 2:

            roc_auc = roc_auc_score(
                y_true,
                probabilities[:, 1],
            )

        else:

            roc_auc = roc_auc_score(
                y_true,
                probabilities,
                multi_class="ovr",
                average="macro",
            )

    except ValueError:

        roc_auc = None

    # --------------------------------------------------------
    # Return plain Python values
    # --------------------------------------------------------

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

    Labels must be:

        Binary:
            0, 1

        Multiclass:
            0, 1, 2, ...
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
    # Feature validation
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

    y_array = np.asarray(
        y,
        dtype=int,
    )

    unique_classes = np.unique(
        y_array
    )

    if len(unique_classes) < 2:
        raise ValueError(
            "VQC requires at least two target classes."
        )

    expected_classes = np.arange(
        len(unique_classes),
        dtype=int,
    )

    if not np.array_equal(
        unique_classes,
        expected_classes,
    ):
        raise ValueError(
            "Target labels must be integer encoded "
            "consecutively starting from 0. "
            f"Received classes: "
            f"{unique_classes.tolist()}"
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

    If X_test and y_test are supplied, they are used directly.

    This preserves the shared train/test split created by
    train_pipeline.py.
    """

    # --------------------------------------------------------
    # Initial validation
    # --------------------------------------------------------

    validate_inputs(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Integrated vs standalone mode
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
            "X_test and y_test must either both be "
            "supplied or both be omitted."
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
    # Feature compatibility
    # --------------------------------------------------------

    if list(
        X_train.columns
    ) != list(
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

    n_qubits = int(
        X_train.shape[1]
    )

    if n_qubits < 1:
        raise ValueError(
            "At least one quantum feature "
            "is required."
        )

    # --------------------------------------------------------
    # Determine classes
    # --------------------------------------------------------

    all_classes = np.unique(
        np.concatenate(
            [
                y_train,
                y_test,
            ]
        )
    )

    all_classes = np.asarray(
        all_classes,
        dtype=int,
    )

    n_classes = int(
        len(all_classes)
    )

    expected_classes = np.arange(
        n_classes,
        dtype=int,
    )

    if not np.array_equal(
        all_classes,
        expected_classes,
    ):
        raise ValueError(
            "VQC labels must be encoded consecutively "
            "starting from 0. "
            f"Received classes: "
            f"{all_classes.tolist()}"
        )

    classification_type = (
        "binary"
        if n_classes == 2
        else "multiclass"
    )

    if n_classes > n_qubits:
        raise ValueError(
            f"VQC requires at least {n_classes} "
            f"qubits for {n_classes} classes. "
            f"Only {n_qubits} quantum features "
            f"were supplied."
        )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(
        "TRAINING VARIATIONAL QUANTUM CLASSIFIER"
    )
    print("=" * 60)

    print(
        f"Qubits:           {n_qubits}"
    )

    print(
        f"Classes:          {n_classes}"
    )

    print(
        f"Classification:   {classification_type}"
    )

    print(
        f"Layers:           {n_layers}"
    )

    print(
        f"Training data:    {len(X_train)}"
    )

    print(
        f"Test data:        {len(X_test)}"
    )

    print(
        f"Epochs:           {epochs}"
    )

    print(
        f"Learning rate:    {learning_rate}"
    )

    # --------------------------------------------------------
    # Scale features
    #
    # Fit ONLY on training data.
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
    # PennyLane training data
    # --------------------------------------------------------

    X_train_q = qml.numpy.array(
        X_train_scaled,
        requires_grad=False,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = VQCModel(
        n_qubits=n_qubits,
        n_classes=n_classes,
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

    print()
    print(
        "Starting VQC optimization..."
    )

    training_start = (
        time.perf_counter()
    )

    for epoch in range(
        int(epochs)
    ):

        def objective(
            current_weights
        ):

            return multiclass_cross_entropy(
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
            np.asarray(
                loss
            )
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

    print()
    print(
        "Evaluating VQC..."
    )

    evaluation_start = (
        time.perf_counter()
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_probabilities = (
        model.predict_proba(
            X_test_scaled,
            weights,
        )
    )

    test_probabilities = np.asarray(
        test_probabilities,
        dtype=float,
    )

    test_predictions = np.argmax(
        test_probabilities,
        axis=1,
    ).astype(int)

    test_metrics = calculate_metrics(
        y_test,
        test_probabilities,
    )

    # --------------------------------------------------------
    # Training set
    # --------------------------------------------------------

    train_probabilities = (
        model.predict_proba(
            X_train_scaled,
            weights,
        )
    )

    train_probabilities = np.asarray(
        train_probabilities,
        dtype=float,
    )

    train_predictions = np.argmax(
        train_probabilities,
        axis=1,
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
    print("=" * 60)
    print(
        "VQC RESULTS"
    )
    print("=" * 60)

    print(
        f"Classification:   {classification_type}"
    )

    print(
        f"Classes:          {n_classes}"
    )

    print(
        f"Accuracy:         "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision:        "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Sensitivity:      "
        f"{test_metrics['sensitivity']:.4f}"
    )

    print(
        f"Specificity:      "
        f"{test_metrics['specificity']:.4f}"
    )

    print(
        f"F1 Score:         "
        f"{test_metrics['f1']:.4f}"
    )

    if (
        test_metrics["roc_auc"]
        is not None
    ):

        print(
            f"ROC-AUC:          "
            f"{test_metrics['roc_auc']:.4f}"
        )

    else:

        print(
            "ROC-AUC:          N/A"
        )

    print(
        f"Training time:    "
        f"{training_time:.2f}s"
    )

    print(
        f"Evaluation time:  "
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

        "n_classes": n_classes,

        "classification_type": (
            classification_type
        ),

        "classes": (
            all_classes.tolist()
        ),

        "n_layers": int(
            n_layers
        ),

        "learning_rate": float(
            learning_rate
        ),

        "epochs": int(
            epochs
        ),

        "metrics": test_metrics,

        "train_metrics": train_metrics,

        "training_time": float(
            training_time
        ),

        "evaluation_time": float(
            evaluation_time
        ),

        "loss_history": [
            float(value)
            for value in loss_history
        ],

        "scaler": scaler,

        "weights": weights,

        "X_train": X_train,

        "X_test": X_test,

        "y_train": y_train,

        "y_test": y_test,

        "train_probabilities": np.asarray(
            train_probabilities,
            dtype=float,
        ),

        "test_probabilities": np.asarray(
            test_probabilities,
            dtype=float,
        ),

        "train_predictions": np.asarray(
            train_predictions,
            dtype=int,
        ),

        "test_predictions": np.asarray(
            test_predictions,
            dtype=int,
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
    # Encode target
    # --------------------------------------------------------

    unique_targets = sorted(
        y.unique()
    )

    target_mapping = {
        value: index
        for index, value in enumerate(
            unique_targets
        )
    }

    y_encoded = y.map(
        target_mapping
    ).astype(int)

    # --------------------------------------------------------
    # Quantum dimensionality reduction
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
        y_encoded,
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
    print("=" * 60)
    print(
        "STANDALONE VQC TEST COMPLETE"
    )
    print("=" * 60)

    print(
        f"Quantum features: "
        f"{quantum_feature_names}"
    )

    print(
        f"Qubits: "
        f"{result['n_qubits']}"
    )

    print(
        f"Classes: "
        f"{result['n_classes']}"
    )

    print(
        f"Classification: "
        f"{result['classification_type']}"
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