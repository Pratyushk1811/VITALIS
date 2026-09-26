"""
Generic Classical Machine Learning Training Engine

Supported models:
    - Logistic Regression
    - Random Forest
    - RBF SVM

The training functions support two modes:

1. Pre-split mode:
       train_classical(X_train, y_train, X_test, y_test)

   This is the preferred mode for the unified VITALIS pipeline.

2. Automatic split mode:
       train_classical(X, y)

   Kept for backward compatibility and standalone testing.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

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


# ============================================================
# DATASET LOADER
# ============================================================

def load_dataset(
    path="data/heart_cleaned.csv"
):
    """
    Load a labeled CSV dataset.

    This loader is retained mainly for standalone testing.

    The generic training engine itself does not depend on the
    heart dataset.
    """

    df = pd.read_csv(path)

    if "target" not in df.columns:
        raise ValueError(
            "Dataset must contain a 'target' column."
        )

    X = df.drop(
        columns=["target"]
    )

    y = df["target"]

    return X, y


# ============================================================
# INPUT VALIDATION
# ============================================================

def _validate_inputs(
    X_train,
    y_train,
    X_test,
    y_test,
):
    """
    Validate pre-split training and test data.
    """

    if not isinstance(
        X_train,
        pd.DataFrame,
    ):
        raise TypeError(
            "X_train must be a pandas DataFrame."
        )

    if not isinstance(
        X_test,
        pd.DataFrame,
    ):
        raise TypeError(
            "X_test must be a pandas DataFrame."
        )

    if len(X_train) != len(y_train):
        raise ValueError(
            "X_train and y_train must have "
            "the same number of samples."
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            "X_test and y_test must have "
            "the same number of samples."
        )

    if len(X_train) == 0:
        raise ValueError(
            "Training dataset is empty."
        )

    if len(X_test) == 0:
        raise ValueError(
            "Test dataset is empty."
        )

    if X_train.shape[1] == 0:
        raise ValueError(
            "No features available for training."
        )

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            "Training and test datasets must "
            "have the same number of features."
        )

    if list(X_train.columns) != list(
        X_test.columns
    ):
        raise ValueError(
            "Training and test feature columns "
            "must be identical and in the same order."
        )

    # --------------------------------------------------------
    # Numeric feature validation
    # --------------------------------------------------------

    for column in X_train.columns:

        if not pd.api.types.is_numeric_dtype(
            X_train[column]
        ):
            raise ValueError(
                f"Feature '{column}' must be numeric."
            )

        if not pd.api.types.is_numeric_dtype(
            X_test[column]
        ):
            raise ValueError(
                f"Feature '{column}' must be numeric."
            )

    # --------------------------------------------------------
    # Target validation
    # --------------------------------------------------------

    y_train_array = np.asarray(
        y_train
    )

    y_test_array = np.asarray(
        y_test
    )

    train_classes = np.unique(
        y_train_array
    )

    test_classes = np.unique(
        y_test_array
    )

    if len(train_classes) != 2:
        raise ValueError(
            "Training target must contain exactly "
            "two classes."
        )

    if len(test_classes) != 2:
        raise ValueError(
            "Test target must contain exactly "
            "two classes."
        )

    if not set(train_classes).issubset(
        {0, 1}
    ):
        raise ValueError(
            "Training labels must be encoded as 0 and 1."
        )

    if not set(test_classes).issubset(
        {0, 1}
    ):
        raise ValueError(
            "Test labels must be encoded as 0 and 1."
        )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    predictions,
    probabilities=None,
):
    """
    Calculate binary classification metrics.

    Returns:
        accuracy
        precision
        sensitivity
        specificity
        f1
        roc_auc
    """

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    predictions = np.asarray(
        predictions,
        dtype=int,
    )

    # --------------------------------------------------------
    # Basic metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    sensitivity = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

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

    roc_auc = None

    if probabilities is not None:

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
# TRAIN CLASSICAL MODELS
# ============================================================

def train_classical(
    X_train,
    y_train,
    X_test=None,
    y_test=None,
    random_state=DEFAULT_RANDOM_STATE,
    test_size=DEFAULT_TEST_SIZE,
):
    """
    Train the three classical baseline models.

    Preferred usage:

        train_classical(
            X_train,
            y_train,
            X_test,
            y_test,
        )

    Backward-compatible usage:

        train_classical(
            X,
            y,
        )

    Models:

        Logistic Regression
        Random Forest
        RBF SVM

    Returns:

        {
            "models": ...,
            "scaler": ...,
            "metrics": ...,
            "predictions": ...,
            "probabilities": ...,
            "training_times": ...,
            "evaluation_times": ...,
            "train_data": ...,
            "test_data": ...
        }
    """

    # ========================================================
    # BACKWARD-COMPATIBLE AUTOMATIC SPLIT
    # ========================================================

    if X_test is None or y_test is None:

        if X_test is not None or y_test is not None:

            raise ValueError(
                "Both X_test and y_test must be "
                "provided together."
            )

        X = X_train
        y = y_train

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

    # ========================================================
    # VALIDATION
    # ========================================================

    _validate_inputs(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    # Copy data so the caller's DataFrames are not modified.
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

    # ========================================================
    # FEATURE SCALING
    # ========================================================

    scaler = StandardScaler()

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

    # ========================================================
    # MODEL DEFINITIONS
    # ========================================================

    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=1000,
                random_state=random_state,
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=300,
                random_state=random_state,
                n_jobs=-1,
            ),

        "RBF SVM":
            CalibratedClassifierCV(
                estimator=SVC(
                    kernel="rbf",
                    random_state=random_state,
                ),
                method="sigmoid",
                cv=5,
            ),
    }

    # ========================================================
    # STORAGE
    # ========================================================

    trained_models = {}

    metrics = {}

    predictions = {}

    probabilities = {}

    training_times = {}

    evaluation_times = {}

    # ========================================================
    # TRAIN EACH MODEL
    # ========================================================

    print()
    print("=" * 50)
    print(
        "TRAINING CLASSICAL MODELS"
    )
    print("=" * 50)

    print(
        f"Training data: {len(X_train)}"
    )

    print(
        f"Test data:     {len(X_test)}"
    )

    print(
        f"Features:       {X_train.shape[1]}"
    )

    print()

    for name, model in models.items():

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        training_start = (
            time.perf_counter()
        )

        model.fit(
            X_train_scaled,
            y_train,
        )

        training_time = (
            time.perf_counter()
            - training_start
        )

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        evaluation_start = (
            time.perf_counter()
        )

        model_predictions = (
            model.predict(
                X_test_scaled
            )
        )

        # All current models expose
        # predict_proba().
        model_probabilities = (
            model.predict_proba(
                X_test_scaled
            )[:, 1]
        )

        evaluation_time = (
            time.perf_counter()
            - evaluation_start
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        model_metrics = calculate_metrics(
            y_test,
            model_predictions,
            model_probabilities,
        )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        trained_models[name] = model

        metrics[name] = model_metrics

        predictions[name] = (
            model_predictions
        )

        probabilities[name] = (
            model_probabilities
        )

        training_times[name] = (
            float(training_time)
        )

        evaluation_times[name] = (
            float(evaluation_time)
        )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        print(
            f"{name}:"
        )

        print(
            f"  Accuracy:    "
            f"{model_metrics['accuracy']:.4f}"
        )

        print(
            f"  Precision:   "
            f"{model_metrics['precision']:.4f}"
        )

        print(
            f"  Sensitivity: "
            f"{model_metrics['sensitivity']:.4f}"
        )

        print(
            f"  Specificity: "
            f"{model_metrics['specificity']:.4f}"
        )

        print(
            f"  F1:          "
            f"{model_metrics['f1']:.4f}"
        )

        if (
            model_metrics["roc_auc"]
            is not None
        ):

            print(
                f"  ROC-AUC:     "
                f"{model_metrics['roc_auc']:.4f}"
            )

        print(
            f"  Train time:  "
            f"{training_time:.4f}s"
        )

        print(
            f"  Eval time:   "
            f"{evaluation_time:.4f}s"
        )

        print()

    # ========================================================
    # RETURN
    # ========================================================

    return {

        "models": trained_models,

        "scaler": scaler,

        "metrics": metrics,

        "predictions": predictions,

        "probabilities": probabilities,

        "training_times": training_times,

        "evaluation_times": evaluation_times,

        "train_data": {
            "X": X_train,
            "y": y_train,
        },

        "test_data": {
            "X": X_test,
            "y": y_test,
        },
    }


# ============================================================
# STANDALONE TEST
# ============================================================

def standalone_test():

    dataset_path = Path(
        "data/heart_cleaned.csv"
    )

    print(
        f"Loading: {dataset_path}"
    )

    X, y = load_dataset(
        dataset_path
    )

    print(
        f"Dataset shape: {X.shape}"
    )

    # --------------------------------------------------------
    # Create one explicit split
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=DEFAULT_TEST_SIZE,
        random_state=DEFAULT_RANDOM_STATE,
        stratify=y,
    )

    # --------------------------------------------------------
    # Train using the explicit split
    # --------------------------------------------------------

    result = train_classical(
        X_train,
        y_train,
        X_test,
        y_test,
        random_state=DEFAULT_RANDOM_STATE,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 50)
    print(
        "STANDALONE CLASSICAL TEST COMPLETE"
    )
    print("=" * 50)

    print(
        f"Training samples: "
        f"{len(y_train)}"
    )

    print(
        f"Test samples:     "
        f"{len(y_test)}"
    )

    print()

    for name, model_metrics in (
        result["metrics"].items()
    ):

        print(
            f"{name}: "
            f"accuracy="
            f"{model_metrics['accuracy']:.4f}, "
            f"ROC-AUC="
            f"{model_metrics['roc_auc']:.4f}"
            if model_metrics["roc_auc"]
            is not None
            else
            f"{name}: "
            f"accuracy="
            f"{model_metrics['accuracy']:.4f}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    standalone_test()