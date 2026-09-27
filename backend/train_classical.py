"""
VITALIS - Generic Classical Machine Learning Training Engine

Supported models:
    - Logistic Regression
    - Random Forest
    - RBF SVM

Features:
    - Numeric + categorical feature support
    - Automatic preprocessing
    - Missing-value handling
    - Binary + multiclass classification
    - Reusable fitted preprocessing artifact
    - Accuracy / precision / recall / F1
    - Specificity
    - ROC-AUC when applicable
    - Training and evaluation timing
    - Scalable RBF SVM training for large datasets

The training functions support two modes:

1. Pre-split mode:
       train_classical(X_train, y_train, X_test, y_test)

   Preferred mode for the unified VITALIS pipeline.

2. Automatic split mode:
       train_classical(X, y)

   Kept for backward compatibility and standalone testing.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
)
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

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

# RBF SVM does not scale well with very large datasets.
#
# Small datasets use all available training samples.
# Large datasets use a stratified subset for the RBF SVM only.
#
# The classical benchmark remains valid because the other
# classical models still train on the complete training set.
RBF_SVM_MAX_TRAIN_SAMPLES = 5000


# ============================================================
# DATASET LOADER
# ============================================================

def load_dataset(
    path="data/heart_cleaned.csv"
):
    """
    Load a labeled CSV dataset.

    This loader is retained mainly for standalone testing.

    The generic training engine itself does not depend on
    the heart dataset.
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

    Supports:
        - numeric features
        - categorical features
        - binary targets
        - multiclass targets
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
    # Target validation
    # --------------------------------------------------------

    y_train_array = np.asarray(
        y_train
    )

    y_test_array = np.asarray(
        y_test
    )

    if pd.isna(y_train_array).any():
        raise ValueError(
            "Training target contains missing values."
        )

    if pd.isna(y_test_array).any():
        raise ValueError(
            "Test target contains missing values."
        )

    train_classes = np.unique(
        y_train_array
    )

    test_classes = np.unique(
        y_test_array
    )

    if len(train_classes) < 2:
        raise ValueError(
            "Training target must contain at least "
            "two classes."
        )

    if len(test_classes) < 2:
        raise ValueError(
            "Test target must contain at least "
            "two classes."
        )

    if not set(test_classes).issubset(
        set(train_classes)
    ):
        raise ValueError(
            "Test target contains classes that are "
            "not present in the training target."
        )


# ============================================================
# CLASSICAL PREPROCESSOR
# ============================================================

def _build_preprocessor(X_train):
    """
    Build a preprocessing pipeline for mixed biomedical data.

    Numeric:
        missing values -> median
        scaling -> StandardScaler

    Categorical:
        missing values -> most frequent
        encoding -> OneHotEncoder

    The returned preprocessor must be fitted ONLY on
    training data and reused during inference.
    """

    numeric_features = (
        X_train.select_dtypes(
            include=[
                "number",
                "bool",
            ]
        ).columns.tolist()
    )

    categorical_features = (
        X_train.select_dtypes(
            include=[
                "object",
                "category",
            ]
        ).columns.tolist()
    )

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    transformers = []

    if numeric_features:
        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )

    if not transformers:
        raise ValueError(
            "No supported numeric or categorical "
            "features were found."
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    metadata = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }

    return preprocessor, metadata


# ============================================================
# STRATIFIED MODEL SUBSET
# ============================================================

def _make_stratified_subset(
    X,
    y,
    max_samples,
    random_state,
):
    """
    Create a stratified subset while preserving class proportions.

    Used only for the scalable RBF SVM path.
    """

    if X.shape[0] <= max_samples:
        return X, y

    X_subset, _, y_subset, _ = train_test_split(
        X,
        y,
        train_size=max_samples,
        random_state=random_state,
        stratify=y,
    )

    return X_subset, y_subset


# ============================================================
# METRICS
# ============================================================

def _calculate_specificity_binary(
    y_true,
    predictions,
):
    """
    Calculate binary specificity.
    """

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    if (tn + fp) == 0:
        return 0.0

    return float(
        tn / (tn + fp)
    )


def _calculate_specificity_multiclass(
    y_true,
    predictions,
    classes,
):
    """
    Calculate macro-average one-vs-rest specificity
    for multiclass classification.
    """

    specificities = []

    for class_value in classes:

        y_true_binary = (
            np.asarray(y_true)
            == class_value
        ).astype(int)

        predictions_binary = (
            np.asarray(predictions)
            == class_value
        ).astype(int)

        cm = confusion_matrix(
            y_true_binary,
            predictions_binary,
            labels=[0, 1],
        )

        tn, fp, fn, tp = cm.ravel()

        if (tn + fp) == 0:
            continue

        specificities.append(
            tn / (tn + fp)
        )

    if not specificities:
        return 0.0

    return float(
        np.mean(specificities)
    )


def calculate_metrics(
    y_true,
    predictions,
    probabilities=None,
    classes=None,
):
    """
    Calculate classification metrics.

    Binary classification:
        accuracy
        precision
        sensitivity
        specificity
        f1
        roc_auc

    Multiclass classification:
        accuracy
        macro precision
        macro sensitivity
        macro specificity
        macro F1
        multiclass ROC-AUC when probabilities are available
    """

    y_true = np.asarray(
        y_true
    )

    predictions = np.asarray(
        predictions
    )

    if classes is None:
        classes = np.unique(
            y_true
        )

    classes = np.asarray(
        classes
    )

    is_binary = len(classes) == 2

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    if is_binary:

        positive_class = classes[1]

        precision = precision_score(
            y_true,
            predictions,
            pos_label=positive_class,
            zero_division=0,
        )

        sensitivity = recall_score(
            y_true,
            predictions,
            pos_label=positive_class,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            predictions,
            pos_label=positive_class,
            zero_division=0,
        )

        specificity = (
            _calculate_specificity_binary(
                y_true,
                predictions,
            )
        )

    else:

        precision = precision_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0,
        )

        sensitivity = recall_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0,
        )

        specificity = (
            _calculate_specificity_multiclass(
                y_true,
                predictions,
                classes,
            )
        )

    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    roc_auc = None

    if probabilities is not None:

        try:

            if is_binary:

                probability_array = np.asarray(
                    probabilities
                )

                if (
                    probability_array.ndim == 2
                    and probability_array.shape[1] >= 2
                ):
                    positive_probabilities = (
                        probability_array[:, 1]
                    )
                else:
                    positive_probabilities = (
                        probability_array
                    )

                roc_auc = roc_auc_score(
                    y_true,
                    positive_probabilities,
                )

            else:

                probability_array = np.asarray(
                    probabilities
                )

                if (
                    probability_array.ndim == 2
                    and probability_array.shape[1]
                    == len(classes)
                ):

                    roc_auc = roc_auc_score(
                        y_true,
                        probability_array,
                        multi_class="ovr",
                        average="macro",
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

        "classification_type": (
            "binary"
            if is_binary
            else "multiclass"
        ),

        "num_classes": int(
            len(classes)
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

    For large datasets, only the RBF SVM uses a capped,
    stratified training subset. Logistic Regression and
    Random Forest continue to use the complete training set.

    Returns:

        {
            "models": ...,
            "preprocessor": ...,
            "classical_preprocessor": ...,
            "scaler": ...,
            "preprocessing_metadata": ...,
            "metrics": ...,
            "predictions": ...,
            "probabilities": ...,
            "training_times": ...,
            "evaluation_times": ...,
            "model_training_metadata": ...,
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
        y_train
    )

    y_test = np.asarray(
        y_test
    )

    # ========================================================
    # TARGET CLASS INFORMATION
    # ========================================================

    classes = np.unique(
        y_train
    )

    num_classes = len(
        classes
    )

    print()
    print("=" * 60)
    print(
        "CLASSICAL TARGET INFORMATION"
    )
    print("=" * 60)

    print(
        f"Classes:       {classes.tolist()}"
    )

    print(
        f"Num classes:   {num_classes}"
    )

    print(
        "Type:           "
        + (
            "binary"
            if num_classes == 2
            else "multiclass"
        )
    )

    print()

    # ========================================================
    # BUILD + FIT PREPROCESSOR
    # ========================================================

    preprocessor, preprocessing_metadata = (
        _build_preprocessor(
            X_train
        )
    )

    print("=" * 60)
    print(
        "CLASSICAL PREPROCESSING"
    )
    print("=" * 60)

    print(
        "Numeric features:"
    )

    if preprocessing_metadata[
        "numeric_features"
    ]:

        for feature in preprocessing_metadata[
            "numeric_features"
        ]:

            print(
                f"  - {feature}"
            )

    else:

        print("  None")

    print()

    print(
        "Categorical features:"
    )

    if preprocessing_metadata[
        "categorical_features"
    ]:

        for feature in preprocessing_metadata[
            "categorical_features"
        ]:

            print(
                f"  - {feature}"
            )

    else:

        print("  None")

    print()

    preprocessing_start = (
        time.perf_counter()
    )

    X_train_processed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    X_test_processed = (
        preprocessor.transform(
            X_test
        )
    )

    preprocessing_time = (
        time.perf_counter()
        - preprocessing_start
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
    }

    # ========================================================
    # RBF SVM SCALABILITY
    # ========================================================

    rbf_train_limit = min(
        len(X_train),
        RBF_SVM_MAX_TRAIN_SAMPLES,
    )

    if len(X_train) > RBF_SVM_MAX_TRAIN_SAMPLES:

        (
            X_rbf_train,
            y_rbf_train,
        ) = _make_stratified_subset(
            X_train_processed,
            y_train,
            RBF_SVM_MAX_TRAIN_SAMPLES,
            random_state,
        )

        rbf_subset_used = True

    else:

        X_rbf_train = X_train_processed
        y_rbf_train = y_train

        rbf_subset_used = False

    models["RBF SVM"] = SVC(
        kernel="rbf",
        probability=True,
        random_state=random_state,
        cache_size=2000,
    )

    # ========================================================
    # STORAGE
    # ========================================================

    trained_models = {}

    metrics = {}

    predictions = {}

    probabilities = {}

    training_times = {}

    evaluation_times = {}

    model_training_metadata = {}

    # ========================================================
    # DISPLAY
    # ========================================================

    print("=" * 60)
    print(
        "TRAINING CLASSICAL MODELS"
    )
    print("=" * 60)

    print(
        f"Training data:       {len(X_train)}"
    )

    print(
        f"Test data:           {len(X_test)}"
    )

    print(
        f"Original features:   {X_train.shape[1]}"
    )

    print(
        "Processed features:  "
        f"{X_train_processed.shape[1]}"
    )

    print(
        f"Preprocessing time:  "
        f"{preprocessing_time:.4f}s"
    )

    print()

    if rbf_subset_used:

        print(
            "RBF SVM scalability:"
        )

        print(
            f"  Full training set: "
            f"{len(X_train)} samples"
        )

        print(
            f"  RBF SVM subset:     "
            f"{len(y_rbf_train)} samples"
        )

        print(
            "  Sampling:           stratified"
        )

        print()

    # ========================================================
    # TRAIN EACH MODEL
    # ========================================================

    for name, model in models.items():

        # ----------------------------------------------------
        # Select training data
        # ----------------------------------------------------

        if name == "RBF SVM":

            model_X_train = X_rbf_train
            model_y_train = y_rbf_train

        else:

            model_X_train = X_train_processed
            model_y_train = y_train

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        training_start = (
            time.perf_counter()
        )

        model.fit(
            model_X_train,
            model_y_train,
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
                X_test_processed
            )
        )

        model_probabilities = None

        if hasattr(
            model,
            "predict_proba",
        ):

            model_probabilities = (
                model.predict_proba(
                    X_test_processed
                )
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
            classes=classes,
        )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        trained_models[name] = model

        metrics[name] = model_metrics

        predictions[name] = (
            model_predictions
        )

        if model_probabilities is not None:

            probabilities[name] = (
                model_probabilities
            )

        training_times[name] = (
            float(training_time)
        )

        evaluation_times[name] = (
            float(evaluation_time)
        )

        model_training_metadata[name] = {
            "training_samples": int(
                len(model_y_train)
            ),

            "full_training_samples": int(
                len(X_train)
            ),

            "used_training_subset": bool(
                name == "RBF SVM"
                and rbf_subset_used
            ),

            "sampling": (
                "stratified"
                if (
                    name == "RBF SVM"
                    and rbf_subset_used
                )
                else "full_training_set"
            ),
        }

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

        if name == "RBF SVM":

            print(
                f"  Train samples:"
                f" {len(model_y_train)}"
            )

        print()

    # ========================================================
    # RETURN
    # ========================================================

    return {

        "models": trained_models,

        # Generic preprocessing artifact.
        "preprocessor": preprocessor,

        # Explicit alias used by the generic backend.
        "classical_preprocessor": preprocessor,

        # Kept for compatibility with older backend code.
        "scaler": None,

        "preprocessing_metadata": (
            preprocessing_metadata
        ),

        "preprocessing_time": float(
            preprocessing_time
        ),

        "metrics": metrics,

        "predictions": predictions,

        "probabilities": probabilities,

        "training_times": training_times,

        "evaluation_times": evaluation_times,

        "model_training_metadata": (
            model_training_metadata
        ),

        "classes": classes.tolist(),

        "num_classes": int(
            num_classes
        ),

        "classification_type": (
            "binary"
            if num_classes == 2
            else "multiclass"
        ),

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
    print("=" * 60)
    print(
        "STANDALONE CLASSICAL TEST COMPLETE"
    )
    print("=" * 60)

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

        if (
            model_metrics["roc_auc"]
            is not None
        ):

            print(
                f"{name}: "
                f"accuracy="
                f"{model_metrics['accuracy']:.4f}, "
                f"ROC-AUC="
                f"{model_metrics['roc_auc']:.4f}"
            )

        else:

            print(
                f"{name}: "
                f"accuracy="
                f"{model_metrics['accuracy']:.4f}"
            )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    standalone_test()