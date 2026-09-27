"""
VITALIS Training Pipeline
-------------------------

End-to-end training orchestration for uploaded biomedical datasets.

Pipeline:

    Raw dataset
        ↓
    Data cleaning
        ↓
    ONE train / test split
        ↓
    Feature selection (TRAIN ONLY)
        ↓
    ┌──────────────────────────────┐
    │                              │
    │ Classical preprocessing      │
    │ + classical models           │
    │                              │
    └──────────────────────────────┘
        ↓
    Quantum dimensionality reduction
        ↓
    Stratified quantum computation subset
        ↓
    QSVM + VQC
        ↓
    Unified results

Important:
    - ONE main train/test split is created here.
    - Classical models use the complete held-out test set.
    - Feature selection is fitted ONLY on training data.
    - Classical preprocessing is fitted ONLY on training data.
    - Quantum dimensionality reduction is fitted ONLY on training data.
    - Quantum models use controlled stratified subsets because
      quantum kernel computation scales poorly with sample count.
    - Mixed numeric/categorical biomedical data is supported.
    - High-dimensional input is reduced before quantum encoding.
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# =====================================================================
# VITALIS MODULES
# =====================================================================

from backend.data_cleaner import clean_dataset
from backend.feature_selector import select_features
from backend.quantum_reducer import QuantumFeatureReducer

from backend.train_classical import train_classical
from backend.train_quantum import train_quantum
from backend.train_vqc import train_vqc


# =====================================================================
# DEFAULT CONFIGURATION
# =====================================================================

DEFAULT_TEST_SIZE = 0.20
DEFAULT_RANDOM_STATE = 42
DEFAULT_QUANTUM_COMPONENTS = 6
DEFAULT_MAX_FEATURES = 15

# Quantum models are computationally expensive.
# Classical models still use the complete dataset.
DEFAULT_MAX_QUANTUM_TRAIN_SAMPLES = 500
DEFAULT_MAX_QUANTUM_TEST_SAMPLES = 500


# =====================================================================
# VALIDATION
# =====================================================================

def _validate_inputs(
    df,
    target_column,
):
    """
    Validate the input dataset.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "Dataset is empty."
        )

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found."
        )

    if df[target_column].nunique() < 2:
        raise ValueError(
            "Target must contain at least two classes."
        )


# =====================================================================
# TARGET VALIDATION
# =====================================================================

def _prepare_target(
    y,
):
    """
    Validate and encode target values.

    The pipeline internally uses consecutive integer labels:
        0, 1, 2, ...

    Original class labels are preserved by the LabelEncoder so that
    predictions can later be decoded back to the dataset's labels.
    """

    y = pd.Series(y).copy()

    if y.isna().any():
        raise ValueError(
            "Target contains missing values."
        )

    if y.nunique() < 2:
        raise ValueError(
            "Target must contain at least two classes."
        )

    # Convert arbitrary biomedical labels such as:
    #   yes / no
    #   benign / malignant
    #   <30 / >30 / NO
    # into consecutive integer labels required by the quantum models.
    target_encoder = LabelEncoder()

    encoded_values = target_encoder.fit_transform(
        y.astype(str)
    )

    y_encoded = pd.Series(
        encoded_values,
        index=y.index,
        name=y.name,
    )

    print()
    print("Target encoding:")

    for encoded_value, original_value in enumerate(
        target_encoder.classes_
    ):
        print(
            f"  {encoded_value} -> {original_value}"
        )

    return y_encoded, target_encoder


# =====================================================================
# TRAIN / TEST SPLIT
# =====================================================================

def _split_dataset(
    X,
    y,
    test_size=DEFAULT_TEST_SIZE,
    random_state=DEFAULT_RANDOM_STATE,
):
    """
    Perform the ONE main stratified train/test split used by
    the complete VITALIS pipeline.
    """

    try:

        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

    except ValueError as exc:

        raise ValueError(
            f"Unable to perform stratified train/test split: {exc}"
        ) from exc


# =====================================================================
# QUANTUM SUBSET
# =====================================================================

def _make_quantum_subset(
    X,
    y,
    max_samples,
    random_state,
    label,
):
    """
    Create a stratified subset for computationally expensive
    quantum models.

    The full classical dataset is never reduced by this function.
    """

    X = X.copy()
    y = pd.Series(y, index=X.index).copy()

    if max_samples is None:
        return X, y

    max_samples = int(max_samples)

    if max_samples <= 0:
        raise ValueError(
            f"{label} quantum sample limit must be greater than zero."
        )

    if len(X) <= max_samples:
        print(
            f"  {label}: using all {len(X)} samples"
        )

        return X, y

    try:

        X_subset, _, y_subset, _ = train_test_split(
            X,
            y,
            train_size=max_samples,
            random_state=random_state,
            stratify=y,
        )

    except ValueError as exc:

        raise ValueError(
            f"Unable to create stratified {label.lower()} "
            f"quantum subset: {exc}"
        ) from exc

    print(
        f"  {label}: {len(X_subset)} / {len(X)} samples"
    )

    return X_subset, y_subset


# =====================================================================
# QUANTUM DATAFRAME HELPER
# =====================================================================

def _make_quantum_dataframe(
    X_reduced,
    reducer,
    index=None,
):
    """
    Convert quantum-reduction output into a DataFrame with
    stable quantum feature names.
    """

    feature_names = reducer.get_feature_names()

    return pd.DataFrame(
        X_reduced,
        columns=feature_names,
        index=index,
    )


# =====================================================================
# MAIN PIPELINE
# =====================================================================

def run_training_pipeline(
    df,
    target_column,
    quantum_components=DEFAULT_QUANTUM_COMPONENTS,
    test_size=DEFAULT_TEST_SIZE,
    random_state=DEFAULT_RANDOM_STATE,
    max_quantum_train_samples=DEFAULT_MAX_QUANTUM_TRAIN_SAMPLES,
    max_quantum_test_samples=DEFAULT_MAX_QUANTUM_TEST_SAMPLES,
):
    """
    Run the complete VITALIS training pipeline.

    Parameters
    ----------
    df : pandas.DataFrame
        Uploaded biomedical dataset.

    target_column : str
        Name of the target column.

    quantum_components : int
        Desired number of quantum dimensions/qubits.

    test_size : float
        Fraction of data used for testing.

    random_state : int
        Random seed.

    max_quantum_train_samples : int
        Maximum number of training samples used by QSVM/VQC.

    max_quantum_test_samples : int
        Maximum number of test samples used by QSVM/VQC.

    Returns
    -------
    dict
        Complete pipeline result.
    """

    # =================================================================
    # 1. VALIDATE INPUT
    # =================================================================

    _validate_inputs(
        df,
        target_column,
    )

    print("=" * 70)
    print("VITALIS TRAINING PIPELINE")
    print("=" * 70)

    print()
    print("Input dataset:")

    print(
        f"  Rows:    {df.shape[0]}"
    )

    print(
        f"  Columns: {df.shape[1]}"
    )

    print(
        f"  Target:  {target_column}"
    )

    # =================================================================
    # 2. CLEAN DATASET
    # =================================================================

    print()
    print("[1/7] Cleaning dataset...")

    cleaned_df, cleaning_report = clean_dataset(
        df,
        target_column=target_column,
    )

    if cleaned_df.empty:
        raise ValueError(
            "Dataset became empty after cleaning."
        )

    y, target_encoder = _prepare_target(
        cleaned_df[target_column]
    )

    X = cleaned_df.drop(
        columns=[target_column]
    )

    print(
        f"  Cleaned shape: {cleaned_df.shape}"
    )

    # =================================================================
    # 3. ONE TRAIN / TEST SPLIT
    # =================================================================

    print()
    print(
        "[2/7] Creating ONE shared train/test split..."
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = _split_dataset(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    print(
        f"  Training samples: {len(X_train)}"
    )

    print(
        f"  Testing samples:  {len(X_test)}"
    )

    # =================================================================
    # 4. FEATURE SELECTION
    # =================================================================

    print()
    print(
        "[3/7] Selecting features from TRAINING data..."
    )

    (
        selected_X_train,
        selection_report,
    ) = select_features(
        X_train,
        y_train,
        target_column=None,
        max_features=DEFAULT_MAX_FEATURES,
    )

    selected_features = (
        selected_X_train.columns.tolist()
    )

    if len(selected_features) == 0:
        raise ValueError(
            "Feature selection produced no usable features."
        )

    # Apply selected feature names to both sets.

    X_train_selected = (
        X_train[selected_features].copy()
    )

    X_test_selected = (
        X_test[selected_features].copy()
    )

    print(
        f"  Original features: "
        f"{X_train.shape[1]}"
    )

    print(
        f"  Selected features: "
        f"{len(selected_features)}"
    )

    print()
    print("  Selected:")

    for feature in selected_features:

        print(
            f"    - {feature}"
        )

    # =================================================================
    # 5. CLASSICAL MODELS
    # =================================================================

    print()
    print(
        "[4/7] Training classical models..."
    )

    # train_classical handles:
    #   - numeric features
    #   - categorical features
    #   - missing values
    #   - scaling
    #   - one-hot encoding
    #   - binary targets
    #   - multiclass targets
    #
    # Preprocessing is fitted ONLY on X_train_selected.

    classical_result = train_classical(
        X_train_selected,
        y_train,
        X_test_selected,
        y_test,
        random_state=random_state,
    )

    classical_models = (
        classical_result["models"]
    )

    classical_preprocessor = (
        classical_result.get(
            "classical_preprocessor"
        )
    )

    # Backward-compatible field.

    classical_scaler = (
        classical_result.get(
            "scaler"
        )
    )

    classical_metrics = (
        classical_result["metrics"]
    )

    print()
    print(
        "  Classical models trained:"
    )

    for model_name in classical_models:

        print(
            f"    - {model_name}"
        )

    # =================================================================
    # 6. QUANTUM DIMENSIONALITY REDUCTION
    # =================================================================

    print()
    print(
        "[5/7] Performing quantum dimensionality reduction..."
    )

    reducer = QuantumFeatureReducer(
        n_components=quantum_components,
        random_state=random_state,
    )

    # IMPORTANT:
    # Fit ONLY on training data.
    #
    # The reducer handles:
    #   - numeric features
    #   - categorical features
    #   - missing values
    #   - encoding
    #   - scaling
    #   - PCA
    #
    # Pass RAW selected features.

    X_train_quantum_array = (
        reducer.fit_transform(
            X_train_selected
        )
    )

    # Transform test using fitted reducer.

    X_test_quantum_array = (
        reducer.transform(
            X_test_selected
        )
    )

    X_train_quantum = (
        _make_quantum_dataframe(
            X_train_quantum_array,
            reducer,
            index=X_train_selected.index,
        )
    )

    X_test_quantum = (
        _make_quantum_dataframe(
            X_test_quantum_array,
            reducer,
            index=X_test_selected.index,
        )
    )

    reducer_metadata = (
        reducer.get_metadata()
    )

    print(
        f"  Original input features: "
        f"{X_train.shape[1]}"
    )

    print(
        f"  Selected features: "
        f"{len(selected_features)}"
    )

    print(
        f"  Quantum dimensions: "
        f"{reducer_metadata['actual_components']}"
    )

    print(
        f"  Full training quantum shape: "
        f"{X_train_quantum.shape}"
    )

    print(
        f"  Full testing quantum shape: "
        f"{X_test_quantum.shape}"
    )

    # =================================================================
    # QUANTUM COMPUTATION SUBSETS
    # =================================================================
    #
    # The full classical benchmark remains unchanged.
    #
    # QSVM/VQC use stratified subsets because quantum kernel methods
    # scale very poorly with sample count.
    #
    # This is especially important for datasets such as diabetes:
    #
    #   81,412 classical training samples
    #   20,354 classical test samples
    #
    # would create an impractical full quantum kernel workload.
    # =================================================================

    print()
    print(
        "Quantum computation subset:"
    )

    X_quantum_train, y_quantum_train = (
        _make_quantum_subset(
            X_train_quantum,
            y_train,
            max_quantum_train_samples,
            random_state,
            "Training",
        )
    )

    X_quantum_test, y_quantum_test = (
        _make_quantum_subset(
            X_test_quantum,
            y_test,
            max_quantum_test_samples,
            random_state,
            "Testing",
        )
    )

    print(
        f"  Quantum training shape: "
        f"{X_quantum_train.shape}"
    )

    print(
        f"  Quantum testing shape: "
        f"{X_quantum_test.shape}"
    )

    # =================================================================
    # 7. QUANTUM KERNEL SVM
    # =================================================================

    print()
    print(
        "[6/7] Training Quantum Kernel SVM..."
    )

    print()
    print("  Quantum target labels:")
    print(
        f"    Train: {sorted(pd.Series(y_quantum_train).unique().tolist())}"
    )
    print(
        f"    Test:  {sorted(pd.Series(y_quantum_test).unique().tolist())}"
    )
    print(
        f"    Classes: {target_encoder.classes_.tolist()}"
    )

    quantum_result = train_quantum(
        X_quantum_train,
        y_quantum_train,
        X_quantum_test,
        y_quantum_test,
        quantum_features=(
            X_quantum_train.columns.tolist()
        ),
        n_qubits=(
            reducer_metadata[
                "actual_components"
            ]
        ),
        random_state=random_state,
    )

    # =================================================================
    # 8. VQC
    # =================================================================

    print()
    print(
        "[7/7] Training Variational Quantum Classifier..."
    )

    vqc_result = train_vqc(
        X_quantum_train,
        y_quantum_train,
        X_quantum_test,
        y_quantum_test,
        random_state=random_state,
    )

    # =================================================================
    # FINAL RESULT
    # =================================================================

    result = {

        "status": "trained",

        "target": {
            "classes": (
                target_encoder.classes_.tolist()
            ),
            "encoding": {
                str(index): str(label)
                for index, label in enumerate(
                    target_encoder.classes_
                )
            },
        },

        "dataset": {

            "rows": int(
                cleaned_df.shape[0]
            ),

            "columns": int(
                cleaned_df.shape[1]
            ),

            "target_column": (
                target_column
            ),

            "original_feature_count": int(
                X_train.shape[1]
            ),

            "selected_feature_count": int(
                len(selected_features)
            ),
        },

        # -------------------------------------------------------------
        # Shared split
        # -------------------------------------------------------------

        "split": {

            "train_samples": int(
                len(X_train)
            ),

            "test_samples": int(
                len(X_test)
            ),

            "test_size": float(
                test_size
            ),

            "random_state": int(
                random_state
            ),
        },

        # -------------------------------------------------------------
        # Preprocessing
        # -------------------------------------------------------------

        "preprocessing": {

            "cleaning": (
                cleaning_report
            ),

            "feature_selection": (
                selection_report
            ),

            "selected_features": (
                selected_features
            ),

            "feature_count": len(
                selected_features
            ),

            "original_feature_count": int(
                X_train.shape[1]
            ),

            "classical_preprocessing": (
                classical_result.get(
                    "preprocessing_metadata",
                    {},
                )
            ),
        },

        # -------------------------------------------------------------
        # Quantum reduction
        # -------------------------------------------------------------

        "quantum_reduction": (
            reducer_metadata
        ),

        # -------------------------------------------------------------
        # Classical models
        # -------------------------------------------------------------

        "classical": {

            "models": list(
                classical_models.keys()
            ),

            "metrics": (
                classical_metrics
            ),

            "classification_type": (
                classical_result.get(
                    "classification_type"
                )
            ),

            "num_classes": (
                classical_result.get(
                    "num_classes"
                )
            ),

            "classes": (
                classical_result.get(
                    "classes"
                )
            ),
        },

        # -------------------------------------------------------------
        # Quantum models
        # -------------------------------------------------------------

        "quantum": {

            "models": [
                "Quantum Kernel SVM",
                "Variational Quantum Classifier",
            ],

            "computation_subset": {

                "max_train_samples": int(
                    max_quantum_train_samples
                ),

                "max_test_samples": int(
                    max_quantum_test_samples
                ),

                "actual_train_samples": int(
                    len(X_quantum_train)
                ),

                "actual_test_samples": int(
                    len(X_quantum_test)
                ),

                "sampling": "stratified",
            },

            "qsvm": {

                "model": (
                    "Quantum Kernel SVM"
                ),

                "features": (
                    X_quantum_train.columns.tolist()
                ),

                "n_qubits": (
                    reducer_metadata[
                        "actual_components"
                    ]
                ),

                "metrics": (
                    quantum_result[
                        "metrics"
                    ]
                ),

                "training_time": (
                    quantum_result[
                        "training_time"
                    ]
                ),

                "evaluation_time": (
                    quantum_result[
                        "evaluation_time"
                    ]
                ),
            },

            "vqc": {

                "model": (
                    "Variational Quantum Classifier"
                ),

                "features": (
                    X_quantum_train.columns.tolist()
                ),

                "n_qubits": (
                    vqc_result[
                        "n_qubits"
                    ]
                ),

                "n_layers": (
                    vqc_result[
                        "n_layers"
                    ]
                ),

                "metrics": (
                    vqc_result[
                        "metrics"
                    ]
                ),

                "training_time": (
                    vqc_result[
                        "training_time"
                    ]
                ),

                "evaluation_time": (
                    vqc_result[
                        "evaluation_time"
                    ]
                ),
            },
        },

        # -------------------------------------------------------------
        # Fitted artifacts
        # -------------------------------------------------------------

        "_artifacts": {

            "cleaned_dataframe": (
                cleaned_df
            ),

            # Target encoding
            "target_encoder": target_encoder,

            "target_classes": (
                target_encoder.classes_.tolist()
            ),

            # Shared original split

            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,

            # Feature selection

            "selected_features": (
                selected_features
            ),

            "selection_report": (
                selection_report
            ),

            "X_train_selected": (
                X_train_selected
            ),

            "X_test_selected": (
                X_test_selected
            ),

            # Classical preprocessing

            "classical_preprocessor": (
                classical_preprocessor
            ),

            "classical_result": (
                classical_result
            ),

            # Backward compatibility

            "classical_scaler": (
                classical_scaler
            ),

            # Classical models

            "classical_models": (
                classical_models
            ),

            # Quantum reduction

            "reducer": reducer,

            # Full reduced data

            "X_train_quantum": (
                X_train_quantum
            ),

            "X_test_quantum": (
                X_test_quantum
            ),

            # Quantum computation subsets

            "X_quantum_train": (
                X_quantum_train
            ),

            "X_quantum_test": (
                X_quantum_test
            ),

            "y_quantum_train": (
                y_quantum_train
            ),

            "y_quantum_test": (
                y_quantum_test
            ),

            # QSVM

            "quantum_result": (
                quantum_result
            ),

            # VQC

            "vqc_result": (
                vqc_result
            ),
        },
    }

    # =================================================================
    # SUMMARY
    # =================================================================

    print()
    print("=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    print()
    print(
        "Shared evaluation set:"
    )

    print(
        f"  Train: {len(X_train)}"
    )

    print(
        f"  Test:  {len(X_test)}"
    )

    print()
    print(
        "Feature-space transformation:"
    )

    print(
        f"  Original:  {X_train.shape[1]}"
    )

    print(
        f"  Selected:  {len(selected_features)}"
    )

    print(
        f"  Quantum:   "
        f"{reducer_metadata['actual_components']}"
    )

    print()
    print(
        "Quantum computation subset:"
    )

    print(
        f"  Train: "
        f"{len(X_quantum_train)}"
    )

    print(
        f"  Test:  "
        f"{len(X_quantum_test)}"
    )

    print()
    print(
        "Classical models:"
    )

    for model_name in classical_metrics:

        print(
            f"  {model_name}: "
            f"accuracy="
            f"{classical_metrics[model_name]['accuracy']:.4f}"
        )

    print()
    print(
        "Quantum Kernel SVM:"
    )

    print(
        f"  Accuracy: "
        f"{quantum_result['metrics']['accuracy']:.4f}"
    )

    print()
    print(
        "Variational Quantum Classifier:"
    )

    print(
        f"  Accuracy: "
        f"{vqc_result['metrics']['accuracy']:.4f}"
    )

    print()
    print(
        "Quantum dimensions:"
    )

    print(
        f"  {reducer_metadata['actual_components']}"
    )

    print()
    print(
        "STATUS: training pipeline successful"
    )

    return result


# =====================================================================
# STANDALONE TEST
# =====================================================================

def standalone_test():
    """
    Run the complete pipeline using the heart dataset.
    """

    dataset_path = (
        "data/heart_cleaned.csv"
    )

    print(
        f"\nLoading dataset: "
        f"{dataset_path}"
    )

    df = pd.read_csv(
        dataset_path
    )

    result = run_training_pipeline(
        df=df,
        target_column="target",
        quantum_components=6,
        test_size=0.20,
        random_state=42,
    )

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print()

    print(
        "Selected features:"
    )

    for feature in result[
        "preprocessing"
    ][
        "selected_features"
    ]:

        print(
            f"  - {feature}"
        )

    print()

    print(
        "Feature reduction:"
    )

    print(
        "  Original:",
        result[
            "dataset"
        ][
            "original_feature_count"
        ],
    )

    print(
        "  Selected:",
        result[
            "dataset"
        ][
            "selected_feature_count"
        ],
    )

    print(
        "  Quantum components:",
        result[
            "quantum_reduction"
        ][
            "actual_components"
        ],
    )

    print()

    print(
        "Classical models:",
        result[
            "classical"
        ][
            "models"
        ],
    )

    print()

    print(
        "Quantum models:",
        result[
            "quantum"
        ][
            "models"
        ],
    )


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":

    standalone_test()