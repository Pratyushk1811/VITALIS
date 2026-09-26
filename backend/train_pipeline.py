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
    Classical models
        ↓
    Quantum dimensionality reduction (TRAIN ONLY)
        ↓
    QSVM + VQC
        ↓
    Unified results

Important:
    - ONE train/test split is created here.
    - Every model receives the SAME held-out test set.
    - Feature selection is fitted ONLY on training data.
    - Quantum dimensionality reduction is fitted ONLY on training data.
    - No integrated model creates a second split.
"""

import pandas as pd

from sklearn.model_selection import train_test_split


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

    if not isinstance(
        df,
        pd.DataFrame,
    ):
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
    Validate target values.
    """

    y = pd.Series(
        y
    ).copy()

    if y.isna().any():
        raise ValueError(
            "Target contains missing values."
        )

    if y.nunique() < 2:
        raise ValueError(
            "Target must contain at least two classes."
        )

    return y


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
    Perform the ONE stratified train/test split used by the
    complete VITALIS pipeline.
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

    feature_names = (
        reducer.get_feature_names()
    )

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

    cleaned_df, cleaning_report = (
        clean_dataset(
            df,
            target_column=target_column,
        )
    )

    if cleaned_df.empty:
        raise ValueError(
            "Dataset became empty after cleaning."
        )

    y = _prepare_target(
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
        X_train[
            selected_features
        ].copy()
    )

    X_test_selected = (
        X_test[
            selected_features
        ].copy()
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

    # train_classical() returns ONE dictionary.
    #
    # We explicitly pass the shared test set so it does not
    # create another split.

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

    classical_scaler = (
        classical_result["scaler"]
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

    X_train_quantum_array = (
        reducer.fit_transform(
            X_train_selected
        )
    )

    # Transform test using the fitted reducer.

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
        f"  Input selected features: "
        f"{len(selected_features)}"
    )

    print(
        f"  Quantum dimensions: "
        f"{reducer_metadata['actual_components']}"
    )

    print(
        f"  Training quantum shape: "
        f"{X_train_quantum.shape}"
    )

    print(
        f"  Testing quantum shape: "
        f"{X_test_quantum.shape}"
    )

    # =================================================================
    # 7. QUANTUM KERNEL SVM
    # =================================================================

    print()
    print(
        "[6/7] Training Quantum Kernel SVM..."
    )

    # Exact same train/test split.

    quantum_result = train_quantum(
        X_train_quantum,
        y_train,
        X_test_quantum,
        y_test,
        quantum_features=(
            X_train_quantum.columns.tolist()
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

    # Exact same train/test split.

    vqc_result = train_vqc(
        X_train_quantum,
        y_train,
        X_test_quantum,
        y_test,
        random_state=random_state,
    )

    # =================================================================
    # FINAL RESULT
    # =================================================================

    result = {

        "status": "trained",

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
        },

        # -------------------------------------------------------------
        # Quantum models
        # -------------------------------------------------------------

        "quantum": {

            "models": [
                "Quantum Kernel SVM",
                "Variational Quantum Classifier",
            ],

            "qsvm": {

                "model": (
                    "Quantum Kernel SVM"
                ),

                "features": (
                    X_train_quantum.columns.tolist()
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
                    X_train_quantum.columns.tolist()
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

            # Quantum reduction
            "reducer": reducer,

            "X_train_quantum": (
                X_train_quantum
            ),

            "X_test_quantum": (
                X_test_quantum
            ),

            # Classical
            "classical_models": (
                classical_models
            ),

            "classical_scaler": (
                classical_scaler
            ),

            "classical_result": (
                classical_result
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
        "Quantum components:",
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