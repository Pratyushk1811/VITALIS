"""
VITALIS Feature Selector

Generic feature-selection layer for uploaded biomedical datasets.

Responsibilities:
    - remove obvious identifier columns
    - remove constant features
    - handle numeric and categorical features
    - measure feature relevance using mutual information
    - identify highly redundant numeric features
    - rank candidate features
    - select a configurable number of features
    - produce a transparent feature-selection report

This module does NOT:
    - clean missing values
    - train ML models
    - perform quantum encoding
    - modify the original DataFrame

Those responsibilities belong to other layers.
"""

import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder


# -------------------------------------------------------------------
# Default configuration
# -------------------------------------------------------------------

DEFAULT_MAX_FEATURES = 15

DEFAULT_QUANTUM_FEATURES = 6

DEFAULT_CORRELATION_THRESHOLD = 0.90


# -------------------------------------------------------------------
# Column-name helpers
# -------------------------------------------------------------------

ID_NAME_HINTS = {
    "id",
    "patient_id",
    "patientid",
    "record_id",
    "recordid",
    "sample_id",
    "sampleid",
    "subject_id",
    "subjectid",
    "index",
}


def _normalize_column_name(column):
    """
    Normalize a column name for identifier detection.
    """

    return (
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def _looks_like_id(column):
    """
    Determine whether a column name looks like an identifier.
    """

    normalized = _normalize_column_name(
        column
    )

    return (
        normalized in ID_NAME_HINTS
        or normalized.endswith("_id")
        or normalized.endswith("id")
    )


# -------------------------------------------------------------------
# Target encoding
# -------------------------------------------------------------------

def _encode_target(y):
    """
    Convert target values into integer labels for mutual-information
    calculation.

    The original target is not modified.
    """

    target = pd.Series(
        y
    ).reset_index(drop=True)

    if target.isna().any():
        raise ValueError(
            "Feature selection target contains missing values."
        )

    encoder = LabelEncoder()

    encoded = encoder.fit_transform(
        target.astype(str)
    )

    return encoded, encoder


# -------------------------------------------------------------------
# Feature preparation
# -------------------------------------------------------------------

def _prepare_features(
    X,
):
    """
    Convert features into a numeric representation suitable for
    mutual-information calculation.

    Numeric columns:
        converted to numeric.

    Categorical columns:
        temporarily label-encoded.

    Returns:
        prepared DataFrame
        categorical mask
        encoding information
    """

    prepared = pd.DataFrame(
        index=X.index
    )

    categorical_mask = []

    encoding_information = {}

    for column in X.columns:

        series = X[column]

        # -----------------------------------------------------------
        # Numeric feature
        # -----------------------------------------------------------

        if pd.api.types.is_numeric_dtype(
            series
        ):

            values = pd.to_numeric(
                series,
                errors="coerce",
            )

            prepared[column] = values

            categorical_mask.append(
                False
            )

        # -----------------------------------------------------------
        # Categorical feature
        # -----------------------------------------------------------

        else:

            values = (
                series
                .astype(str)
                .fillna("__MISSING__")
            )

            encoder = LabelEncoder()

            encoded = encoder.fit_transform(
                values
            )

            prepared[column] = encoded

            categorical_mask.append(
                True
            )

            encoding_information[
                column
            ] = {
                "type": "categorical",
                "classes": [
                    str(value)
                    for value in encoder.classes_
                ],
            }

    return (
        prepared,
        categorical_mask,
        encoding_information,
    )


# -------------------------------------------------------------------
# Remove unusable features
# -------------------------------------------------------------------

def _remove_unusable_features(
    X,
    target_column,
):
    """
    Identify obvious features that should not participate in
    feature selection.

    Reasons:
        - target column
        - constant columns
        - obvious ID columns
    """

    removed = []

    remaining = X.copy()

    for column in list(
        remaining.columns
    ):

        # -----------------------------------------------------------
        # Never use target as a feature
        # -----------------------------------------------------------

        if column == target_column:

            remaining = remaining.drop(
                columns=[column]
            )

            removed.append(
                {
                    "feature": column,
                    "reason": "target_column",
                }
            )

            continue

        # -----------------------------------------------------------
        # Constant features
        # -----------------------------------------------------------

        if remaining[column].nunique(
            dropna=False
        ) <= 1:

            remaining = remaining.drop(
                columns=[column]
            )

            removed.append(
                {
                    "feature": column,
                    "reason": "constant_feature",
                }
            )

            continue

        # -----------------------------------------------------------
        # Identifier-like features
        # -----------------------------------------------------------

        if _looks_like_id(column):

            remaining = remaining.drop(
                columns=[column]
            )

            removed.append(
                {
                    "feature": column,
                    "reason": "identifier_column",
                }
            )

    return remaining, removed


# -------------------------------------------------------------------
# Mutual information
# -------------------------------------------------------------------

def _calculate_mutual_information(
    X,
    y,
):
    """
    Calculate mutual information between each feature and target.

    Mutual information measures statistical dependence between a
    feature and the target.

    Higher values indicate stronger measured dependence.
    """

    if X.empty:
        return {}

    (
        prepared,
        categorical_mask,
        encoding_information,
    ) = _prepare_features(X)

    # ---------------------------------------------------------------
    # Ensure all values are finite.
    #
    # The cleaning layer should normally already guarantee this,
    # but this check prevents obscure sklearn failures.
    # ---------------------------------------------------------------

    prepared = prepared.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    if prepared.isna().any().any():

        raise ValueError(
            "Feature selection received missing or non-finite "
            "feature values. Run data_cleaner.py first."
        )

    scores = mutual_info_classif(
        prepared,
        y,
        discrete_features=categorical_mask,
        random_state=42,
    )

    result = {}

    for index, column in enumerate(
        prepared.columns
    ):

        result[column] = float(
            scores[index]
        )

    return result


# -------------------------------------------------------------------
# Correlation analysis
# -------------------------------------------------------------------

def _find_redundant_features(
    X,
    scores,
    threshold,
):
    """
    Find highly correlated numeric features.

    When two numeric features are highly correlated, the feature
    with the weaker target relevance is removed.

    This is a redundancy-reduction step, not a causal statement.
    """

    numeric_columns = list(
        X.select_dtypes(
            include=[np.number]
        ).columns
    )

    if len(numeric_columns) < 2:
        return [], []

    correlation_matrix = (
        X[numeric_columns]
        .corr()
        .abs()
    )

    redundant = []

    comparisons = []

    for i in range(
        len(numeric_columns)
    ):

        for j in range(
            i + 1,
            len(numeric_columns)
        ):

            feature_a = numeric_columns[i]
            feature_b = numeric_columns[j]

            correlation = correlation_matrix.loc[
                feature_a,
                feature_b,
            ]

            if pd.isna(correlation):
                continue

            if correlation >= threshold:

                score_a = scores.get(
                    feature_a,
                    0.0,
                )

                score_b = scores.get(
                    feature_b,
                    0.0,
                )

                # Keep the feature with the stronger
                # measured relationship with the target.
                if score_a >= score_b:

                    removed = feature_b
                    kept = feature_a

                else:

                    removed = feature_a
                    kept = feature_b

                if removed not in redundant:

                    redundant.append(
                        removed
                    )

                comparisons.append(
                    {
                        "feature_a": feature_a,
                        "feature_b": feature_b,
                        "correlation": float(
                            correlation
                        ),
                        "kept": kept,
                        "removed": removed,
                    }
                )

    return (
        redundant,
        comparisons,
    )


# -------------------------------------------------------------------
# Ranking
# -------------------------------------------------------------------

def _rank_features(
    features,
    scores,
):
    """
    Rank features by mutual-information score.
    """

    ranked = sorted(
        features,
        key=lambda feature: scores.get(
            feature,
            0.0,
        ),
        reverse=True,
    )

    return [
        {
            "feature": feature,
            "mutual_information": round(
                float(
                    scores.get(
                        feature,
                        0.0,
                    )
                ),
                6,
            ),
            "rank": index + 1,
        }
        for index, feature in enumerate(
            ranked
        )
    ]


# -------------------------------------------------------------------
# Main feature-selection function
# -------------------------------------------------------------------

def select_features(
    X,
    y,
    target_column=None,
    max_features=DEFAULT_MAX_FEATURES,
    correlation_threshold=DEFAULT_CORRELATION_THRESHOLD,
):
    """
    Automatically select relevant features.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature dataset.

    y : array-like
        Target values.

    target_column : str, optional
        Target column name if X still contains it.

    max_features : int
        Maximum number of features returned for the classical
        feature set.

    correlation_threshold : float
        Absolute correlation above which one of two numeric features
        is considered redundant.

    Returns
    -------
    selected_X : pandas.DataFrame
        Dataset containing selected features.

    report : dict
        Detailed feature-selection report.

    Notes
    -----
    The input DataFrame is not modified.
    """

    if not isinstance(
        X,
        pd.DataFrame,
    ):
        X = pd.DataFrame(X)

    if X.empty:
        raise ValueError(
            "Feature selection received an empty dataset."
        )

    if max_features <= 0:
        raise ValueError(
            "max_features must be greater than zero."
        )

    if not (
        0.0
        < correlation_threshold
        <= 1.0
    ):
        raise ValueError(
            "correlation_threshold must be between "
            "0 and 1."
        )

    # ---------------------------------------------------------------
    # Prepare target
    # ---------------------------------------------------------------

    y = pd.Series(
        y
    ).reset_index(drop=True)

    if len(X) != len(y):
        raise ValueError(
            "Feature and target lengths do not match."
        )

    if y.isna().any():
        raise ValueError(
            "Feature selection target contains missing values."
        )

    y_encoded, target_encoder = _encode_target(
        y
    )

    # ---------------------------------------------------------------
    # Remove target from X if present
    # ---------------------------------------------------------------

    working_X = X.copy()

    if (
        target_column is not None
        and target_column in working_X.columns
    ):

        working_X = working_X.drop(
            columns=[target_column]
        )

    # ---------------------------------------------------------------
    # Remove unusable features
    # ---------------------------------------------------------------

    working_X, initially_removed = (
        _remove_unusable_features(
            working_X,
            target_column,
        )
    )

    if working_X.empty:

        raise ValueError(
            "No usable features remain after removing "
            "target, constant, and identifier columns."
        )

    # ---------------------------------------------------------------
    # Mutual information
    # ---------------------------------------------------------------

    scores = _calculate_mutual_information(
        working_X,
        y_encoded,
    )

    # ---------------------------------------------------------------
    # Remove redundant numeric features
    # ---------------------------------------------------------------

    (
        redundant_features,
        correlation_comparisons,
    ) = _find_redundant_features(
        working_X,
        scores,
        correlation_threshold,
    )

    reduced_X = working_X.drop(
        columns=redundant_features,
        errors="ignore",
    )

    if reduced_X.empty:

        raise ValueError(
            "No features remain after redundancy reduction."
        )

    # ---------------------------------------------------------------
    # Rank remaining features
    # ---------------------------------------------------------------

    ranked = _rank_features(
        reduced_X.columns,
        scores,
    )

    # ---------------------------------------------------------------
    # Select top K
    # ---------------------------------------------------------------

    selected_features = [
        item["feature"]
        for item in ranked[
            :max_features
        ]
    ]

    selected_X = reduced_X[
        selected_features
    ].copy()

    # ---------------------------------------------------------------
    # Build report
    # ---------------------------------------------------------------

    report = {
        "original_feature_count": int(
            len(X.columns)
        ),

        "usable_feature_count": int(
            len(working_X.columns)
        ),

        "redundant_feature_count": int(
            len(redundant_features)
        ),

        "selected_feature_count": int(
            len(selected_features)
        ),

        "max_features": int(
            max_features
        ),

        "correlation_threshold": float(
            correlation_threshold
        ),

        "target": {
            "classes": [
                str(value)
                for value in target_encoder.classes_
            ],
            "class_count": int(
                len(target_encoder.classes_)
            ),
        },

        "initially_removed": initially_removed,

        "redundant_features": [
            {
                "feature": feature,
                "reason": "high_correlation",
            }
            for feature in redundant_features
        ],

        "correlation_analysis": (
            correlation_comparisons
        ),

        "ranking": ranked,

        "selected_features": selected_features,

        "status": "complete",
    }

    return (
        selected_X,
        report,
    )


# -------------------------------------------------------------------
# Quantum feature selection
# -------------------------------------------------------------------

def select_quantum_features(
    X,
    y,
    quantum_feature_count=DEFAULT_QUANTUM_FEATURES,
    target_column=None,
    correlation_threshold=DEFAULT_CORRELATION_THRESHOLD,
):
    """
    Select a small feature subset suitable for the quantum model.

    This function reuses the same feature-selection logic but applies
    a smaller feature budget.

    The quantum model currently uses one feature per qubit, so the
    returned feature count determines the required number of qubits.
    """

    if quantum_feature_count <= 0:
        raise ValueError(
            "quantum_feature_count must be greater than zero."
        )

    (
        selected_X,
        report,
    ) = select_features(
        X,
        y,
        target_column=target_column,
        max_features=quantum_feature_count,
        correlation_threshold=correlation_threshold,
    )

    quantum_features = list(
        selected_X.columns
    )

    quantum_report = {
        "quantum_feature_count": len(
            quantum_features
        ),

        "quantum_features": quantum_features,

        "n_qubits": len(
            quantum_features
        ),

        "selection_method": (
            "mutual_information_with_correlation_reduction"
        ),

        "feature_selection_report": report,
    }

    return (
        selected_X,
        quantum_report,
    )


# -------------------------------------------------------------------
# Standalone test
# -------------------------------------------------------------------

if __name__ == "__main__":

    dataset_path = (
        "data/heart_cleaned.csv"
    )

    target_column = "target"

    print("=" * 60)
    print("VITALIS FEATURE SELECTOR")
    print("=" * 60)

    print(
        f"\nLoading: {dataset_path}"
    )

    df = pd.read_csv(
        dataset_path
    )

    X = df.drop(
        columns=[target_column]
    )

    y = df[
        target_column
    ]

    # ---------------------------------------------------------------
    # Classical feature selection
    # ---------------------------------------------------------------

    (
        selected_X,
        report,
    ) = select_features(
        X,
        y,
        target_column=None,
        max_features=15,
    )

    print(
        "\nOriginal features:",
        report[
            "original_feature_count"
        ],
    )

    print(
        "Usable features:",
        report[
            "usable_feature_count"
        ],
    )

    print(
        "Redundant features:",
        report[
            "redundant_feature_count"
        ],
    )

    print(
        "Selected features:",
        report[
            "selected_feature_count"
        ],
    )

    print(
        "\nSelected features:"
    )

    for item in report[
        "ranking"
    ][:report["selected_feature_count"]]:

        print(
            f"  {item['rank']}. "
            f"{item['feature']} "
            f"(MI={item['mutual_information']:.6f})"
        )

    # ---------------------------------------------------------------
    # Quantum feature selection
    # ---------------------------------------------------------------

    (
        quantum_X,
        quantum_report,
    ) = select_quantum_features(
        X,
        y,
        quantum_feature_count=6,
    )

    print(
        "\nQuantum features:"
    )

    for feature in quantum_report[
        "quantum_features"
    ]:

        print(
            f"  - {feature}"
        )

    print(
        "\nQuantum qubits:",
        quantum_report[
            "n_qubits"
        ],
    )