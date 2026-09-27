"""
VITALIS Feature Selection
--------------------------

Select useful biomedical features from the training data.

Design goals:
    - Never use obvious identifier columns as ML features.
    - Support numeric and categorical features.
    - Fit selection using TRAINING data only.
    - Keep the selected feature count bounded.
    - Preserve original feature names for downstream preprocessing.
"""

import re

import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder


# =====================================================================
# CONFIGURATION
# =====================================================================

DEFAULT_MAX_FEATURES = 15
DEFAULT_CORRELATION_THRESHOLD = 0.90


# =====================================================================
# IDENTIFIER DETECTION
# =====================================================================

# Exact names that should never be used as predictive features.
EXACT_IDENTIFIER_NAMES = {
    "id",
    "ID",
    "identifier",
    "identifier_id",
    "patient_id",
    "patient_nbr",
    "patient_number",
    "patient_number_id",
    "encounter_id",
    "encounter_number",
    "record_id",
    "record_number",
    "row_id",
    "subject_id",
    "subject_nbr",
    "case_id",
    "case_number",
    "sample_id",
    "sample_number",
}


def _is_identifier_column(column_name, series):
    """
    Determine whether a column is very likely an identifier.

    We deliberately use conservative rules so that legitimate
    biomedical variables such as age, number_diagnoses, etc.
    are not accidentally removed.
    """

    name = str(column_name).strip()

    normalized = name.lower()

    # -------------------------------------------------------------
    # Exact identifier names
    # -------------------------------------------------------------

    if name in EXACT_IDENTIFIER_NAMES:
        return True

    if normalized in {
        value.lower()
        for value in EXACT_IDENTIFIER_NAMES
    }:
        return True

    # -------------------------------------------------------------
    # Common identifier naming patterns
    # -------------------------------------------------------------

    identifier_patterns = [
        r"^id$",
        r".*_id$",
        r"^id_.*",
        r".*_identifier$",
        r"^identifier_.*",
        r".*_number$",
        r"^number_.*_id$",
    ]

    for pattern in identifier_patterns:

        if re.match(
            pattern,
            normalized,
        ):
            return True

    # -------------------------------------------------------------
    # High-cardinality integer identifier heuristic
    # -------------------------------------------------------------
    #
    # Only use this as a fallback.
    #
    # A numeric column with nearly one unique value per row is
    # usually an identifier rather than a biomedical measurement.
    #
    # We deliberately require BOTH:
    #   - numeric dtype
    #   - very high uniqueness
    #
    # This avoids removing legitimate variables such as age,
    # blood pressure, glucose, etc.
    # -------------------------------------------------------------

    if pd.api.types.is_numeric_dtype(series):

        non_null = series.dropna()

        if len(non_null) > 20:

            uniqueness_ratio = (
                non_null.nunique()
                / len(non_null)
            )

            if uniqueness_ratio >= 0.98:
                return True

    return False


def _detect_identifier_columns(X):
    """
    Return columns that are likely identifiers.
    """

    identifier_columns = []

    for column in X.columns:

        if _is_identifier_column(
            column,
            X[column],
        ):
            identifier_columns.append(
                column
            )

    return identifier_columns


# =====================================================================
# DATA PREPARATION
# =====================================================================

def _prepare_feature_for_selection(series):
    """
    Convert a feature into a representation suitable for
    mutual-information feature selection.

    Numeric:
        converted to numeric and median-filled.

    Categorical:
        converted to strings and label encoded.
    """

    series = series.copy()

    # -------------------------------------------------------------
    # Numeric feature
    # -------------------------------------------------------------

    if pd.api.types.is_numeric_dtype(series):

        values = pd.to_numeric(
            series,
            errors="coerce",
        )

        median = values.median()

        if pd.isna(median):
            median = 0.0

        values = values.fillna(
            median
        )

        return (
            values.to_numpy(
                dtype=float
            ),
            False,
        )

    # -------------------------------------------------------------
    # Categorical feature
    # -------------------------------------------------------------

    values = (
        series
        .astype("string")
        .fillna("__MISSING__")
        .astype(str)
    )

    encoder = LabelEncoder()

    encoded = encoder.fit_transform(
        values
    )

    return (
        encoded.astype(float),
        True,
    )


# =====================================================================
# TARGET PREPARATION
# =====================================================================

def _prepare_target(y):
    """
    Convert target values to a numeric representation for
    mutual-information calculation.
    """

    y = pd.Series(y).copy()

    if y.isna().any():

        y = y.fillna(
            "__MISSING_TARGET__"
        )

    if pd.api.types.is_numeric_dtype(y):

        return pd.to_numeric(
            y,
            errors="coerce",
        ).fillna(
            0
        ).to_numpy()

    encoder = LabelEncoder()

    return encoder.fit_transform(
        y.astype(str)
    )


# =====================================================================
# CORRELATION REDUCTION
# =====================================================================

def _remove_highly_correlated_features(
    X,
    ranked_features,
    threshold=DEFAULT_CORRELATION_THRESHOLD,
):
    """
    Remove redundant numeric features.

    Features are considered in their mutual-information ranking
    order, so a stronger feature is retained when two features
    are highly correlated.
    """

    numeric_columns = [
        column
        for column in ranked_features
        if pd.api.types.is_numeric_dtype(
            X[column]
        )
    ]

    if len(numeric_columns) <= 1:

        return ranked_features

    numeric_data = X[
        numeric_columns
    ].copy()

    numeric_data = numeric_data.apply(
        pd.to_numeric,
        errors="coerce",
    )

    numeric_data = numeric_data.fillna(
        numeric_data.median()
    )

    correlation_matrix = (
        numeric_data.corr()
        .abs()
    )

    selected = []

    for feature in ranked_features:

        if feature not in numeric_columns:

            selected.append(
                feature
            )

            continue

        should_remove = False

        for already_selected in selected:

            if (
                already_selected
                not in numeric_columns
            ):
                continue

            correlation = correlation_matrix.loc[
                feature,
                already_selected,
            ]

            if (
                pd.notna(correlation)
                and correlation >= threshold
            ):
                should_remove = True
                break

        if not should_remove:

            selected.append(
                feature
            )

    return selected


# =====================================================================
# MAIN FEATURE SELECTION
# =====================================================================

def select_features(
    X,
    y,
    target_column=None,
    max_features=DEFAULT_MAX_FEATURES,
    correlation_threshold=DEFAULT_CORRELATION_THRESHOLD,
):
    """
    Select useful features from training data.

    Parameters
    ----------
    X : pandas.DataFrame
        Training feature matrix.

    y : pandas.Series
        Training target.

    target_column : str or None
        Optional target name. If supplied and present in X,
        it is removed.

    max_features : int
        Maximum number of features to return.

    correlation_threshold : float
        Threshold used to remove highly correlated numeric
        features.

    Returns
    -------
    selected_X : pandas.DataFrame
        Training data containing selected features.

    report : dict
        Feature-selection metadata.
    """

    # =================================================================
    # VALIDATION
    # =================================================================

    if not isinstance(
        X,
        pd.DataFrame,
    ):
        raise TypeError(
            "X must be a pandas DataFrame."
        )

    if X.empty:
        raise ValueError(
            "Feature matrix is empty."
        )

    if max_features < 1:
        raise ValueError(
            "max_features must be at least 1."
        )

    if not (
        0.0
        < correlation_threshold
        <= 1.0
    ):
        raise ValueError(
            "correlation_threshold must be between 0 and 1."
        )

    X = X.copy()

    # =================================================================
    # REMOVE TARGET IF PRESENT
    # =================================================================

    if (
        target_column is not None
        and target_column in X.columns
    ):

        X = X.drop(
            columns=[
                target_column
            ]
        )

    if X.empty:

        raise ValueError(
            "No features remain after removing the target."
        )

    # =================================================================
    # REMOVE IDENTIFIERS
    # =================================================================

    identifier_columns = (
        _detect_identifier_columns(X)
    )

    X_candidate = X.drop(
        columns=identifier_columns,
        errors="ignore",
    )

    if X_candidate.empty:

        raise ValueError(
            "No usable features remain after removing identifier columns."
        )

    # =================================================================
    # PREPARE TARGET
    # =================================================================

    y_encoded = _prepare_target(
        y
    )

    # =================================================================
    # PREPARE FEATURES FOR MUTUAL INFORMATION
    # =================================================================

    prepared_features = []
    discrete_features = []
    usable_columns = []

    for column in X_candidate.columns:

        values, is_categorical = (
            _prepare_feature_for_selection(
                X_candidate[column]
            )
        )

        # Skip completely unusable columns.
        if not np.isfinite(
            values
        ).all():

            continue

        prepared_features.append(
            values
        )

        discrete_features.append(
            is_categorical
        )

        usable_columns.append(
            column
        )

    if not prepared_features:

        raise ValueError(
            "No usable features remain for feature selection."
        )

    feature_matrix = np.column_stack(
        prepared_features
    )

    # =================================================================
    # MUTUAL INFORMATION
    # =================================================================

    try:

        scores = mutual_info_classif(
            feature_matrix,
            y_encoded,
            discrete_features=(
                discrete_features
            ),
            random_state=42,
        )

    except Exception as exc:

        raise ValueError(
            "Mutual-information feature selection failed: "
            f"{exc}"
        ) from exc

    score_table = pd.DataFrame(
        {
            "feature": usable_columns,
            "score": scores,
        }
    )

    score_table = (
        score_table
        .sort_values(
            "score",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    # =================================================================
    # REMOVE REDUNDANT FEATURES
    # =================================================================

    ranked_features = (
        score_table[
            "feature"
        ].tolist()
    )

    ranked_features = (
        _remove_highly_correlated_features(
            X_candidate,
            ranked_features,
            threshold=(
                correlation_threshold
            ),
        )
    )

    # =================================================================
    # LIMIT FEATURE COUNT
    # =================================================================

    selected_features = (
        ranked_features[
            :max_features
        ]
    )

    if not selected_features:

        raise ValueError(
            "Feature selection produced no usable features."
        )

    selected_X = X[
        selected_features
    ].copy()

    # =================================================================
    # REPORT
    # =================================================================

    score_lookup = dict(
        zip(
            score_table["feature"],
            score_table["score"],
        )
    )

    selected_scores = {
        feature: float(
            score_lookup.get(
                feature,
                0.0,
            )
        )
        for feature in selected_features
    }

    report = {

        "method": (
            "Mutual information "
            "with identifier removal "
            "and correlation filtering"
        ),

        "original_feature_count": int(
            X.shape[1]
        ),

        "identifier_columns_removed": (
            identifier_columns
        ),

        "candidate_feature_count": int(
            X_candidate.shape[1]
        ),

        "usable_feature_count": int(
            len(usable_columns)
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

        "selected_features": (
            selected_features
        ),

        "selected_feature_scores": (
            selected_scores
        ),

        "ranking": [
            {
                "feature": row["feature"],
                "score": float(
                    row["score"]
                ),
            }
            for _, row
            in score_table.iterrows()
        ],
    }

    return (
        selected_X,
        report,
    )


# =====================================================================
# STANDALONE TEST
# =====================================================================

def standalone_test():

    dataset_path = (
        "data/heart_cleaned.csv"
    )

    print(
        f"Loading: {dataset_path}"
    )

    df = pd.read_csv(
        dataset_path
    )

    X = df.drop(
        columns=["target"]
    )

    y = df["target"]

    selected_X, report = (
        select_features(
            X,
            y,
            target_column=None,
            max_features=15,
        )
    )

    print()
    print(
        "=" * 60
    )
    print(
        "FEATURE SELECTION TEST"
    )
    print(
        "=" * 60
    )

    print(
        f"Original features: "
        f"{report['original_feature_count']}"
    )

    print(
        f"Identifier columns removed: "
        f"{report['identifier_columns_removed']}"
    )

    print(
        f"Candidate features: "
        f"{report['candidate_feature_count']}"
    )

    print(
        f"Selected features: "
        f"{report['selected_feature_count']}"
    )

    print()
    print(
        "Selected:"
    )

    for feature in report[
        "selected_features"
    ]:

        score = report[
            "selected_feature_scores"
        ][feature]

        print(
            f"  {feature}: "
            f"{score:.6f}"
        )


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":

    standalone_test()