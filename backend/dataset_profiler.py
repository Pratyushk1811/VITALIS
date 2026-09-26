"""
VITALIS Dataset Profiler

Inspects an uploaded biomedical dataset and produces a structured
description of its contents.

This module does NOT modify the dataset.

It identifies:
    - dataset dimensions
    - numeric features
    - categorical features
    - missing values
    - non-finite values
    - duplicate rows
    - constant columns
    - possible ID columns
    - possible target columns
    - target/class distribution

The output is intended to be consumed by the preprocessing,
feature-selection, and training layers.
"""

import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# Target column candidates
# -------------------------------------------------------------------

TARGET_CANDIDATES = [
    "target",
    "label",
    "diagnosis",
    "class",
    "outcome",
    "condition",
    "result",
    "y",
]


# -------------------------------------------------------------------
# ID column candidates
# -------------------------------------------------------------------

ID_NAME_HINTS = [
    "id",
    "patient_id",
    "patientid",
    "record_id",
    "recordid",
    "index",
]


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def _normalize_column_name(column):
    """
    Convert a column name into a normalized form for comparison.
    """

    return (
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def _find_target_columns(df):
    """
    Identify columns that look like target/label columns.

    This is heuristic only. The profiler does not automatically
    assume that a candidate is definitely the target.
    """

    targets = []

    normalized_columns = {
        column: _normalize_column_name(column)
        for column in df.columns
    }

    for column, normalized in normalized_columns.items():

        if normalized in TARGET_CANDIDATES:
            targets.append(column)

    return targets


def _find_id_columns(df):
    """
    Identify columns that may represent IDs.

    This uses both column-name hints and uniqueness.
    """

    id_columns = []

    for column in df.columns:

        normalized = _normalize_column_name(column)

        # -----------------------------------------------------------
        # Name-based detection
        # -----------------------------------------------------------

        name_hint = (
            normalized in ID_NAME_HINTS
            or normalized.endswith("_id")
            or normalized.endswith("id")
        )

        # -----------------------------------------------------------
        # Uniqueness-based detection
        # -----------------------------------------------------------

        unique_ratio = (
            df[column].nunique(dropna=False)
            / len(df)
            if len(df) > 0
            else 0
        )

        unique_like_id = (
            unique_ratio >= 0.98
            and len(df) >= 20
        )

        if name_hint or unique_like_id:
            id_columns.append(column)

    return id_columns


def _find_constant_columns(df):
    """
    Find columns containing only one unique value.

    Missing values are included in the uniqueness calculation.
    """

    constant_columns = []

    for column in df.columns:

        if df[column].nunique(dropna=False) <= 1:
            constant_columns.append(column)

    return constant_columns


def _get_column_types(df):
    """
    Classify columns as numeric, categorical, boolean, datetime,
    or other.
    """

    numeric = []
    categorical = []
    boolean = []
    datetime_columns = []
    other = []

    for column in df.columns:

        series = df[column]

        if pd.api.types.is_bool_dtype(series):
            boolean.append(column)

        elif pd.api.types.is_numeric_dtype(series):
            numeric.append(column)

        elif pd.api.types.is_datetime64_any_dtype(series):
            datetime_columns.append(column)

        elif (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_categorical_dtype(series)
        ):
            categorical.append(column)

        else:
            other.append(column)

    return {
        "numeric": numeric,
        "categorical": categorical,
        "boolean": boolean,
        "datetime": datetime_columns,
        "other": other,
    }


def _missing_value_summary(df):
    """
    Return missing-value counts and percentages for each column.
    """

    summary = {}

    for column in df.columns:

        missing_count = int(
            df[column].isna().sum()
        )

        missing_percentage = (
            (missing_count / len(df)) * 100
            if len(df) > 0
            else 0.0
        )

        summary[column] = {
            "count": missing_count,
            "percentage": round(
                float(missing_percentage),
                4,
            ),
        }

    return summary


def _non_finite_summary(df):
    """
    Detect positive/negative infinity in numeric columns.
    """

    summary = {}

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    for column in numeric_columns:

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        positive_infinity = int(
            np.isposinf(values).sum()
        )

        negative_infinity = int(
            np.isneginf(values).sum()
        )

        total = (
            positive_infinity
            + negative_infinity
        )

        if total > 0:
            summary[column] = {
                "positive_infinity": positive_infinity,
                "negative_infinity": negative_infinity,
                "total": total,
            }

    return summary


def _numeric_statistics(df, numeric_columns):
    """
    Calculate basic statistics for numeric features.
    """

    statistics = {}

    for column in numeric_columns:

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        statistics[column] = {
            "min": (
                float(values.min())
                if not values.dropna().empty
                else None
            ),
            "max": (
                float(values.max())
                if not values.dropna().empty
                else None
            ),
            "mean": (
                float(values.mean())
                if not values.dropna().empty
                else None
            ),
            "median": (
                float(values.median())
                if not values.dropna().empty
                else None
            ),
            "unique_values": int(
                values.nunique(dropna=True)
            ),
        }

    return statistics


def _categorical_statistics(
    df,
    categorical_columns,
):
    """
    Calculate basic information about categorical features.
    """

    statistics = {}

    for column in categorical_columns:

        series = df[column]

        value_counts = (
            series
            .value_counts(
                dropna=False
            )
            .head(20)
            .to_dict()
        )

        statistics[column] = {
            "unique_values": int(
                series.nunique(
                    dropna=True
                )
            ),
            "top_values": {
                str(key): int(value)
                for key, value in value_counts.items()
            },
        }

    return statistics


def _target_information(
    df,
    target_candidates,
):
    """
    Generate information about possible target columns.

    Multiple candidates may exist. No target is selected
    automatically here.
    """

    information = {}

    for column in target_candidates:

        series = df[column]

        counts = (
            series
            .value_counts(
                dropna=False
            )
            .to_dict()
        )

        information[column] = {
            "dtype": str(series.dtype),
            "unique_classes": int(
                series.nunique(
                    dropna=True
                )
            ),
            "class_distribution": {
                str(key): int(value)
                for key, value in counts.items()
            },
            "missing_values": int(
                series.isna().sum()
            ),
        }

    return information


# -------------------------------------------------------------------
# Main profiler
# -------------------------------------------------------------------

def profile_dataset(df):
    """
    Profile a pandas DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to inspect.

    Returns
    -------
    dict
        Structured dataset profile.

    Notes
    -----
    The input DataFrame is NOT modified.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "profile_dataset() expects a pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "Cannot profile an empty dataset."
        )

    # ---------------------------------------------------------------
    # Basic information
    # ---------------------------------------------------------------

    row_count = len(df)
    column_count = len(df.columns)

    # ---------------------------------------------------------------
    # Column type analysis
    # ---------------------------------------------------------------

    column_types = _get_column_types(df)

    numeric_columns = column_types["numeric"]
    categorical_columns = column_types["categorical"]

    # ---------------------------------------------------------------
    # Dataset-level issues
    # ---------------------------------------------------------------

    duplicate_rows = int(
        df.duplicated().sum()
    )

    target_candidates = _find_target_columns(df)

    id_columns = _find_id_columns(df)

    constant_columns = _find_constant_columns(df)

    # ---------------------------------------------------------------
    # Missing and non-finite values
    # ---------------------------------------------------------------

    missing_values = _missing_value_summary(df)

    total_missing_values = int(
        df.isna().sum().sum()
    )

    non_finite_values = _non_finite_summary(df)

    total_non_finite_values = sum(
        item["total"]
        for item in non_finite_values.values()
    )

    # ---------------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------------

    numeric_statistics = _numeric_statistics(
        df,
        numeric_columns,
    )

    categorical_statistics = _categorical_statistics(
        df,
        categorical_columns,
    )

    # ---------------------------------------------------------------
    # Target information
    # ---------------------------------------------------------------

    target_information = _target_information(
        df,
        target_candidates,
    )

    # ---------------------------------------------------------------
    # Profile result
    # ---------------------------------------------------------------

    profile = {
        "shape": {
            "rows": row_count,
            "columns": column_count,
        },

        "columns": [
            str(column)
            for column in df.columns
        ],

        "column_types": {
            "numeric": numeric_columns,
            "categorical": categorical_columns,
            "boolean": column_types["boolean"],
            "datetime": column_types["datetime"],
            "other": column_types["other"],
        },

        "target_candidates": target_candidates,

        "id_candidates": id_columns,

        "constant_columns": constant_columns,

        "duplicates": {
            "count": duplicate_rows,
            "percentage": round(
                (
                    duplicate_rows
                    / row_count
                    * 100
                )
                if row_count > 0
                else 0.0,
                4,
            ),
        },

        "missing_values": {
            "total": total_missing_values,
            "columns": missing_values,
        },

        "non_finite_values": {
            "total": total_non_finite_values,
            "columns": non_finite_values,
        },

        "numeric_statistics": numeric_statistics,

        "categorical_statistics": categorical_statistics,

        "target_information": target_information,
    }

    return profile


# -------------------------------------------------------------------
# Compact summary
# -------------------------------------------------------------------

def get_profile_summary(profile):
    """
    Convert a detailed profile into a compact summary suitable
    for an API response or frontend.
    """

    shape = profile["shape"]

    return {
        "rows": shape["rows"],
        "columns": shape["columns"],
        "numeric_features": len(
            profile["column_types"]["numeric"]
        ),
        "categorical_features": len(
            profile["column_types"]["categorical"]
        ),
        "missing_values": profile[
            "missing_values"
        ]["total"],
        "non_finite_values": profile[
            "non_finite_values"
        ]["total"],
        "duplicate_rows": profile[
            "duplicates"
        ]["count"],
        "constant_columns": profile[
            "constant_columns"
        ],
        "id_candidates": profile[
            "id_candidates"
        ],
        "target_candidates": profile[
            "target_candidates"
        ],
    }


# -------------------------------------------------------------------
# Standalone test
# -------------------------------------------------------------------

if __name__ == "__main__":

    # Use the existing heart dataset to verify the profiler.
    dataset_path = "data/heart_cleaned.csv"

    print("=" * 60)
    print("VITALIS DATASET PROFILER")
    print("=" * 60)

    print(
        f"\nLoading: {dataset_path}"
    )

    df = pd.read_csv(
        dataset_path
    )

    profile = profile_dataset(
        df
    )

    summary = get_profile_summary(
        profile
    )

    print("\nDataset summary:")
    print(
        f"Rows:                 {summary['rows']}"
    )
    print(
        f"Columns:              {summary['columns']}"
    )
    print(
        f"Numeric features:     {summary['numeric_features']}"
    )
    print(
        f"Categorical features: {summary['categorical_features']}"
    )
    print(
        f"Missing values:       {summary['missing_values']}"
    )
    print(
        f"Non-finite values:    {summary['non_finite_values']}"
    )
    print(
        f"Duplicate rows:       {summary['duplicate_rows']}"
    )

    print(
        "\nTarget candidates:",
        summary["target_candidates"],
    )

    print(
        "ID candidates:",
        summary["id_candidates"],
    )

    print(
        "Constant columns:",
        summary["constant_columns"],
    )

    print("\nTarget information:")

    for column, information in profile[
        "target_information"
    ].items():

        print(
            f"\n  {column}:"
        )

        print(
            f"    Classes: "
            f"{information['unique_classes']}"
        )

        print(
            f"    Distribution: "
            f"{information['class_distribution']}"
        )