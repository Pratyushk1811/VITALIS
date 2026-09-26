"""
VITALIS Data Cleaner

Generic preprocessing layer for uploaded biomedical datasets.

This module cleans a pandas DataFrame without making disease-specific
assumptions.

Responsibilities:
    - validate the input dataset
    - identify the target column
    - remove duplicate rows
    - replace infinite numeric values
    - detect completely empty columns
    - handle missing numeric values
    - handle missing categorical values
    - preserve the target column
    - produce a transparent preprocessing report

Feature selection and quantum dimensionality reduction are handled
by separate modules.
"""

import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _is_numeric_like(series, threshold=0.90):
    """
    Determine whether an object/string column is mostly numeric.

    A column is considered numeric-like when at least `threshold`
    fraction of its non-null values can be converted to numbers.
    """

    if not pd.api.types.is_object_dtype(series):
        return False

    non_null = series.dropna()

    if len(non_null) == 0:
        return False

    converted = pd.to_numeric(
        non_null,
        errors="coerce",
    )

    valid_ratio = (
        converted.notna().sum()
        / len(non_null)
    )

    return valid_ratio >= threshold


def _convert_numeric_like_columns(df):
    """
    Convert object/string columns that are predominantly numeric
    into numeric columns.

    Returns:
        cleaned dataframe
        list of converted columns
    """

    converted_columns = []

    for column in df.columns:

        if _is_numeric_like(df[column]):

            converted = pd.to_numeric(
                df[column],
                errors="coerce",
            )

            df[column] = converted

            converted_columns.append(
                column
            )

    return df, converted_columns


def _replace_non_finite_values(df):
    """
    Replace positive and negative infinity with NaN.

    Missing-value imputation happens later.
    """

    replaced = {}

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    for column in numeric_columns:

        values = df[column]

        positive_inf = int(
            np.isposinf(values).sum()
        )

        negative_inf = int(
            np.isneginf(values).sum()
        )

        total = (
            positive_inf
            + negative_inf
        )

        if total > 0:

            df[column] = (
                df[column]
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
            )

            replaced[column] = {
                "positive_infinity": positive_inf,
                "negative_infinity": negative_inf,
                "total": total,
            }

    return df, replaced


def _remove_empty_columns(df, target_column):
    """
    Remove columns that contain no usable values.

    The target column is never silently removed.
    """

    removed = []

    for column in list(df.columns):

        if column == target_column:
            continue

        if df[column].notna().sum() == 0:

            df = df.drop(
                columns=[column]
            )

            removed.append(
                column
            )

    return df, removed


def _impute_numeric_columns(df, target_column):
    """
    Fill missing numeric feature values with the median.

    The target column is excluded from automatic feature
    imputation because target handling requires explicit validation.
    """

    imputed = {}

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    for column in numeric_columns:

        if column == target_column:
            continue

        missing_before = int(
            df[column].isna().sum()
        )

        if missing_before == 0:
            continue

        median = df[column].median()

        # If a numeric column has no valid values, it should have
        # already been removed by _remove_empty_columns(). This
        # check protects against unexpected edge cases.
        if pd.isna(median):
            continue

        df[column] = df[column].fillna(
            median
        )

        imputed[column] = {
            "method": "median",
            "value": float(median),
            "values_imputed": missing_before,
        }

    return df, imputed


def _impute_categorical_columns(
    df,
    target_column,
):
    """
    Fill missing categorical feature values using the most frequent
    category (mode).

    The target column is not automatically imputed.
    """

    imputed = {}

    categorical_columns = df.select_dtypes(
        include=[
            "object",
            "category",
            "string",
        ]
    ).columns

    for column in categorical_columns:

        if column == target_column:
            continue

        missing_before = int(
            df[column].isna().sum()
        )

        if missing_before == 0:
            continue

        mode = df[column].mode(
            dropna=True
        )

        if mode.empty:
            continue

        mode_value = mode.iloc[0]

        df[column] = df[column].fillna(
            mode_value
        )

        imputed[column] = {
            "method": "mode",
            "value": str(mode_value),
            "values_imputed": missing_before,
        }

    return df, imputed


def _validate_target(df, target_column):
    """
    Validate that the requested target exists and contains usable
    values.

    The cleaner does not infer or change target labels.
    """

    if target_column not in df.columns:

        raise ValueError(
            f"Target column '{target_column}' "
            "was not found in the dataset."
        )

    missing_target = int(
        df[target_column].isna().sum()
    )

    if missing_target > 0:

        raise ValueError(
            f"Target column '{target_column}' "
            f"contains {missing_target} missing values. "
            "Target values must be provided explicitly."
        )

    unique_targets = df[
        target_column
    ].nunique()

    if unique_targets < 2:

        raise ValueError(
            f"Target column '{target_column}' "
            "must contain at least two classes."
        )

    return {
        "column": target_column,
        "unique_classes": int(
            unique_targets
        ),
        "class_distribution": {
            str(key): int(value)
            for key, value in (
                df[target_column]
                .value_counts()
                .to_dict()
                .items()
            )
        },
    }


# -------------------------------------------------------------------
# Main cleaning function
# -------------------------------------------------------------------

def clean_dataset(
    df,
    target_column,
    remove_duplicates=True,
):
    """
    Clean a dataset using generic preprocessing rules.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.

    target_column : str
        Name of the target/label column.

    remove_duplicates : bool
        Whether duplicate rows should be removed.

    Returns
    -------
    cleaned_df : pandas.DataFrame
        Cleaned dataset.

    report : dict
        Transparent description of all transformations.

    Notes
    -----
    The original DataFrame is not modified.
    """

    if not isinstance(
        df,
        pd.DataFrame,
    ):
        raise TypeError(
            "clean_dataset() expects a pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "Cannot clean an empty dataset."
        )

    # ---------------------------------------------------------------
    # Work on a copy
    # ---------------------------------------------------------------

    cleaned = df.copy()

    original_shape = {
        "rows": len(cleaned),
        "columns": len(cleaned.columns),
    }

    # ---------------------------------------------------------------
    # Validate target before transformations
    # ---------------------------------------------------------------

    target_info_before = _validate_target(
        cleaned,
        target_column,
    )

    # ---------------------------------------------------------------
    # Normalize column names only for comparison?
    #
    # We intentionally do NOT rename columns here.
    # The original dataset schema should remain visible to the user.
    # ---------------------------------------------------------------

    # ---------------------------------------------------------------
    # Convert numeric-looking columns
    # ---------------------------------------------------------------

    (
        cleaned,
        converted_columns,
    ) = _convert_numeric_like_columns(
        cleaned
    )

    # ---------------------------------------------------------------
    # Replace infinite values
    # ---------------------------------------------------------------

    (
        cleaned,
        non_finite_replacements,
    ) = _replace_non_finite_values(
        cleaned
    )

    # ---------------------------------------------------------------
    # Remove completely empty feature columns
    # ---------------------------------------------------------------

    (
        cleaned,
        removed_empty_columns,
    ) = _remove_empty_columns(
        cleaned,
        target_column,
    )

    # ---------------------------------------------------------------
    # Handle duplicate rows
    # ---------------------------------------------------------------

    duplicate_rows_removed = 0

    if remove_duplicates:

        duplicate_rows_removed = int(
            cleaned.duplicated().sum()
        )

        if duplicate_rows_removed > 0:

            cleaned = cleaned.drop_duplicates(
                keep="first"
            ).reset_index(
                drop=True
            )

    # ---------------------------------------------------------------
    # Impute numeric features
    # ---------------------------------------------------------------

    (
        cleaned,
        numeric_imputation,
    ) = _impute_numeric_columns(
        cleaned,
        target_column,
    )

    # ---------------------------------------------------------------
    # Impute categorical features
    # ---------------------------------------------------------------

    (
        cleaned,
        categorical_imputation,
    ) = _impute_categorical_columns(
        cleaned,
        target_column,
    )

    # ---------------------------------------------------------------
    # Final target validation
    # ---------------------------------------------------------------

    target_info_after = _validate_target(
        cleaned,
        target_column,
    )

    # ---------------------------------------------------------------
    # Check whether missing feature values remain
    # ---------------------------------------------------------------

    remaining_missing = {}

    for column in cleaned.columns:

        missing = int(
            cleaned[column].isna().sum()
        )

        if missing > 0:

            remaining_missing[column] = missing

    # ---------------------------------------------------------------
    # Check remaining non-finite numeric values
    # ---------------------------------------------------------------

    remaining_non_finite = {}

    numeric_columns = cleaned.select_dtypes(
        include=[np.number]
    ).columns

    for column in numeric_columns:

        count = int(
            np.isinf(
                cleaned[column].to_numpy()
            ).sum()
        )

        if count > 0:
            remaining_non_finite[column] = count

    # ---------------------------------------------------------------
    # Final shape
    # ---------------------------------------------------------------

    final_shape = {
        "rows": len(cleaned),
        "columns": len(cleaned.columns),
    }

    # ---------------------------------------------------------------
    # Build report
    # ---------------------------------------------------------------

    report = {
        "original_shape": original_shape,

        "final_shape": final_shape,

        "rows_removed": (
            original_shape["rows"]
            - final_shape["rows"]
        ),

        "columns_removed": (
            original_shape["columns"]
            - final_shape["columns"]
        ),

        "operations": {
            "numeric_columns_converted": (
                converted_columns
            ),

            "non_finite_values_replaced": (
                non_finite_replacements
            ),

            "empty_columns_removed": (
                removed_empty_columns
            ),

            "duplicate_rows_removed": (
                duplicate_rows_removed
            ),

            "numeric_imputation": (
                numeric_imputation
            ),

            "categorical_imputation": (
                categorical_imputation
            ),
        },

        "target": {
            "before": target_info_before,
            "after": target_info_after,
        },

        "remaining_missing_values": (
            remaining_missing
        ),

        "remaining_non_finite_values": (
            remaining_non_finite
        ),

        "status": (
            "clean"
            if (
                not remaining_missing
                and not remaining_non_finite
            )
            else "requires_review"
        ),
    }

    return cleaned, report


# -------------------------------------------------------------------
# Compact report
# -------------------------------------------------------------------

def get_cleaning_summary(report):
    """
    Convert the detailed cleaning report into a compact summary
    suitable for an API response or frontend.
    """

    operations = report[
        "operations"
    ]

    numeric_imputation = operations[
        "numeric_imputation"
    ]

    categorical_imputation = operations[
        "categorical_imputation"
    ]

    return {
        "original_shape": report[
            "original_shape"
        ],

        "final_shape": report[
            "final_shape"
        ],

        "rows_removed": report[
            "rows_removed"
        ],

        "columns_removed": report[
            "columns_removed"
        ],

        "duplicates_removed": operations[
            "duplicate_rows_removed"
        ],

        "non_finite_values_replaced": sum(
            item["total"]
            for item in operations[
                "non_finite_values_replaced"
            ].values()
        ),

        "numeric_columns_imputed": len(
            numeric_imputation
        ),

        "categorical_columns_imputed": len(
            categorical_imputation
        ),

        "empty_columns_removed": operations[
            "empty_columns_removed"
        ],

        "status": report[
            "status"
        ],
    }


# -------------------------------------------------------------------
# Standalone test
# -------------------------------------------------------------------

if __name__ == "__main__":

    dataset_path = (
        "data/heart_cleaned.csv"
    )

    target_column = "target"

    print("=" * 60)
    print("VITALIS DATA CLEANER")
    print("=" * 60)

    print(
        f"\nLoading: {dataset_path}"
    )

    df = pd.read_csv(
        dataset_path
    )

    cleaned_df, report = clean_dataset(
        df,
        target_column=target_column,
    )

    summary = get_cleaning_summary(
        report
    )

    print("\nCleaning summary:")

    print(
        f"Original shape:      "
        f"{summary['original_shape']}"
    )

    print(
        f"Final shape:         "
        f"{summary['final_shape']}"
    )

    print(
        f"Rows removed:        "
        f"{summary['rows_removed']}"
    )

    print(
        f"Columns removed:     "
        f"{summary['columns_removed']}"
    )

    print(
        f"Duplicates removed:  "
        f"{summary['duplicates_removed']}"
    )

    print(
        f"Non-finite replaced:  "
        f"{summary['non_finite_values_replaced']}"
    )

    print(
        f"Numeric imputed:      "
        f"{summary['numeric_columns_imputed']}"
    )

    print(
        f"Categorical imputed:  "
        f"{summary['categorical_columns_imputed']}"
    )

    print(
        f"Empty columns removed:"
        f" {summary['empty_columns_removed']}"
    )

    print(
        f"Status:               "
        f"{summary['status']}"
    )

    print("\nTarget after cleaning:")

    print(
        report["target"]["after"]
    )