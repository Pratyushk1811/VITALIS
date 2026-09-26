"""
Quantum Feature Reducer
-----------------------

Reduces an arbitrary selected feature set to a smaller numerical
representation suitable for quantum machine-learning models.

Pipeline:

    Selected features
          ↓
    Numeric/Categorical preprocessing
          ↓
    Standardization / One-hot encoding
          ↓
    PCA
          ↓
    Quantum feature representation
          ↓
    n quantum dimensions / qubits

Important:
    The reducer must be FIT ONLY on training data.
    The same fitted reducer is then used to transform validation/test data.
"""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_QUANTUM_COMPONENTS = 6


class QuantumFeatureReducer:
    """
    Preprocessing + PCA reducer for quantum machine-learning pipelines.

    Handles:
        - numerical features
        - categorical features
        - standardization
        - one-hot encoding
        - dimensionality reduction using PCA
    """

    def __init__(
        self,
        n_components=DEFAULT_QUANTUM_COMPONENTS,
        random_state=42,
    ):
        if not isinstance(n_components, int) or n_components < 1:
            raise ValueError("n_components must be a positive integer.")

        self.requested_components = n_components
        self.random_state = random_state

        self.preprocessor = None
        self.pca = None

        self.numeric_features = []
        self.categorical_features = []

        self.original_feature_count = 0
        self.transformed_feature_count = 0
        self.actual_components = 0

        self.explained_variance_ratio_ = None
        self.cumulative_explained_variance_ = None

        self.is_fitted = False

    def _validate_input(self, X):
        """
        Validate input feature data.
        """

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        if X.empty:
            raise ValueError("X cannot be empty.")

        if X.shape[1] == 0:
            raise ValueError("X must contain at least one feature.")

        return X.copy()

    def _build_preprocessor(self, X):
        """
        Build preprocessing pipeline based on the feature types.
        """

        self.numeric_features = X.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        self.categorical_features = X.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        transformers = []

        if self.numeric_features:
            numeric_pipeline = StandardScaler()

            transformers.append(
                (
                    "numeric",
                    numeric_pipeline,
                    self.numeric_features,
                )
            )

        if self.categorical_features:
            # sklearn >= 1.2
            try:
                encoder = OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                )
            except TypeError:
                # Compatibility with older sklearn versions
                encoder = OneHotEncoder(
                    handle_unknown="ignore",
                    sparse=False,
                )

            transformers.append(
                (
                    "categorical",
                    encoder,
                    self.categorical_features,
                )
            )

        if not transformers:
            raise ValueError(
                "No supported numeric or categorical features were found."
            )

        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )

    def fit(self, X):
        """
        Fit the preprocessing and PCA pipeline on training data.

        IMPORTANT:
            Call this only with training data.
        """

        X = self._validate_input(X)

        self.original_feature_count = X.shape[1]

        self._build_preprocessor(X)

        # Fit preprocessing
        X_prepared = self.preprocessor.fit_transform(X)

        X_prepared = np.asarray(X_prepared, dtype=float)

        if not np.isfinite(X_prepared).all():
            raise ValueError(
                "Preprocessed feature matrix contains non-finite values."
            )

        self.transformed_feature_count = X_prepared.shape[1]

        # PCA cannot have more components than either:
        #   - number of transformed features
        #   - number of training samples
        max_components = min(
            X_prepared.shape[0],
            X_prepared.shape[1],
        )

        if max_components < 1:
            raise ValueError(
                "Not enough data to perform dimensionality reduction."
            )

        self.actual_components = min(
            self.requested_components,
            max_components,
        )

        self.pca = PCA(
            n_components=self.actual_components,
            random_state=self.random_state,
        )

        self.pca.fit(X_prepared)

        self.explained_variance_ratio_ = (
            self.pca.explained_variance_ratio_
        )

        self.cumulative_explained_variance_ = np.cumsum(
            self.explained_variance_ratio_
        )

        self.is_fitted = True

        return self

    def transform(self, X):
        """
        Transform new data using the already-fitted reducer.

        This should be used for validation/test/inference data.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "QuantumFeatureReducer has not been fitted yet."
            )

        X = self._validate_input(X)

        X_prepared = self.preprocessor.transform(X)

        X_prepared = np.asarray(X_prepared, dtype=float)

        if not np.isfinite(X_prepared).all():
            raise ValueError(
                "Preprocessed feature matrix contains non-finite values."
            )

        X_reduced = self.pca.transform(X_prepared)

        return X_reduced

    def fit_transform(self, X):
        """
        Fit the reducer and transform the same dataset.

        Use this for training data only.
        """

        self.fit(X)

        return self.transform(X)

    def get_feature_names(self):
        """
        Return names for the reduced quantum dimensions.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "QuantumFeatureReducer has not been fitted yet."
            )

        return [
            f"quantum_feature_{i + 1}"
            for i in range(self.actual_components)
        ]

    def get_metadata(self):
        """
        Return information about the reduction process.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "QuantumFeatureReducer has not been fitted yet."
            )

        return {
            "method": "StandardScaler + OneHotEncoder + PCA",
            "original_feature_count": self.original_feature_count,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "preprocessed_feature_count": self.transformed_feature_count,
            "requested_components": self.requested_components,
            "actual_components": self.actual_components,
            "explained_variance_ratio": (
                self.explained_variance_ratio_.tolist()
            ),
            "cumulative_explained_variance": (
                self.cumulative_explained_variance_.tolist()
            ),
        }


def fit_quantum_reducer(
    X_train,
    n_components=DEFAULT_QUANTUM_COMPONENTS,
    random_state=42,
):
    """
    Convenience function for fitting a quantum reducer.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training features only.

    n_components : int
        Desired number of quantum dimensions/qubits.

    Returns
    -------
    QuantumFeatureReducer
        Fitted reducer.
    """

    reducer = QuantumFeatureReducer(
        n_components=n_components,
        random_state=random_state,
    )

    reducer.fit(X_train)

    return reducer


def transform_quantum_features(reducer, X):
    """
    Transform features using an already-fitted reducer.
    """

    if not isinstance(reducer, QuantumFeatureReducer):
        raise TypeError(
            "reducer must be a QuantumFeatureReducer instance."
        )

    return reducer.transform(X)


def fit_transform_quantum_reducer(
    X_train,
    n_components=DEFAULT_QUANTUM_COMPONENTS,
    random_state=42,
):
    """
    Convenience function that fits and transforms training data.

    Returns
    -------
    X_reduced : numpy.ndarray
    reducer : QuantumFeatureReducer
    """

    reducer = fit_quantum_reducer(
        X_train=X_train,
        n_components=n_components,
        random_state=random_state,
    )

    X_reduced = reducer.transform(X_train)

    return X_reduced, reducer


# ---------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------

def standalone_test():
    """
    Test the reducer using the current heart dataset.

    This is only a demonstration.
    The actual platform will fit the reducer on training data
    inside the complete training pipeline.
    """

    print("=" * 60)
    print("QUANTUM FEATURE REDUCER TEST")
    print("=" * 60)

    path = "data/heart_cleaned.csv"

    df = pd.read_csv(path)

    if "target" not in df.columns:
        raise ValueError(
            "Expected 'target' column in heart dataset."
        )

    X = df.drop(columns=["target"])

    print(f"Original dataset shape: {df.shape}")
    print(f"Original feature count: {X.shape[1]}")

    reducer = QuantumFeatureReducer(
        n_components=DEFAULT_QUANTUM_COMPONENTS
    )

    X_reduced = reducer.fit_transform(X)

    metadata = reducer.get_metadata()

    print()
    print("Original features:")
    for feature in X.columns:
        print(f"  - {feature}")

    print()
    print(
        "Preprocessed feature count:",
        metadata["preprocessed_feature_count"],
    )

    print(
        "Requested quantum components:",
        metadata["requested_components"],
    )

    print(
        "Actual quantum components:",
        metadata["actual_components"],
    )

    print(
        "Reduced shape:",
        X_reduced.shape,
    )

    print()
    print("Quantum feature names:")

    for feature in reducer.get_feature_names():
        print(f"  - {feature}")

    print()
    print("Explained variance ratio:")

    for i, value in enumerate(
        metadata["explained_variance_ratio"],
        start=1,
    ):
        print(f"  Component {i}: {value:.6f}")

    print()
    print("Cumulative explained variance:")

    for i, value in enumerate(
        metadata["cumulative_explained_variance"],
        start=1,
    ):
        print(f"  Component {i}: {value:.6f}")

    print()
    print("First 5 reduced samples:")

    print(
        pd.DataFrame(
            X_reduced[:5],
            columns=reducer.get_feature_names(),
        )
    )

    print()
    print("STATUS: quantum reduction successful")
    print("=" * 60)


if __name__ == "__main__":
    standalone_test()