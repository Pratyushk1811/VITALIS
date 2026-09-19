import os
import sys
import json
import joblib
import pandas as pd


# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# ============================================================
# LOAD SAVED MODELS
# ============================================================

scaler = joblib.load(
    os.path.join(
        MODELS_DIR,
        "scaler.joblib"
    )
)

logistic_regression = joblib.load(
    os.path.join(
        MODELS_DIR,
        "logistic_regression.joblib"
    )
)

random_forest = joblib.load(
    os.path.join(
        MODELS_DIR,
        "random_forest.joblib"
    )
)

rbf_svm = joblib.load(
    os.path.join(
        MODELS_DIR,
        "rbf_svm.joblib"
    )
)


# ============================================================
# FEATURE ORDER
# ============================================================

SELECTED_FEATURES = [
    "cp",
    "thal",
    "thalach",
    "oldpeak",
    "ca",
    "age"
]


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_classical(patient):
    """
    Run the three trained classical models on one patient.

    Expected patient dictionary:

    {
        "cp": 2,
        "thal": 3,
        "thalach": 150,
        "oldpeak": 1.2,
        "ca": 0,
        "age": 54
    }
    """

    # --------------------------------------------------------
    # 1. Validate required features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in SELECTED_FEATURES
        if feature not in patient
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )


    # --------------------------------------------------------
    # 2. Create dataframe in EXACT training order
    # --------------------------------------------------------

    X = pd.DataFrame(
        [[patient[feature] for feature in SELECTED_FEATURES]],
        columns=SELECTED_FEATURES
    )


    # --------------------------------------------------------
    # 3. Create scaled version
    # --------------------------------------------------------

    X_scaled = scaler.transform(X)


    # --------------------------------------------------------
    # 4. Logistic Regression
    # --------------------------------------------------------

    lr_prediction = int(
        logistic_regression.predict(X_scaled)[0]
    )

    lr_probability = float(
        logistic_regression.predict_proba(
            X_scaled
        )[0][lr_prediction]
    )


    # --------------------------------------------------------
    # 5. Random Forest
    #
    # IMPORTANT:
    # Random Forest was trained WITHOUT scaling.
    # --------------------------------------------------------

    rf_prediction = int(
        random_forest.predict(X)[0]
    )

    rf_probability = float(
        random_forest.predict_proba(
            X
        )[0][rf_prediction]
    )


    # --------------------------------------------------------
    # 6. RBF SVM
    #
    # SVM was trained using scaled features.
    # --------------------------------------------------------

    svm_prediction = int(
        rbf_svm.predict(X_scaled)[0]
    )

    svm_probability = float(
        rbf_svm.predict_proba(
            X_scaled
        )[0][svm_prediction]
    )


    # --------------------------------------------------------
    # 7. Return structured result
    # --------------------------------------------------------

    return {

        "lr": {
            "prediction": lr_prediction,
            "score": lr_probability
        },

        "rf": {
            "prediction": rf_prediction,
            "score": rf_probability
        },

        "svm": {
            "prediction": svm_prediction,
            "score": svm_probability
        }

    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_patient = {

        "cp": 2,
        "thal": 3,
        "thalach": 150,
        "oldpeak": 1.2,
        "ca": 0,
        "age": 54

    }

    print("=" * 50)
    print("VITALIS CLASSICAL INFERENCE")
    print("=" * 50)

    print("\nPatient:")
    for feature in SELECTED_FEATURES:
        print(
            f"{feature}: {test_patient[feature]}"
        )


    result = predict_classical(
        test_patient
    )


    print("\nResults:")

    print(
        "\nLogistic Regression"
    )

    print(
        "Prediction:",
        result["lr"]["prediction"]
    )

    print(
        "Probability:",
        f"{result['lr']['score']:.4f}"
    )


    print(
        "\nRandom Forest"
    )

    print(
        "Prediction:",
        result["rf"]["prediction"]
    )

    print(
        "Probability:",
        f"{result['rf']['score']:.4f}"
    )


    print(
        "\nRBF SVM"
    )

    print(
        "Prediction:",
        result["svm"]["prediction"]
    )

    print(
        "Probability:",
        f"{result['svm']['score']:.4f}"
    )

    print("\n" + "=" * 50)