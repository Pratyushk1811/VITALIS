import joblib
import pandas as pd
import pennylane as qml

from pennylane import numpy as np


# ============================================================
# 1. FINAL FEATURES
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
# 2. LOAD SAVED QUANTUM MODEL ARTIFACTS
# ============================================================

scaler = joblib.load(
    "models/quantum_scaler.joblib"
)

X_train_scaled = joblib.load(
    "models/quantum_training_data.joblib"
)

qsvm = joblib.load(
    "models/qsvm.joblib"
)


# ============================================================
# 3. QUANTUM DEVICE
# ============================================================

n_qubits = len(SELECTED_FEATURES)

dev = qml.device(
    "default.qubit",
    wires=n_qubits
)


# ============================================================
# 4. QUANTUM FEATURE MAP
# ============================================================

@qml.qnode(dev)
def feature_map(x):

    # Encode each selected feature into a qubit
    for i in range(n_qubits):
        qml.RY(
            x[i],
            wires=i
        )

    # Entangle neighboring qubits
    for i in range(n_qubits - 1):
        qml.CNOT(
            wires=[i, i + 1]
        )

    return qml.state()


# ============================================================
# 5. QUANTUM KERNEL
# ============================================================

def quantum_kernel(x1, x2):

    state1 = feature_map(x1)
    state2 = feature_map(x2)

    overlap = np.vdot(
        state1,
        state2
    )

    return float(
        np.abs(overlap) ** 2
    )


# ============================================================
# 6. INPUT VALIDATION
# ============================================================

def validate_patient(patient):

    if not isinstance(patient, dict):
        raise TypeError(
            "Patient data must be provided as a dictionary."
        )

    # Check for missing features
    missing_features = [
        feature
        for feature in SELECTED_FEATURES
        if feature not in patient
    ]

    if missing_features:
        raise ValueError(
            f"Missing patient features: {missing_features}"
        )

    # Check for extra features
    extra_features = [
        feature
        for feature in patient
        if feature not in SELECTED_FEATURES
    ]

    if extra_features:
        raise ValueError(
            f"Unexpected patient features: {extra_features}"
        )

    # Check that values are numeric
    for feature in SELECTED_FEATURES:

        value = patient[feature]

        if not isinstance(
            value,
            (int, float)
        ):
            raise TypeError(
                f"{feature} must be numeric."
            )


# ============================================================
# 7. QUANTUM PREDICTION
# ============================================================

def predict_quantum(patient):

    # Validate input
    validate_patient(patient)

    # --------------------------------------------------------
    # Convert patient dictionary into DataFrame
    # --------------------------------------------------------

    patient_df = pd.DataFrame(
        [[
            patient[feature]
            for feature in SELECTED_FEATURES
        ]],
        columns=SELECTED_FEATURES
    )

    # --------------------------------------------------------
    # Apply the SAME scaler used during training
    # --------------------------------------------------------

    patient_scaled = scaler.transform(
        patient_df
    )[0]

    # --------------------------------------------------------
    # Calculate quantum kernel between the new patient
    # and every training patient
    # --------------------------------------------------------

    kernel_values = []

    for training_patient in X_train_scaled:

        value = quantum_kernel(
            patient_scaled,
            training_patient
        )

        kernel_values.append(value)

    # SVC with kernel="precomputed" expects:
    #
    #     (number_of_test_samples,
    #      number_of_training_samples)
    #
    K_new = np.array(
        [kernel_values]
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = qsvm.predict(
        K_new
    )[0]

    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    probability = qsvm.predict_proba(
        K_new
    )[0]

    # --------------------------------------------------------
    # Return structured result
    # --------------------------------------------------------

    result = {
        "prediction": int(prediction),

        "class_0_probability": round(
            float(probability[0]),
            4
        ),

        "class_1_probability": round(
            float(probability[1]),
            4
        )
    }

    return result


# ============================================================
# 8. TEST THE MODEL DIRECTLY
# ============================================================

if __name__ == "__main__":

    # Example patient
    patient = {
        "cp": 2,
        "thal": 3,
        "thalach": 150,
        "oldpeak": 1.2,
        "ca": 0,
        "age": 54
    }

    result = predict_quantum(
        patient
    )

    print("\n" + "=" * 50)
    print("VITALIS QUANTUM INFERENCE")
    print("=" * 50)

    print("\nPatient:")

    for feature in SELECTED_FEATURES:
        print(
            f"{feature}: {patient[feature]}"
        )

    print("\nResult:")

    print(
        "Prediction:",
        result["prediction"]
    )

    print(
        "Class 0 Probability:",
        result["class_0_probability"]
    )

    print(
        "Class 1 Probability:",
        result["class_1_probability"]
    )