import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import pennylane as qml
from pennylane import numpy as np


# =========================
# 1. FINAL FEATURES
# =========================

SELECTED_FEATURES = [
    "cp",
    "thal",
    "thalach",
    "oldpeak",
    "ca",
    "age"
]


# =========================
# 2. LOAD DATA
# =========================

df = pd.read_csv("data/heart_cleaned.csv")

X = df[SELECTED_FEATURES]
y = df["target"]

print("Selected features:")
print(SELECTED_FEATURES)

print("\nDataset shape:", X.shape)


# =========================
# 3. TRAIN-TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# 4. SCALE FEATURES
# =========================

# Quantum rotation angles work better
# when our input values are mapped to [0, pi].

scaler = MinMaxScaler(
    feature_range=(0, np.pi)
)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


print("\nFirst scaled training sample:")
print(X_train_scaled[0])


# =========================
# 5. QUANTUM DEVICE
# =========================

n_qubits = len(SELECTED_FEATURES)

dev = qml.device(
    "default.qubit",
    wires=n_qubits
)


# =========================
# 6. QUANTUM FEATURE MAP
# =========================

@qml.qnode(dev)
def quantum_feature_map(features):

    # Encode each feature into one qubit
    for i in range(n_qubits):
        qml.RY(
            features[i],
            wires=i
        )

    # Entangle neighbouring qubits
    for i in range(n_qubits - 1):
        qml.CNOT(
            wires=[i, i + 1]
        )

    return qml.state()


# =========================
# 7. TEST REAL DATA
# =========================

sample = X_train_scaled[0]

state = quantum_feature_map(sample)

print("\nQuantum state generated successfully.")

print("Number of qubits:", n_qubits)

print("State vector length:", len(state))


# =========================
# 8. DISPLAY CIRCUIT
# =========================

print("\nQuantum feature map:")

print(
    qml.draw(quantum_feature_map)(sample)
)