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
# 4. NORMALIZE FEATURES
# =========================

scaler = MinMaxScaler(
    feature_range=(0, np.pi)
)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


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
def feature_map(x):

    # Encode classical features
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)

    # Entangle neighbouring qubits
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])

    return qml.state()


# =========================
# 7. QUANTUM KERNEL
# =========================

def quantum_kernel(x1, x2):

    state1 = feature_map(x1)
    state2 = feature_map(x2)

    # Quantum state overlap
    overlap = np.vdot(state1, state2)

    # Kernel value
    kernel_value = np.abs(overlap) ** 2

    return float(kernel_value)


# =========================
# 8. TEST TWO PATIENTS
# =========================

patient_a = X_train_scaled[0]
patient_b = X_train_scaled[1]

similarity = quantum_kernel(
    patient_a,
    patient_b
)


print("Quantum kernel similarity:")
print(similarity)


# =========================
# 9. BASIC SANITY CHECKS
# =========================

print("\nKernel self-similarity:")

print(
    "K(A,A) =",
    quantum_kernel(patient_a, patient_a)
)

print(
    "K(B,B) =",
    quantum_kernel(patient_b, patient_b)
)

print(
    "K(A,B) =",
    quantum_kernel(patient_a, patient_b)
)