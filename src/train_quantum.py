import json
import time

import pandas as pd
import pennylane as qml

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

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

print("Dataset shape:", df.shape)
print("Quantum features:", SELECTED_FEATURES)


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

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================
# 4. NORMALIZATION
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

    for i in range(n_qubits):
        qml.RY(x[i], wires=i)

    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])

    return qml.state()


# =========================
# 7. QUANTUM KERNEL
# =========================

def quantum_kernel(x1, x2):

    state1 = feature_map(x1)
    state2 = feature_map(x2)

    overlap = np.vdot(state1, state2)

    return float(np.abs(overlap) ** 2)


# =========================
# 8. BUILD KERNEL MATRIX
# =========================

def build_kernel_matrix(X1, X2):

    matrix = np.zeros(
        (len(X1), len(X2))
    )

    total = len(X1) * len(X2)

    completed = 0

    start_time = time.time()

    for i in range(len(X1)):

        for j in range(len(X2)):

            matrix[i, j] = quantum_kernel(
                X1[i],
                X2[j]
            )

            completed += 1

        # Progress every row
        elapsed = time.time() - start_time

        print(
            f"Completed {completed}/{total} "
            f"kernel evaluations "
            f"({elapsed:.1f}s)"
        )

    return matrix


# =========================
# 9. TRAINING KERNEL MATRIX
# =========================

print("\n" + "=" * 50)
print("BUILDING TRAINING QUANTUM KERNEL")
print("=" * 50)

K_train = build_kernel_matrix(
    X_train_scaled,
    X_train_scaled
)



# 10. TEST KERNEL MATRIX


print("\n" + "=" * 50)
print("BUILDING TEST QUANTUM KERNEL")
print("=" * 50)

K_test = build_kernel_matrix(
    X_test_scaled,
    X_train_scaled
)



# 11. TRAIN QSVM


print("\n" + "=" * 50)
print("TRAINING QUANTUM KERNEL SVM")
print("=" * 50)

qsvm = SVC(
    kernel="precomputed",
    probability=True,
    random_state=42
)

qsvm.fit(
    K_train,
    y_train
)


# =========================
# 12. PREDICTIONS
# =========================

y_pred = qsvm.predict(K_test)

y_prob = qsvm.predict_proba(K_test)[:, 1]


# =========================
# 13. METRICS
# =========================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

sensitivity = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

auc = roc_auc_score(
    y_test,
    y_prob
)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    y_pred
).ravel()

specificity = tn / (tn + fp)


# =========================
# 14. DISPLAY RESULTS
# =========================

print("\n" + "=" * 50)
print("QUANTUM KERNEL SVM RESULTS")
print("=" * 50)

print(f"Accuracy:     {accuracy:.4f}")
print(f"Precision:    {precision:.4f}")
print(f"Sensitivity:  {sensitivity:.4f}")
print(f"Specificity:  {specificity:.4f}")
print(f"F1 Score:     {f1:.4f}")
print(f"ROC-AUC:      {auc:.4f}")


# =========================
# 15. SAVE RESULTS
# =========================

results = {
    "model": "Quantum Kernel SVM",
    "features": SELECTED_FEATURES,
    "n_qubits": n_qubits,
    "accuracy": round(float(accuracy), 4),
    "precision": round(float(precision), 4),
    "sensitivity": round(float(sensitivity), 4),
    "specificity": round(float(specificity), 4),
    "f1_score": round(float(f1), 4),
    "roc_auc": round(float(auc), 4)
}

with open(
    "models/quantum_benchmark.json",
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


print("\nSaved:")
print("models/quantum_benchmark.json")
import joblib

joblib.dump(
    scaler,
    "models/quantum_scaler.joblib"
)

joblib.dump(
    X_train_scaled,
    "models/quantum_training_data.joblib"
)

joblib.dump(
    y_train.to_numpy(),
    "models/quantum_training_labels.joblib"
)

joblib.dump(
    qsvm,
    "models/qsvm.joblib"
)

print("models/quantum_scaler.joblib")
print("models/quantum_training_data.joblib")
print("models/quantum_training_labels.joblib")
print("models/qsvm.joblib")