import json
import time
import joblib

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from src.breast_cancer_quantum import breast_cancer_feature_map


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(
    "data/bcancer.csv",
    header=None
)

X = df.iloc[:, 2:].astype(float).values

# M = malignant = 1
# B = benign = 0
y = df.iloc[:, 1].map({
    "M": 1,
    "B": 0
}).values

print("Dataset shape:", df.shape)
print("Feature matrix:", X.shape)
print("Labels:", y.shape)


# ============================================================
# 2. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 3. SCALE FEATURES
# ============================================================

scaler = MinMaxScaler(
    feature_range=(0, np.pi)
)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 4. GENERATE QUANTUM STATES
# ============================================================

print("\n" + "=" * 60)
print("GENERATING TRAINING QUANTUM STATES")
print("=" * 60)

start_time = time.time()

train_states = np.array([
    breast_cancer_feature_map(x)
    for x in X_train_scaled
])

train_state_time = time.time() - start_time

print(
    f"Training quantum states generated in "
    f"{train_state_time:.3f} seconds"
)


print("\n" + "=" * 60)
print("GENERATING TEST QUANTUM STATES")
print("=" * 60)

start_time = time.time()

test_states = np.array([
    breast_cancer_feature_map(x)
    for x in X_test_scaled
])

test_state_time = time.time() - start_time

print(
    f"Test quantum states generated in "
    f"{test_state_time:.3f} seconds"
)


# ============================================================
# 5. BUILD QUANTUM KERNEL MATRICES
# ============================================================

print("\n" + "=" * 60)
print("BUILDING TRAINING QUANTUM KERNEL")
print("=" * 60)

start_time = time.time()

K_train = np.abs(
    train_states @ train_states.conj().T
) ** 2

train_kernel_time = time.time() - start_time

print(
    f"Training kernel generated in "
    f"{train_kernel_time:.3f} seconds"
)

print("Training kernel shape:", K_train.shape)


print("\n" + "=" * 60)
print("BUILDING TEST QUANTUM KERNEL")
print("=" * 60)

start_time = time.time()

K_test = np.abs(
    test_states @ train_states.conj().T
) ** 2

test_kernel_time = time.time() - start_time

print(
    f"Test kernel generated in "
    f"{test_kernel_time:.3f} seconds"
)

print("Test kernel shape:", K_test.shape)


# ============================================================
# 6. TRAIN QSVM
# ============================================================

print("\n" + "=" * 60)
print("TRAINING QUANTUM KERNEL SVM")
print("=" * 60)

qsvm = SVC(
    kernel="precomputed",
    probability=True,
    random_state=42
)

qsvm.fit(
    K_train,
    y_train
)


# ============================================================
# 7. PREDICTIONS
# ============================================================

y_pred = qsvm.predict(K_test)

y_prob = qsvm.predict_proba(K_test)[:, 1]


# ============================================================
# 8. METRICS
# ============================================================

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


# ============================================================
# 9. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("BREAST CANCER QSVM RESULTS")
print("=" * 60)

print(f"Accuracy:       {accuracy:.4f}")
print(f"Precision:      {precision:.4f}")
print(f"Sensitivity:    {sensitivity:.4f}")
print(f"Specificity:    {specificity:.4f}")
print(f"F1 Score:       {f1:.4f}")
print(f"ROC-AUC:        {auc:.4f}")


# ============================================================
# 10. SAVE BENCHMARK
# ============================================================

results = {
    "model": "Quantum Kernel SVM",
    "dataset": "Breast Cancer Wisconsin Diagnostic",
    "features": 30,
    "n_qubits": 6,
    "reuploading_layers": 5,
    "accuracy": round(float(accuracy), 4),
    "precision": round(float(precision), 4),
    "sensitivity": round(float(sensitivity), 4),
    "specificity": round(float(specificity), 4),
    "f1_score": round(float(f1), 4),
    "roc_auc": round(float(auc), 4),
    "train_state_time_seconds": round(
        float(train_state_time), 4
    ),
    "test_state_time_seconds": round(
        float(test_state_time), 4
    ),
    "train_kernel_time_seconds": round(
        float(train_kernel_time), 4
    ),
    "test_kernel_time_seconds": round(
        float(test_kernel_time), 4
    ),
}

with open(
    "models/breast_cancer_quantum_benchmark.json",
    "w"
) as f:
    json.dump(
        results,
        f,
        indent=4
    )


# ============================================================
# 11. SAVE MODEL ARTIFACTS
# ============================================================

joblib.dump(
    scaler,
    "models/breast_cancer_quantum_scaler.joblib"
)

joblib.dump(
    train_states,
    "models/breast_cancer_quantum_training_states.joblib"
)

joblib.dump(
    y_train,
    "models/breast_cancer_quantum_training_labels.joblib"
)

joblib.dump(
    qsvm,
    "models/breast_cancer_qsvm.joblib"
)


print("\nSaved:")
print("models/breast_cancer_quantum_benchmark.json")
print("models/breast_cancer_quantum_scaler.joblib")
print("models/breast_cancer_quantum_training_states.joblib")
print("models/breast_cancer_quantum_training_labels.joblib")
print("models/breast_cancer_qsvm.joblib")