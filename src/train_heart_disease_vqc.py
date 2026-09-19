import json
import time
import joblib

import numpy as onp
import pandas as pd

import pennylane as qml
from pennylane import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from src.heart_disease_vqc import (
    vqc_circuit,
    N_LAYERS,
    N_QUBITS,
    N_FEATURES,
)


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
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(
    "data/heart_cleaned.csv"
)

X = df[
    SELECTED_FEATURES
]

y = df[
    "target"
].astype(int)

print("Dataset:", X.shape)
print("Labels:", y.shape)
print("Selected features:", SELECTED_FEATURES)


# ============================================================
# 3. TRAIN / TEST SPLIT
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
# 4. QUANTUM SCALING
# ============================================================

# Quantum gates use the features as rotation angles.
# Therefore map all features to [0, pi].

scaler = MinMaxScaler(
    feature_range=(0, onp.pi)
)

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# 5. CONVERT TO PENNYLANE NUMPY
# ============================================================

X_train_q = np.array(
    X_train_scaled,
    requires_grad=False
)

X_test_q = np.array(
    X_test_scaled,
    requires_grad=False
)


# ============================================================
# 6. VQC OUTPUT → PROBABILITY
# ============================================================

def prediction_probability(
    x,
    weights
):

    expectation = vqc_circuit(
        x,
        weights
    )

    # Pauli-Z expectation is in [-1, +1].
    #
    # Convert:
    #
    # [-1, +1] → [0, 1]
    #
    # so it can be interpreted as a
    # model probability.

    probability = (
        expectation + 1
    ) / 2

    return probability


# ============================================================
# 7. LOSS FUNCTION
# ============================================================

def loss_function(
    weights,
    X_batch,
    y_batch
):

    losses = []

    for x, y_true in zip(
        X_batch,
        y_batch
    ):

        probability = prediction_probability(
            x,
            weights
        )

        # Prevent log(0)

        epsilon = 1e-7

        probability = np.clip(
            probability,
            epsilon,
            1 - epsilon
        )

        loss = -(
            y_true * np.log(
                probability
            )
            +
            (1 - y_true)
            * np.log(
                1 - probability
            )
        )

        losses.append(
            loss
        )

    return np.mean(
        np.array(losses)
    )


# ============================================================
# 8. INITIALIZE TRAINABLE PARAMETERS
# ============================================================

rng = onp.random.default_rng(
    42
)

initial_weights = rng.uniform(
    -0.1,
    0.1,
    size=(
        N_LAYERS,
        2 * N_QUBITS
    )
)

weights = np.array(
    initial_weights,
    requires_grad=True
)

print(
    "\nTrainable parameters:",
    weights.size
)


# ============================================================
# 9. OPTIMIZER
# ============================================================

optimizer = qml.AdamOptimizer(
    stepsize=0.01
)


# ============================================================
# 10. TRAINING CONFIGURATION
# ============================================================

EPOCHS = 50
BATCH_SIZE = 32

print("\n" + "=" * 60)
print("TRAINING HEART DISEASE VQC V2")
print("=" * 60)

print(
    f"Qubits:              {N_QUBITS}"
)

print(
    f"Features:            {N_FEATURES}"
)

print(
    f"Variational layers:  {N_LAYERS}"
)

print(
    f"Parameters:          {weights.size}"
)

print(
    f"Learning rate:       0.01"
)

print(
    f"Epochs:              {EPOCHS}"
)

print(
    f"Batch size:          {BATCH_SIZE}"
)

start_time = time.time()


# ============================================================
# 11. TRAINING LOOP
# ============================================================

for epoch in range(
    EPOCHS
):

    # --------------------------------------------------------
    # Shuffle training data
    # --------------------------------------------------------

    indices = rng.permutation(
        len(X_train_q)
    )

    X_train_shuffled = (
        X_train_q[indices]
    )

    y_train_shuffled = (
        y_train.values[indices]
    )

    batch_losses = []

    # --------------------------------------------------------
    # Mini-batch training
    # --------------------------------------------------------

    for start in range(
        0,
        len(X_train_q),
        BATCH_SIZE
    ):

        end = start + BATCH_SIZE

        X_batch = (
            X_train_shuffled[start:end]
        )

        y_batch = (
            y_train_shuffled[start:end]
        )

        weights, batch_loss = (
            optimizer.step_and_cost(
                lambda w:
                    loss_function(
                        w,
                        X_batch,
                        y_batch
                    ),
                weights
            )
        )

        batch_losses.append(
            float(batch_loss)
        )

    # --------------------------------------------------------
    # Mean epoch loss
    # --------------------------------------------------------

    epoch_loss = (
        sum(batch_losses)
        /
        len(batch_losses)
    )

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} "
        f"| Loss: {epoch_loss:.6f}"
    )


# ============================================================
# 12. TRAINING TIME
# ============================================================

training_time = (
    time.time()
    - start_time
)

print(
    f"\nTraining completed in "
    f"{training_time:.2f} seconds"
)


# ============================================================
# 13. GENERATE TEST PROBABILITIES
# ============================================================

print("\n" + "=" * 60)
print("EVALUATING HEART DISEASE VQC V2")
print("=" * 60)

probabilities = []

for x in X_test_q:

    probability = prediction_probability(
        x,
        weights
    )

    probabilities.append(
        float(probability)
    )


# ============================================================
# 14. CONVERT TO STANDARD NUMPY
# ============================================================

y_prob = onp.array(
    probabilities,
    dtype=float
)


# ============================================================
# 15. CLASS PREDICTIONS
# ============================================================

y_pred = (
    y_prob >= 0.5
).astype(int)


# ============================================================
# 16. DEBUG OUTPUT
# ============================================================

print("\n" + "-" * 60)
print("VQC OUTPUT DEBUG")
print("-" * 60)

print(
    "y_test unique:",
    onp.unique(y_test)
)

print(
    "y_pred unique:",
    onp.unique(y_pred)
)

print(
    "y_prob min:",
    y_prob.min()
)

print(
    "y_prob max:",
    y_prob.max()
)

print(
    "y_prob shape:",
    y_prob.shape
)

print(
    "y_pred shape:",
    y_pred.shape
)

print(
    "y_test shape:",
    y_test.shape
)

print(
    "Predicted class counts:",
    onp.unique(
        y_pred,
        return_counts=True
    )
)

print(
    "Actual class counts:",
    onp.unique(
        y_test,
        return_counts=True
    )
)


# ============================================================
# 17. SAFETY CHECKS
# ============================================================

if not onp.all(
    onp.isfinite(y_prob)
):

    raise ValueError(
        "VQC produced NaN or infinite probabilities."
    )


if len(y_prob) != len(y_test):

    raise ValueError(
        f"Prediction length mismatch: "
        f"{len(y_prob)} predictions vs "
        f"{len(y_test)} test labels."
    )


unique_predictions = onp.unique(
    y_pred
)

if not onp.all(
    onp.isin(
        unique_predictions,
        [0, 1]
    )
):

    raise ValueError(
        f"Unexpected prediction classes: "
        f"{unique_predictions}"
    )


# ============================================================
# 18. METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="binary",
    zero_division=0
)

sensitivity = recall_score(
    y_test,
    y_pred,
    average="binary",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="binary",
    zero_division=0
)

auc = roc_auc_score(
    y_test,
    y_prob
)


# ============================================================
# 19. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

if (
    tn + fp
) > 0:

    specificity = (
        tn
        /
        (tn + fp)
    )

else:

    specificity = 0.0


# ============================================================
# 20. RESULTS
# ============================================================

print("\n" + "=" * 60)
print("HEART DISEASE VQC V2 RESULTS")
print("=" * 60)

print(
    f"Accuracy:       {accuracy:.4f}"
)

print(
    f"Precision:      {precision:.4f}"
)

print(
    f"Sensitivity:    {sensitivity:.4f}"
)

print(
    f"Specificity:    {specificity:.4f}"
)

print(
    f"F1 Score:       {f1:.4f}"
)

print(
    f"ROC-AUC:        {auc:.4f}"
)

print(
    "\nConfusion Matrix:"
)

print(cm)


# ============================================================
# 21. SAVE BENCHMARK
# ============================================================

results = {

    "model":
        "Variational Quantum Classifier",

    "version":
        "V2",

    "dataset":
        "UCI Cleveland Heart Disease",

    "selected_features":
        SELECTED_FEATURES,

    "features":
        N_FEATURES,

    "n_qubits":
        N_QUBITS,

    "reuploading_layers":
        N_LAYERS,

    "trainable_parameters":
        int(weights.size),

    "epochs":
        EPOCHS,

    "batch_size":
        BATCH_SIZE,

    "learning_rate":
        0.01,

    "test_size":
        0.2,

    "random_state":
        42,

    "accuracy":
        round(
            float(accuracy),
            4
        ),

    "precision":
        round(
            float(precision),
            4
        ),

    "sensitivity":
        round(
            float(sensitivity),
            4
        ),

    "specificity":
        round(
            float(specificity),
            4
        ),

    "f1_score":
        round(
            float(f1),
            4
        ),

    "roc_auc":
        round(
            float(auc),
            4
        ),

    "training_time_seconds":
        round(
            float(training_time),
            4
        )
}


# ============================================================
# 22. SAVE BENCHMARK JSON
# ============================================================

with open(
    "models/heart_disease_vqc_benchmark.json",
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


# ============================================================
# 23. SAVE SCALER
# ============================================================

joblib.dump(
    scaler,
    "models/heart_disease_vqc_scaler.joblib"
)


# ============================================================
# 24. SAVE TRAINED WEIGHTS
# ============================================================

final_weights = onp.asarray(
    weights,
    dtype=float
)

joblib.dump(
    final_weights,
    "models/heart_disease_vqc_weights.joblib"
)


# ============================================================
# 25. FINAL CONFIRMATION
# ============================================================

print("\nSaved:")

print(
    "models/heart_disease_vqc_benchmark.json"
)

print(
    "models/heart_disease_vqc_scaler.joblib"
)

print(
    "models/heart_disease_vqc_weights.joblib"
)

print(
    "\nHeart disease VQC V2 training and "
    "evaluation completed successfully."
)