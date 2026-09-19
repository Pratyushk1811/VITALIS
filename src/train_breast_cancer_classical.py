import json
import joblib

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(
    "data/bcancer.csv",
    header=None
)

X = df.iloc[:, 2:].astype(float)

y = df.iloc[:, 1].map({
    "B": 0,
    "M": 1
})


print("Dataset shape:", df.shape)
print("Feature matrix:", X.shape)


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

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 3. MODELS
# ============================================================

models = {

    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "random_forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "rbf_svm": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        ))
    ])
}


# ============================================================
# 4. TRAIN + EVALUATE
# ============================================================

results = {}

for name, model in models.items():

    print("\n" + "=" * 60)
    print(f"TRAINING: {name.upper()}")
    print("=" * 60)

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

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

    results[name] = {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4)
    }

    print(f"Accuracy:       {accuracy:.4f}")
    print(f"Precision:      {precision:.4f}")
    print(f"Sensitivity:    {sensitivity:.4f}")
    print(f"Specificity:    {specificity:.4f}")
    print(f"F1 Score:       {f1:.4f}")
    print(f"ROC-AUC:        {auc:.4f}")


# ============================================================
# 5. SAVE BENCHMARK
# ============================================================

benchmark = {
    "dataset": "Breast Cancer Wisconsin Diagnostic",
    "features": 30,
    "train_samples": len(X_train),
    "test_samples": len(X_test),
    "random_state": 42,
    "models": results
}

with open(
    "models/breast_cancer_classical_benchmark.json",
    "w"
) as f:

    json.dump(
        benchmark,
        f,
        indent=4
    )


# ============================================================
# 6. SAVE MODELS
# ============================================================

joblib.dump(
    models["logistic_regression"],
    "models/breast_cancer_logistic_regression.joblib"
)

joblib.dump(
    models["random_forest"],
    "models/breast_cancer_random_forest.joblib"
)

joblib.dump(
    models["rbf_svm"],
    "models/breast_cancer_rbf_svm.joblib"
)


print("\n" + "=" * 60)
print("SAVED")
print("=" * 60)

print("models/breast_cancer_classical_benchmark.json")
print("models/breast_cancer_logistic_regression.joblib")
print("models/breast_cancer_random_forest.joblib")
print("models/breast_cancer_rbf_svm.joblib")