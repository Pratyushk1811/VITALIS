import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

from final_features import SELECTED_FEATURES


# =========================
# 1. CREATE MODELS FOLDER
# =========================

os.makedirs("models", exist_ok=True)


# =========================
# 2. LOAD DATA
# =========================

df = pd.read_csv("data/heart_cleaned.csv")

print("Dataset shape:", df.shape)


# =========================
# 3. SELECT FINAL FEATURES
# =========================

X = df[SELECTED_FEATURES]
y = df["target"]

print("\nSelected features:")
print(SELECTED_FEATURES)


# =========================
# 4. TRAIN-TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================
# 5. FEATURE SCALING
# =========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, "models/scaler.joblib")


# =========================
# 6. DEFINE MODELS
# =========================

models = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000),
        "filename": "models/logistic_regression.joblib",
        "scaled": True
    },

    "Random Forest": {
        "model": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),
        "filename": "models/random_forest.joblib",
        "scaled": False
    },

    "RBF SVM": {
        "model": SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        ),
        "filename": "models/rbf_svm.joblib",
        "scaled": True
    }
}


# =========================
# 7. TRAIN, EVALUATE, SAVE
# =========================

benchmark = {}

for name, config in models.items():

    print("\n" + "=" * 45)
    print(name)
    print("=" * 45)

    model = config["model"]

    if config["scaled"]:
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

    else:
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred
    ).ravel()

    specificity = tn / (tn + fp)

    # Store results
    benchmark[name] = {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4)
    }

    # Print results
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Precision:    {precision:.4f}")
    print(f"Sensitivity:  {sensitivity:.4f}")
    print(f"Specificity:  {specificity:.4f}")
    print(f"F1 Score:     {f1:.4f}")
    print(f"ROC-AUC:      {auc:.4f}")

    # Save model
    joblib.dump(
        model,
        config["filename"]
    )


# =========================
# 8. SAVE BENCHMARK
# =========================

with open(
    "models/benchmark.json",
    "w"
) as f:

    json.dump(
        benchmark,
        f,
        indent=4
    )


# =========================
# 9. SAVE FEATURE LIST
# =========================

with open(
    "models/selected_features.json",
    "w"
) as f:

    json.dump(
        SELECTED_FEATURES,
        f,
        indent=4
    )


print("\n" + "=" * 45)
print("FINAL TRAINING COMPLETE")
print("=" * 45)

print("\nSaved files:")

print("models/scaler.joblib")
print("models/logistic_regression.joblib")
print("models/random_forest.joblib")
print("models/rbf_svm.joblib")
print("models/benchmark.json")
print("models/selected_features.json")