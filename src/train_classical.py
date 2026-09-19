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


# =========================
# 1. LOAD DATA
# =========================

df = pd.read_csv("data/heart_cleaned.csv")

print("Dataset shape:", df.shape)


# =========================
# 2. FEATURES AND TARGET
# =========================

X = df.drop("target", axis=1)
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

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================
# 4. FEATURE SCALING
# =========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================
# 5. DEFINE MODELS
# =========================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "RBF SVM": SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    )
}


# =========================
# 6. TRAIN AND EVALUATE
# =========================

for name, model in models.items():

    print("\n" + "=" * 40)
    print(name)
    print("=" * 40)

    # Random Forest does not require scaling,
    # but using scaled data here keeps the pipeline comparison simple
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = model.predict(X_test_scaled)

    # Probabilities for ROC-AUC
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # Specificity
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    specificity = tn / (tn + fp)

    # Print results
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Precision:    {precision:.4f}")
    print(f"Sensitivity:  {sensitivity:.4f}")
    print(f"Specificity:  {specificity:.4f}")
    print(f"F1 Score:     {f1:.4f}")
    print(f"ROC-AUC:      {auc:.4f}")