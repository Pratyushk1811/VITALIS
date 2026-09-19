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

X = df.drop("target", axis=1)
y = df["target"]


# =========================
# 2. TRAIN-TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# 3. FEATURE SETS
# =========================

feature_sets = {

    "Top 6": [
        "cp",
        "thal",
        "thalach",
        "oldpeak",
        "ca",
        "age"
    ],

    "Top 8": [
        "cp",
        "thal",
        "thalach",
        "oldpeak",
        "ca",
        "age",
        "chol",
        "trestbps"
    ]
}


# =========================
# 4. EVALUATION FUNCTION
# =========================

def evaluate_model(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    specificity = tn / (tn + fp)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F1": f1,
        "ROC-AUC": auc
    }


# =========================
# 5. TEST EACH FEATURE SET
# =========================

for feature_set_name, features in feature_sets.items():

    print("\n" + "#" * 55)
    print(f"FEATURE SET: {feature_set_name}")
    print("Features:", features)
    print("#" * 55)

    # Select only these features
    X_train_selected = X_train[features]
    X_test_selected = X_test[features]

    # Scale data for Logistic Regression and SVM
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)

    models = {

        "Logistic Regression":
            LogisticRegression(max_iter=1000),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            ),

        "RBF SVM":
            SVC(
                kernel="rbf",
                probability=True,
                random_state=42
            )
    }

    for name, model in models.items():

        print("\n" + "-" * 40)
        print(name)
        print("-" * 40)

        # Random Forest uses unscaled data
        if name == "Random Forest":

            results = evaluate_model(
                model,
                X_train_selected,
                X_test_selected,
                y_train,
                y_test
            )

        else:

            results = evaluate_model(
                model,
                X_train_scaled,
                X_test_scaled,
                y_train,
                y_test
            )

        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")