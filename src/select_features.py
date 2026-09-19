import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# =========================
# 1. LOAD DATASET
# =========================

df = pd.read_csv("data/heart_cleaned.csv")

print("Dataset shape:", df.shape)


# =========================
# 2. SEPARATE FEATURES
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


# =========================
# 4. TRAIN RANDOM FOREST
#    FOR FEATURE IMPORTANCE
# =========================

feature_model = RandomForestClassifier(
    n_estimators=500,
    random_state=42
)

feature_model.fit(X_train, y_train)


# =========================
# 5. FEATURE IMPORTANCE
# =========================

importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": feature_model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

print("\n===== FEATURE IMPORTANCE RANKING =====\n")

print(importance_df.to_string(index=False))