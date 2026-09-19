import pandas as pd
import numpy as np

columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target"
]

# Load the raw dataset
df = pd.read_csv(
    "data/heart.csv",
    header=None,
    names=columns
)

print("Original dataset shape:", df.shape)

# Replace ? with actual missing values
df = df.replace("?", np.nan)

# Convert all columns to numeric values
df = df.apply(pd.to_numeric)

print("\nMissing values after conversion:")
print(df.isnull().sum())

# Remove rows containing missing values
df = df.dropna()

print("\nDataset shape after removing missing values:", df.shape)

# Convert the target into binary classification
df["target"] = (df["target"] > 0).astype(int)

print("\nBinary target distribution:")
print(df["target"].value_counts())

# Save cleaned dataset
df.to_csv("data/heart_cleaned.csv", index=False)

print("\nCleaned dataset saved as: data/heart_cleaned.csv")