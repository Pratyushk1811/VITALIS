import pandas as pd

# Column names for the UCI Cleveland Heart Disease dataset
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

# Load the local dataset
df = pd.read_csv(
    "data/heart.csv",
    header=None,
    names=columns
)

print("=== FIRST 5 ROWS ===")
print(df.head())

print("\n=== DATASET SHAPE ===")
print(df.shape)

print("\n=== DATA TYPES ===")
print(df.dtypes)

print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

print("\n=== TARGET DISTRIBUTION ===")
print(df["target"].value_counts())
