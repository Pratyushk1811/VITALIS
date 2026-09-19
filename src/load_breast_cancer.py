import pandas as pd
from data.breast_cancer_columns import COLUMNS


def load_breast_cancer():
    df = pd.read_csv(
        "data/bcancer.csv",
        header=None,
        names=COLUMNS
    )

    # Remove patient ID
    df = df.drop(columns=["id"])

    # Encode diagnosis: Malignant = 1, Benign = 0
    df["diagnosis"] = df["diagnosis"].map({
        "M": 1,
        "B": 0
    })

    return df


if __name__ == "__main__":
    df = load_breast_cancer()

    print("Shape:", df.shape)
    print("\nTarget counts:")
    print(df["diagnosis"].value_counts())

    print("\nMissing values:")
    print(df.isnull().sum().sum())

    print("\nFirst 2 rows:")
    print(df.head(2))