"""
1_data_preprocessing.py
------------------------
Loads and preprocesses the UCI Cleveland Heart Disease (CVD) dataset.
All other scripts import `load_data()` from this file so every model
is trained/tested on an identical split.

Dataset: 14 attributes, target column 'num' (0 = no disease, 1-4 = disease
severity). We binarize target: 0 -> 0 (no CVD), 1-4 -> 1 (CVD present).

Install requirements:
    pip install pandas numpy scikit-learn
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
]


def load_raw_dataframe(local_path: str = None) -> pd.DataFrame:
    """
    Load the raw UCI CVD dataset either from a local CSV/data file
    (recommended, since the UCI server can be unreliable) or by
    downloading it directly.

    Parameters
    ----------
    local_path : str, optional
        Path to a local copy of 'processed.cleveland.data' or similar CSV.
        If None, attempts to download from the UCI repository.
    """
    source = local_path if local_path else UCI_URL
    df = pd.read_csv(source, names=COLUMN_NAMES, na_values="?")
    return df


def load_data(local_path: str = None, test_size: float = 0.2, random_state: int = 42):
    """
    Returns train/test splits, already imputed and standardized.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
    feature_names : list[str]
    scaler : fitted StandardScaler (useful for inference on new patients)
    """
    df = load_raw_dataframe(local_path)

    # Binarize target: 0 = healthy, 1 = any level of CVD
    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])

    feature_names = [c for c in df.columns if c != "target"]
    X = df[feature_names].values.astype(float)
    y = df["target"].values.astype(int)

    # Handle missing values ('?' -> NaN -> median impute)
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, feature_names, scaler


if __name__ == "__main__":
    # Quick sanity check. Point local_path at your downloaded copy of the
    # dataset, e.g. load_data(local_path="processed.cleveland.data")
    X_train, X_test, y_train, y_test, features, scaler = load_data()
    print("Features:", features)
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
    print("Class balance (train):", np.bincount(y_train))
