import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_and_clean(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    # Drop customerID — not a feature
    df.drop(columns=["customerID"], inplace=True)

    # TotalCharges is a string — convert to numeric, coerce blanks to NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill the ~11 missing TotalCharges with MonthlyCharges (new customers, tenure=0)
    df["TotalCharges"].fillna(df["MonthlyCharges"], inplace=True)

    # Encode target: Yes -> 1, No -> 0
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Useful ratio features
    df["charges_per_month"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["is_new_customer"] = (df["tenure"] <= 3).astype(int)

    # Count how many add-on services the customer has
    addons = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
              "TechSupport", "StreamingTV", "StreamingMovies"]
    df["num_addons"] = df[addons].apply(
        lambda row: sum(v == "Yes" for v in row), axis=1
    )

    return df


def encode_features(df: pd.DataFrame):
    df = df.copy()

    # Binary Yes/No columns
    binary_cols = [
        "Partner", "Dependents", "PhoneService",
        "PaperlessBilling", "MultipleLines",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0, "No phone service": 0, "No internet service": 0})

    # Encode remaining categoricals
    cat_cols = ["gender", "InternetService", "Contract", "PaymentMethod"]
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    return df


def get_train_test(filepath: str):
    df = load_and_clean(filepath)
    df = engineer_features(df)
    df = encode_features(df)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, X.columns.tolist()


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, features = get_train_test("data/telco_churn.csv")
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    print(f"Churn rate in train: {y_train.mean():.2%}")
    print(f"Features: {features}")
