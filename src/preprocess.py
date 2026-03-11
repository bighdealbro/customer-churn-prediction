import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data(path):

    df = pd.read_csv(path)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df = df.dropna()

    df = df.drop("customerID", axis=1)

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def encode_features(df):

    df_encoded = df.copy()

    for col in df_encoded.columns:

        if df_encoded[col].dtype == "object":

            encoder = LabelEncoder()

            df_encoded[col] = encoder.fit_transform(df_encoded[col])

    return df_encoded


def preprocess_data(path):

    df = load_data(path)

    df_encoded = encode_features(df)

    X = df_encoded.drop("Churn", axis=1)

    y = df_encoded["Churn"]

    return X, y