from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

import joblib

from preprocess import preprocess_data


DATA_PATH = "data/telco_churn.csv"
MODEL_PATH = "models/churn_model.pkl"

def train():

    X, y = preprocess_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    prob = model.predict_proba(X_test)[:, 1]

    print("\nModel Evaluation\n")

    print(classification_report(y_test, pred))

    print("ROC AUC:", roc_auc_score(y_test, prob))

    joblib.dump(model, MODEL_PATH)

    print("\nModel saved at:", MODEL_PATH)


if __name__ == "__main__":
    train()