import joblib
import pandas as pd

MODEL_PATH = "models/churn_model.pkl"


def load_model():

    model = joblib.load(MODEL_PATH)

    return model


def predict(customer_data):

    model = load_model()

    df = pd.DataFrame([customer_data])

    prediction = model.predict(df)[0]

    probability = model.predict_proba(df)[0][1]

    return prediction, probability


if __name__ == "__main__":

    sample_customer = {
        "gender": 1,
        "SeniorCitizen": 0,
        "Partner": 1,
        "Dependents": 0,
        "tenure": 12,
        "PhoneService": 1,
        "MultipleLines": 0,
        "InternetService": 1,
        "OnlineSecurity": 0,
        "OnlineBackup": 1,
        "DeviceProtection": 0,
        "TechSupport": 0,
        "StreamingTV": 1,
        "StreamingMovies": 1,
        "Contract": 0,
        "PaperlessBilling": 1,
        "PaymentMethod": 2,
        "MonthlyCharges": 70,
        "TotalCharges": 840
    }

    pred, prob = predict(sample_customer)

    print("Prediction:", pred)

    print("Churn probability:", prob)