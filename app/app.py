import streamlit as st
import pandas as pd
import joblib

MODEL_PATH = "models/churn_model.pkl"

model = joblib.load(MODEL_PATH)

st.title("Customer Churn Prediction")

st.write("Enter customer information to estimate churn risk.")

gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])

tenure = st.slider("Tenure (months)", 0, 72)

PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])

PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0)


# ----- Encoding mappings -----

gender = 1 if gender == "Male" else 0
Partner = 1 if Partner == "Yes" else 0
Dependents = 1 if Dependents == "Yes" else 0
PhoneService = 1 if PhoneService == "Yes" else 0
PaperlessBilling = 1 if PaperlessBilling == "Yes" else 0

multiple_lines_map = {
    "No": 0,
    "Yes": 1,
    "No phone service": 2
}

internet_map = {
    "DSL": 0,
    "Fiber optic": 1,
    "No": 2
}

yes_no_map = {
    "No": 0,
    "Yes": 1,
    "No internet service": 2
}

contract_map = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}

payment_map = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer (automatic)": 2,
    "Credit card (automatic)": 3
}

MultipleLines = multiple_lines_map[MultipleLines]
InternetService = internet_map[InternetService]

OnlineSecurity = yes_no_map[OnlineSecurity]
OnlineBackup = yes_no_map[OnlineBackup]
DeviceProtection = yes_no_map[DeviceProtection]
TechSupport = yes_no_map[TechSupport]
StreamingTV = yes_no_map[StreamingTV]
StreamingMovies = yes_no_map[StreamingMovies]

Contract = contract_map[Contract]
PaymentMethod = payment_map[PaymentMethod]


if st.button("Predict Churn"):

    input_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"Customer likely to churn. Probability: {probability:.2f}")
    else:
        st.success(f"Customer likely to stay. Churn probability: {probability:.2f}")