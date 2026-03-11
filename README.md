# Customer Churn Prediction

## Overview

Customer churn is a major challenge for subscription-based businesses such as telecom companies and SaaS platforms. Predicting which customers are likely to leave helps businesses take preventive actions and improve retention.

This project builds a machine learning model that predicts whether a telecom customer will churn based on account information, services used, and billing details.

The project includes data analysis, preprocessing, model training, evaluation, and a simple web app for predictions.

---

## Dataset

Telco Customer Churn dataset.

Each row represents a customer and includes features such as:

* tenure
* contract type
* internet service
* monthly charges
* payment method
* service subscriptions

Target variable:

**Churn**

* 1 → Customer leaves
* 0 → Customer stays

---

## Project Structure

```
customer-churn-prediction
│
├── data
│   └── telco_churn.csv
│
├── models
│   └── churn_model.pkl
│
├── notebooks
│   └── churn_analysis.ipynb
│
├── src
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
│
├── app
│   └── app.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Model

The model used is a **Random Forest Classifier**.

Performance on the test set:

* Accuracy: ~79%
* ROC-AUC: ~0.81

The model helps identify customers at risk of leaving so businesses can take retention actions.

---

## Installation

Clone the repository:

```
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
cd customer-churn-prediction
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Train the Model

Run:

```
python src/train.py
```

This will preprocess the data, train the model, and save it in the `models` folder.

---

## Run the App

Start the Streamlit app:

```
streamlit run app/app.py
```

Then open the local URL shown in the terminal to use the churn prediction interface.

---

## Future Improvements

* Hyperparameter tuning
* Feature importance visualization
* Model explainability
* Deployment to a cloud platform

---

## Author

Data science project demonstrating an end-to-end machine learning workflow including data preprocessing, model training, evaluation, and deployment.
