# Customer Churn Prediction

## Overview

Customer churn is a major challenge for subscription-based businesses such as telecom providers, SaaS platforms, and financial services. Retaining customers is often significantly cheaper than acquiring new ones.

This project builds a machine learning model that predicts whether a customer is likely to churn based on demographic information, service usage, and billing data.

The project includes:

- Exploratory data analysis
- Data preprocessing pipeline
- Machine learning model training
- Model evaluation
- Interactive prediction app using Streamlit

---

## Dataset

Telco Customer Churn dataset.

Each row represents a telecom customer. The goal is to predict the **Churn** column.

Key features include:

- tenure
- contract type
- internet service
- monthly charges
- payment method
- service subscriptions

Target variable:

Churn  
1 = customer leaves  
0 = customer stays

---

## Project Structure
customer-churn-prediction
│
├── data
│ └── telco_churn.csv
│
├── models
│ └── churn_model.pkl
│
├── notebooks
│ └── churn_analysis.ipynb
│
├── src
│ ├── preprocess.py
│ ├── train.py
│ └── predict.py
│
├── app
│ └── app.py
│
├── requirements.txt
└── README.md

---

## Workflow

1. Load dataset
2. Clean data and handle missing values
3. Encode categorical variables
4. Train/test split
5. Train machine learning models
6. Evaluate model performance
7. Save trained model
8. Deploy prediction interface with Streamlit

---

## Model

Random Forest Classifier

Evaluation results:

Accuracy: ~79%  
ROC-AUC: ~0.81

Example classification metrics:

- Precision (churn): ~0.63
- Recall (churn): ~0.48

The model identifies customers at risk of leaving, allowing businesses to take proactive retention actions.

---

## Installation

Clone the repository and install dependencies.
