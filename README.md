# Customer Churn Prediction System

A Machine Learning project that predicts whether a telecom customer is likely to churn based on customer information, service usage, contract details, and billing information.

## Project Overview

Customer churn occurs when a customer stops using a company's services. Predicting churn can help businesses identify customers who may leave and take appropriate retention actions.

This project uses **Logistic Regression** to classify customers into:

- **No Churn** – Customer is likely to stay
- **Churn** – Customer is likely to leave

The trained model is integrated with a **Streamlit web application** where users can enter customer details and receive a churn prediction with probability.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit

## Project Structure

```text
customer-churn-project/
│
├── app/
│   └── app.py
│
├── data/
│   └── customer_churn.csv
│
├── database/
│
├── models/
│   └── logistic_regression_model.pkl
│
├── notebooks/
│
├── src/
│   ├── data_preprocessing.py
│   ├── eda.py
│   └── train_model.py
│
├── .gitignore
├── README.md
└── requirements.txt