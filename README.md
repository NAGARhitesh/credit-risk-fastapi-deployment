# Credit Risk Prediction

An end-to-end machine learning project for predicting whether a borrower is likely to default based on demographic, financial, loan, and credit-history features.

## Project Overview

This project solves a binary classification problem for credit risk assessment. It includes:

- Data preprocessing using `Pipeline` and `ColumnTransformer`
- Missing value imputation
- Categorical encoding and numerical scaling
- Multiple model comparison
- Random Forest hyperparameter tuning using `GridSearchCV`
- Evaluation using Accuracy, Precision, Recall, F1-score, and ROC-AUC
- Threshold-based business-oriented prediction analysis
- FastAPI deployment for real-time inference

## Features Used

- `person_age`
- `person_income`
- `person_home_ownership`
- `person_emp_length`
- `loan_intent`
- `loan_grade`
- `loan_amnt`
- `loan_int_rate`
- `loan_percent_income`
- `cb_person_default_on_file`
- `cb_person_cred_hist_length`

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- FastAPI
- Joblib

## Models Compared

- Logistic Regression
- SVM
- Decision Tree
- Random Forest
- KNN (optional depending on environment support)

## Key Improvements Over Basic Notebook Version

- Replaced manual preprocessing with `Pipeline`
- Added `ColumnTransformer` to prevent data leakage
- Used imputation instead of dropping rows
- Added model comparison and hyperparameter tuning
- Evaluated with Recall, Precision, F1, ROC-AUC
- Added threshold tuning for business-sensitive classification
- Saved trained artifacts for deployment

## How to Run

### 1. Train the model
```bash
python train_credit_risk.py