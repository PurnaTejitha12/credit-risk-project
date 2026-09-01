# FinWise Lending - Credit Risk Scoring System

An end-to-end explainable AI/ML credit risk scoring system built using XGBoost and SHAP. The system predicts loan risk, assigns applicants to Low, Medium, or High risk bands, and explains individual predictions using SHAP.

---

## Project Overview

FinWise Lending is a FinTech company that requires an intelligent and explainable credit risk scoring system for evaluating loan applications.

This project builds a machine learning solution that:

- Predicts credit risk using XGBoost
- Generates a risk probability score
- Assigns applicants to Low, Medium, or High risk bands
- Uses SHAP for model explainability
- Displays the top 3 factors influencing each prediction
- Provides predictions through a REST API
- Provides a React-based frontend
- Performs an age-based fairness audit
- Evaluates model performance using ROC-AUC and Gini
- Tests API performance under concurrent requests

---

## Objectives

The main objectives of this project are:

1. Build a binary credit risk classification model.
2. Achieve a Gini coefficient of at least 0.45 on the test set.
3. Provide explainable predictions using SHAP.
4. Generate a SHAP waterfall chart for individual applications.
5. Classify applicants into Low, Medium, and High risk bands.
6. Provide a REST API for model predictions.
7. Build a React frontend for the prediction system.
8. Perform fairness analysis across age groups.
9. Evaluate API performance with 50 concurrent requests.

---

## Architecture

```text
                    Loan Application Data
                             |
                             v
                    Data Preprocessing
                             |
                             v
                       XGBoost Model
                             |
              +--------------+--------------+
              |                             |
              v                             v
        Model Evaluation                SHAP
        ROC-AUC / Gini              Explainability
              |                             |
              +--------------+--------------+
                             |
                             v
                       FastAPI REST API
                             |
                             v
                       React Frontend
                             |
              +--------------+--------------+
              |              |              |
              v              v              v
          Risk Score      Risk Band      SHAP Factors
