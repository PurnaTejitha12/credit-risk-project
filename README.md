# Customer Support Intent Classification using XGBoost

An NLP-based machine learning application that classifies customer support messages into six predefined intents using TF-IDF and XGBoost.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Supported Intents](#supported-intents)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [API](#api)
- [Frontend](#frontend)
- [Testing](#testing)
- [Load Testing](#load-testing)
- [Model Files](#model-files)
- [Project Checkpoints](#project-checkpoints)
- [Author](#author)

## Overview

This project automatically identifies the intent of a customer support message.

The application uses a saved TF-IDF vectorizer to convert text into numerical features and an XGBoost classifier to predict the customer's intent.

The prediction is served through a FastAPI REST API and displayed through a web-based frontend.

## Features

- Customer support intent classification.
- TF-IDF text vectorization.
- XGBoost multiclass classification.
- Prediction confidence score.
- FastAPI REST API.
- HTML, CSS, and JavaScript frontend.
- Pytest integration testing.
- Locust load testing.

## Supported Intents

| Intent | Description |
|---|---|
| `order_status` | Queries about order status or tracking |
| `return_request` | Requests to return a product |
| `refund_enquiry` | Questions about refunds |
| `delivery_delay` | Queries about delayed deliveries |
| `product_query` | Questions about products |
| `complaint` | Customer complaints or issues |

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Application and machine learning development |
| XGBoost | Intent classification |
| Scikit-learn | TF-IDF vectorization |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Joblib | Loading the saved vectorizer |
| FastAPI | REST API |
| Pytest | API testing |
| Locust | Load testing |
| HTML | Frontend structure |
| CSS | Frontend styling |
| JavaScript | Frontend interaction |

## Project Structure

```text
credit-risk-project/
├── app.py
├── data_pipeline.py
├── test_api.py
├── locustfile.py
├── index.html
├── xgboost_credit_risk_model.json
├── text_vectorizer.pkl
├── requirements.txt
├── .gitignore
└── README.md
