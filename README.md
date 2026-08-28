Customer Support Intent Classification using XGBoost

An XGBoost-based NLP system that classifies customer support queries into six intents using TF-IDF text features and provides prediction confidence.

Overview

This project uses Natural Language Processing (NLP) and Machine Learning to automatically identify the intent behind customer support messages.

## Model Architecture

The system follows an NLP-based customer intent classification pipeline:

```text
Customer Query
      ↓
Text Preprocessing
      ↓
TF-IDF Vectorization
      ↓
Feature Representation
      ↓
XGBoost Classifier
      ↓
Intent Prediction
      ↓
Confidence Score

## Data Pipeline

The project follows the following data processing and machine learning pipeline:

```text
Customer Support Dataset
        ↓
Data Loading
        ↓
Data Validation & Cleaning
        ↓
Text and Intent Separation
        ↓
Label Encoding
        ↓
Train-Test Split
        ↓
TF-IDF Feature Extraction
        ↓
XGBoost Multiclass Classification
        ↓
Model Evaluation
        ↓
Model & Vectorizer Saving
        ↓
New Customer Query
        ↓
Intent Prediction + Confidence Score

Pipeline Steps
Data Loading – Load customer support queries from the CSV dataset.
Data Validation – Check the dataset structure and missing values.
Text Cleaning – Prepare customer messages for NLP processing.
Label Encoding – Convert the six intent categories into numerical labels.
Train-Test Split – Divide the dataset into training and testing sets.
TF-IDF Vectorization – Convert text messages into numerical feature vectors.
XGBoost Classification – Train the multiclass XGBoost model using the extracted features.
Model Evaluation – Evaluate performance using accuracy, precision, recall, F1-score, and a confusion matrix.
Model Saving – Save the trained XGBoost model and TF-IDF vectorizer for future predictions.
Prediction – Process new customer queries and return the predicted intent and confidence score.

The system classifies user queries into six categories:

order_status
return_request
refund_enquiry
delivery_delay
product_query
complaint
Technologies Used
Python
XGBoost
Scikit-learn
TF-IDF
Pandas
NumPy
Joblib
Project Structure
credit-risk-project/
│
├── app.py
├── xgboost_credit_risk_model.json
├── text_vectorizer.pkl
├── requirements.txt
├── .gitignore
└── README.md

How It Works
Customer Query
      ↓
Text Input
      ↓
TF-IDF Vectorization
      ↓
XGBoost Model
      ↓
Intent Classification
      ↓
Confidence Score

Example

Input:

Where is my order?


Output:

Intent: order_status
Confidence: 80.27%

Installation

Clone the repository:

git clone YOUR_GITHUB_REPOSITORY_URL
cd credit-risk-project


Install the required dependencies:

pip install -r requirements.txt

Run the Application

Run the application using:

python app.py


Enter a customer query when prompted:

Enter your message: Where is my order?


The application returns the predicted customer-support intent and confidence score.

Dataset

The training dataset contains customer support messages with their corresponding intent labels and confidence values.

The six intent categories are:

Order Status
Return Request
Refund Enquiry
Delivery Delay
Product Query
Complaint
Machine Learning Model

The project uses an XGBoost classification model trained on TF-IDF text features. The trained model is stored as:

xgboost_credit_risk_model.json


The fitted TF-IDF vectorizer is stored as:

text_vectorizer.pkl

Future Improvements
Add a Streamlit web interface
Add SHAP-based model explanations
Improve prediction accuracy with additional training data
Add automated responses for each intent
Deploy the application as a web service
Add comprehensive model evaluation metrics
Author

Purna Tejitha