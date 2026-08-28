Customer Support Intent Classification using XGBoost

An NLP-based machine learning system that automatically classifies customer support queries into six different intents using TF-IDF and an XGBoost classifier.

Project Overview

Customer support teams receive a large number of queries every day. This project uses Natural Language Processing (NLP) and Machine Learning to automatically identify the intent of a customer's message.

The system classifies customer queries into six categories:

order_status
return_request
refund_enquiry
delivery_delay
product_query
complaint

The model also provides a confidence score for the predicted intent.

Objectives
Automatically classify customer support queries.
Reduce manual effort in categorizing customer requests.
Use NLP techniques to convert text into machine-readable features.
Apply XGBoost for multiclass intent classification.
Provide prediction confidence for each classification.
Prepare the system for integration into a customer-support application.
Dataset

The training dataset contains 230 customer support samples with three columns:

Column	Description
text	Customer's support query
intent	Target intent/category
confidence	Confidence value associated with the sample
Intent Categories
order_status – Questions about order tracking or order status.
return_request – Requests related to returning a product.
refund_enquiry – Questions about refunds.
delivery_delay – Queries about delayed deliveries.
product_query – Questions about products.
complaint – Customer complaints or issues.
Model Architecture

The system follows an NLP-based classification architecture:

Customer Support Query
          ↓
Text Preprocessing
          ↓
TF-IDF Vectorization
          ↓
Feature Representation
          ↓
XGBoost Multiclass Classifier
          ↓
Intent Prediction
          ↓
Confidence Score

Architecture Description

The customer query is first processed and converted into numerical features using TF-IDF vectorization. These features are passed to an XGBoost multiclass classifier, which predicts one of the six predefined customer-support intents. The system then calculates the confidence of the prediction.

Data Pipeline
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
XGBoost Classification
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
Data Validation – Check dataset structure and missing values.
Text Cleaning – Prepare customer messages for NLP processing.
Label Encoding – Convert intent categories into numerical labels.
Train-Test Split – Divide the data into training and testing sets.
TF-IDF Feature Extraction – Convert text into numerical feature vectors.
XGBoost Classification – Use XGBoost for multiclass classification.
Model Evaluation – Evaluate classification performance.
Model Saving – Save the trained model and vectorizer.
Prediction – Process new queries and return the predicted intent and confidence.
Evaluation Metrics

The model will be evaluated using:

Accuracy – Overall percentage of correctly classified queries.
Precision – Measures the correctness of positive predictions for each intent.
Recall – Measures how many queries of each intent are correctly identified.
F1-Score – Harmonic mean of precision and recall.
Macro F1-Score – Gives equal importance to all six intent categories.
Confusion Matrix – Shows correct and incorrect classifications for each intent.
Example
Input
Where is my order?

Expected Output
Intent: order_status
Confidence: 80.27%


Another example:

I want to return the product I purchased.


Expected intent:

return_request

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
├── data_pipeline.py
├── xgboost_credit_risk_model.json
├── text_vectorizer.pkl
├── requirements.txt
├── .gitignore
└── README.md

Installation

Clone the repository:

git clone YOUR_GITHUB_REPOSITORY_URL
cd credit-risk-project


Install the required dependencies:

pip install -r requirements.txt

Running the Application

Run:

python app.py


Enter a customer query when prompted:

Enter your message: Where is my order?


The system returns the predicted intent and confidence score.

Model Files

The trained XGBoost model is stored as:

xgboost_credit_risk_model.json


The TF-IDF vectorizer is stored as:

text_vectorizer.pkl


These files allow the trained model to be reused without retraining.

Current Development Status
 Dataset collected and inspected
 Six intent categories identified
 XGBoost model obtained
 TF-IDF vectorizer obtained
 Model loading implemented
 Basic prediction tested
 Project README created
 Requirements file created
 Resolve final feature preprocessing mismatch
 Add label decoder for human-readable intent names
 Complete model evaluation
 Add Streamlit user interface
 Add SHAP-based explanations
 Deploy application
Future Improvements
Develop a Streamlit-based web interface.
Add SHAP explanations for model predictions.
Improve the training dataset with more customer queries.
Perform hyperparameter tuning for XGBoost.
Add automated responses for each intent.
Deploy the application as a web service.
Add continuous model evaluation and monitoring.
Conclusion

This project demonstrates an end-to-end NLP intent classification approach using TF-IDF and XGBoost. It provides a foundation for developing an intelligent customer-support system capable of automatically understanding and categorizing customer queries.
