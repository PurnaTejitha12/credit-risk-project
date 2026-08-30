Customer Support Intent Classification using XGBoost

An NLP-based machine learning system that automatically classifies customer support queries into six different intents using TF-IDF and an XGBoost classifier.

📌 Project Overview

Customer support teams receive a large number of queries every day. Manually categorizing these queries can be time-consuming and inefficient.

This project uses Natural Language Processing (NLP) and Machine Learning to automatically identify the intent of a customer's message.

The system classifies customer queries into six categories:

order_status
return_request
refund_enquiry
delivery_delay
product_query
complaint

The system also provides a confidence score for the predicted intent.

🎯 Objectives
Automatically classify customer support queries.
Reduce manual effort in categorizing customer requests.
Convert text data into numerical features using TF-IDF.
Use XGBoost for multiclass intent classification.
Provide prediction confidence for each query.
Build a foundation for an intelligent customer-support application.
Integrate the trained model into an API-based serving layer.
Test the API and evaluate its performance under concurrent load.
📊 Dataset

The project uses a customer-support dataset containing 230 samples and three columns.

Column	Description
text	Customer support query
intent	Target intent/category
confidence	Confidence value associated with the sample
Intent Categories
Intent	Description
order_status	Questions about order tracking or status
return_request	Requests related to returning a product
refund_enquiry	Questions about refunds
delivery_delay	Queries about delayed deliveries
product_query	Questions about products
complaint	Customer complaints or issues
🏗️ Model Architecture

The system follows an NLP-based classification pipeline:

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

Architecture Description
The user enters a customer-support query.
The text is preprocessed for NLP analysis.
TF-IDF converts the text into numerical feature vectors.
The feature vectors are passed to the XGBoost classifier.
The classifier predicts one of the six customer-support intents.
The system returns the predicted intent and confidence score.
🔄 Data Pipeline
Customer Support Dataset
          ↓
Data Loading
          ↓
Data Validation and Cleaning
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
Model and Vectorizer Saving
          ↓
New Customer Query
          ↓
Intent Prediction
          ↓
Confidence Score

Pipeline Steps

Data Loading
Load customer-support queries from the dataset.

Data Validation
Check the dataset structure and identify missing or invalid values.

Text Cleaning
Prepare customer messages for NLP processing.

Label Encoding
Convert intent categories into numerical labels for model training.

Train-Test Split
Divide the dataset into training and testing sets.

TF-IDF Feature Extraction
Convert customer-support text into numerical feature vectors.

XGBoost Classification
Train the XGBoost multiclass classification model.

Model Evaluation
Evaluate the model using standard classification metrics.

Model Saving
Save the trained model and TF-IDF vectorizer for future predictions.

Prediction
Process new customer queries and return the predicted intent and confidence score.

📈 Evaluation Metrics

The model is evaluated using the following metrics:

Metric	Purpose
Accuracy	Measures the overall percentage of correct predictions
Precision	Measures the correctness of predictions for each intent
Recall	Measures how many samples of each intent are correctly identified
F1-Score	Combines precision and recall
Macro F1-Score	Gives equal importance to all six intent classes
Confusion Matrix	Shows correct and incorrect predictions for each class

Macro F1-Score is particularly useful for this multiclass classification problem because it gives equal importance to all six intent categories.

💻 Technologies Used
Python
XGBoost
Scikit-learn
TF-IDF
Pandas
NumPy
Joblib
FastAPI
Pytest
Locust
MLflow
Git
GitHub
📁 Project Structure
credit-risk-project/
│
├── app.py
├── data_pipeline.py
├── test_api.py
├── locustfile.py
├── xgboost_credit_risk_model.json
├── text_vectorizer.pkl
├── requirements.txt
├── .gitignore
└── README.md

⚙️ Installation
1. Clone the Repository
git clone YOUR_GITHUB_REPOSITORY_URL

2. Navigate to the Project Directory
cd credit-risk-project

3. Install Dependencies
pip install -r requirements.txt

🚀 Running the Application

The model is served through a FastAPI application.

Start the API using:

uvicorn app:app --host 127.0.0.1 --port 8000


The API provides:

GET  /
POST /predict


The /predict endpoint accepts a customer-support query and returns the predicted intent and confidence score.

Example Request
{
  "text": "Where is my order?"
}

Example Response
{
  "prediction": "order_status",
  "confidence": 80.27
}

🧪 Integration Testing

Integration testing is performed using Pytest.

Run the tests using:

pytest test_api.py -v


The tests verify:

API availability.
Prediction endpoint functionality.
Successful response status.
Presence of prediction results.
Presence of confidence scores.
⚡ Load Testing

Load testing is performed using Locust to evaluate API performance under concurrent requests.

The required test uses:

50 concurrent users
60-second test duration

Run the load test using:

locust -f locustfile.py --headless -u 50 -r 50 -t 60s --host http://127.0.0.1:8000

Performance Results
Metric	Result
Concurrent Users	50
Test Duration	60 seconds
Average Latency	Add Locust result
Throughput	Add Locust result
Failed Requests	Add Locust result

Replace the result placeholders above with the actual values generated by Locust.

🤖 Model Information

The project uses an XGBoost multiclass classification model.

Trained Model
xgboost_credit_risk_model.json

TF-IDF Vectorizer
text_vectorizer.pkl


These files allow the trained model and text preprocessing components to be reused for prediction.

📌 Project Checkpoints
Checkpoint 1 — Research and Architecture Design

The project architecture, data pipeline, evaluation metrics, and repository structure were defined.

Model

XGBoost multiclass classifier

NLP Technique

TF-IDF text vectorization

Number of Classes

6 customer-support intents

Evaluation Metrics
Accuracy
Precision
Recall
F1-Score
Macro F1-Score
Confusion Matrix
Checkpoint 2 — Data Ingestion, Preprocessing and Baseline Model
Objective

The goal of Checkpoint 2 was to build the complete data-ingestion and preprocessing pipeline, train a baseline machine-learning model, establish an evaluation benchmark, and document the baseline results.

Work Completed
Loaded the customer-support dataset.
Cleaned the data and handled missing values.
Removed duplicate and empty records.
Prepared the required features.
Applied preprocessing using a Scikit-learn pipeline.
Split the dataset into training and testing sets.
Trained a baseline Logistic Regression model.
Evaluated the baseline model using:
Accuracy
Precision
Recall
F1-Score
Saved the processed dataset and baseline model.
Created a baseline evaluation report.
Baseline Evaluation

The baseline model provides the initial benchmark for the project. Future models can be compared against this benchmark to determine whether model improvements are effective.

Checkpoint 3 — Model Training, Hyperparameter Tuning and MLflow
Objective

The goal of Checkpoint 3 was to train the primary classification model using multiple hyperparameter configurations, track experiments using MLflow, compare the results, and select the best-performing configuration.

Work Completed
Used Logistic Regression as the primary model for the checkpoint experiment.
Tested three different hyperparameter configurations.
Changed the regularization parameter C.
Tested different class-weight settings.
Used the same train-test split for fair comparison.
Evaluated each configuration using:
Accuracy
Precision
Recall
F1-Score
Logged model parameters and evaluation metrics using MLflow.
Compared experimental results.
Selected the configuration with the best F1-Score.
Saved the best-performing model.
Saved the experiment comparison results.
Hyperparameter Configurations
Configuration	C	Solver	Class Weight
Configuration 1	0.1	liblinear	None
Configuration 2	1.0	liblinear	balanced
Configuration 3	10.0	liblinear	None
Checkpoint 3 Files
src/
└── train_mlflow.py

models/
└── best_model.joblib

MLflow Experiment Tracking

MLflow was used to record each model configuration and its evaluation metrics.

The experiments allow different configurations to be compared and provide evidence for selecting the best-performing configuration.

The best configuration was selected based on the highest weighted F1-Score obtained during evaluation.

Reproducibility

The experiments use:

Random State = 42
Test Size = 20%


These settings help make the experiments reproducible.

Conclusion

Checkpoint 2 established the data-processing pipeline and baseline benchmark.

Checkpoint 3 improved the modelling workflow by testing multiple hyperparameter configurations and tracking experiments with MLflow.

The best-performing configuration was selected based on the evaluation results and saved for future use.

Checkpoint 4 — Integration and Testing
Objective

The goal of Checkpoint 4 was to integrate the trained model into a serving layer, create integration tests, and evaluate API performance under concurrent load.

Integration

The trained model was integrated into a FastAPI serving layer with a /predict endpoint for making predictions.

Integration Testing

Pytest was used to test the API endpoints and verify that valid prediction responses are returned successfully.

Load Testing

Load testing was performed using Locust with 50 concurrent users for 60 seconds.

Performance Results
Metric	Result
Concurrent Users	50
Test Duration	60 seconds
Average Latency	Add actual result
Throughput	Add actual result
Failed Requests	Add actual result
Checkpoint 4 Files
app.py – FastAPI model-serving application
test_api.py – API integration tests
locustfile.py – Load-testing configuration
Checkpoint 4 Commands

Start the API:

uvicorn app:app --host 127.0.0.1 --port 8000


Run integration tests:

pytest test_api.py -v


Run load testing:

locust -f locustfile.py --headless -u 50 -r 50 -t 60s --host http://127.0.0.1:8000

📌 Current Project Status
 Dataset collected and inspected
 Six intent categories identified
 XGBoost model obtained
 TF-IDF vectorizer obtained
 Model loading implemented
 Basic prediction tested
 Requirements file created
 README documentation created
 Model integrated with FastAPI
 Integration tests implemented
 Load testing performed with 50 concurrent users
 Latency and throughput measured
 Resolve final feature preprocessing mismatch
 Complete label decoding
 Complete detailed model evaluation
 Add Streamlit web interface
 Add SHAP model explanations
 Test all six intent categories
 Deploy the application
🚀 Future Enhancements
Develop a Streamlit-based web interface.
Add SHAP-based model explanations.
Increase the size and diversity of the training dataset.
Perform XGBoost hyperparameter tuning.
Add automated responses for each intent.
Improve classification accuracy.
Deploy the application as a web service.
Add continuous model monitoring.
🔮 Expected Application Workflow
User
  ↓
Enter Customer Query
  ↓
Text Processing
  ↓
TF-IDF Vectorizer
  ↓
XGBoost Classifier
  ↓
Predicted Intent
  ↓
Confidence Score
  ↓
Customer Support Action

👩‍💻 Author

Purna Tejitha