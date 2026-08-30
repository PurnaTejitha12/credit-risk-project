# Customer Support Intent Classification using XGBoost

An NLP-based machine learning system that automatically classifies customer support queries into six different intents using TF-IDF and an XGBoost classifier.

---

## 📌 Project Overview

Customer support teams receive a large number of queries every day. Manually categorizing these queries can be time-consuming.

This project uses Natural Language Processing (NLP) and Machine Learning to automatically identify the intent of a customer's message.

The system classifies customer queries into six categories:

- `order_status`
- `return_request`
- `refund_enquiry`
- `delivery_delay`
- `product_query`
- `complaint`

The system also provides a confidence score for the predicted intent.

---

## 🎯 Objectives

- Automatically classify customer support queries.
- Reduce manual effort in categorizing customer requests.
- Convert text data into numerical features using TF-IDF.
- Use XGBoost for multiclass intent classification.
- Provide prediction confidence for each query.
- Create a foundation for an intelligent customer-support application.

---

## 📊 Dataset

The project uses a customer-support dataset containing 230 samples and three columns.

| Column | Description |
|--------|-------------|
| `text` | Customer support query |
| `intent` | Target intent/category |
| `confidence` | Confidence value associated with the sample |

### Intent Categories

| Intent | Description |
|--------|-------------|
| `order_status` | Questions about order tracking or status |
| `return_request` | Requests related to returning a product |
| `refund_enquiry` | Questions about refunds |
| `delivery_delay` | Queries about delayed deliveries |
| `product_query` | Questions about products |
| `complaint` | Customer complaints or issues |

---

## 🏗️ Model Architecture

The system follows an NLP-based classification pipeline:

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
=======
📌 Project Overview

Customer support teams receive a large number of queries every day. Manually categorizing these queries can be time-consuming.

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
Create a foundation for an intelligent customer-support application.
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

The user enters a customer support query.

The text is preprocessed for NLP analysis.

TF-IDF converts the text into numerical feature vectors.

The feature vectors are passed to the XGBoost classifier.

The classifier predicts one of the six customer-support intents.

The system provides the predicted intent and confidence score.

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

Data Loading – Load customer support queries from the CSV dataset.

Data Validation – Check the dataset structure and missing values.

Text Cleaning – Prepare customer messages for NLP processing.

Label Encoding – Convert intent categories into numerical labels.

Train-Test Split – Divide the dataset into training and testing sets.

TF-IDF Feature Extraction – Convert text into numerical feature vectors.

XGBoost Classification – Use XGBoost for multiclass classification.

Model Evaluation – Evaluate the classification performance.

Model Saving – Save the trained model and vectorizer for future predictions.

Prediction – Process new customer queries and return the predicted intent and confidence score.

📈 Evaluation Metrics

The model will be evaluated using the following metrics:

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
📁 Project Structure
credit-risk-project/
│
├── app.py
├── data_pipeline.py
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

▶️ Running the Application

Run the application using:

python app.py


The application will ask for a customer query:

Enter your message: Where is my order?


The model will process the query and return the predicted intent and confidence score.

🧪 Example
Input
Where is my order?

Output
Intent: order_status
Confidence: 80.27%

Another Example
Input:
I want to return the product I purchased.

Output:
Intent: return_request

🤖 Model Information

The project uses an XGBoost multiclass classification model.

The trained model is stored in:

xgboost_credit_risk_model.json


The TF-IDF vectorizer is stored in:

text_vectorizer.pkl


These files allow the trained model and text preprocessing components to be reused for prediction.

📌 Current Project Status
 
 Dataset collected and inspected
 
 Six intent categories identified
 
 XGBoost model obtained
 
 TF-IDF vectorizer obtained
 
 Model loading implemented
 
 Basic prediction tested
 
 Requirements file created
 
 README documentation created
 
 Resolve final feature preprocessing mismatch
 
 Complete label decoding
 
 Complete model evaluation
 
 Add Streamlit web interface
 
 Add SHAP model explanations
 
 Test all six intent categories
 
 Deploy the application
🚀 Future Enhancements
D
evelop a Streamlit-based web interface.

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

📝 Checkpoint 1
Research and Architecture Design

The project architecture, data pipeline, evaluation metrics, and project repository structure have been defined.

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
The goal of Checkpoint 2 was to build the complete data ingestion and preprocessing pipeline, train a baseline machine learning model, establish an evaluation benchmark, and document the baseline results.

Work Completed

Loaded the credit-risk dataset.

Cleaned the data and handled missing values.

Removed duplicate and empty records.

Prepared numerical and categorical features.

Applied preprocessing using a Scikit-learn pipeline.

Split the dataset into training and testing sets.

Trained a baseline Logistic Regression model.

Evaluated the baseline model using:

Accuracy

Precision

Recall

F1-score

Saved the processed dataset and baseline model.

Created a baseline evaluation report.

Baseline Evaluation
The baseline model provides the initial benchmark for the project. Future models can be compared against these baseline results to determine whether model improvements are effective.

Checkpoint 3 — Model Training, Hyperparameter Tuning and MLflow

Objective
The goal of Checkpoint 3 was to train the primary credit-risk model using multiple hyperparameter configurations, track the experiments using MLflow, compare the results, and select the best-performing configuration.


Work Completed

Used Logistic Regression as the primary model.

Tested three different hyperparameter configurations.

Changed the regularization parameter C.

Tested different class-weight settings.

Used the same train/test split for fair comparison.

Evaluated each configuration using:

Accuracy

Precision

Recall

F1-score

Logged model parameters and evaluation metrics using MLflow.

Compared the experimental results.

Selected the configuration with the best F1-score.

Saved the best-performing model.

Saved the experiment comparison results.

Saved the best model configuration.

Hyperparameter Configurations

Configuration 1

C = 0.1

solver = liblinear

class_weight = None


Configuration 2

C = 1.0

solver = liblinear

class_weight = balanced


Configuration 3

C = 10.0

solver = liblinear

class_weight = None


Checkpoint 3 Files

src/

└── train_mlflow.py

models/

└── best_model.joblib


MLflow Experiment Tracking

MLflow was used to record each model configuration and its evaluation metrics. The experiments allow the different configurations to be compared and provide evidence for selecting the best model.

The best configuration was selected based on the highest weighted F1-score obtained during evaluation.


Reproducibility

The experiments use a fixed random state of 42 and a test size of 20% to make the results reproducible.


Technologies Used

Python

Pandas

NumPy

Scikit-learn

MLflow

Joblib

Git

GitHub

Conclusion
Checkpoint 2 established the data processing pipeline and baseline benchmark. Checkpoint 3 improved the modelling workflow by testing multiple hyperparameter configurations and tracking experiments with MLflow. The best-performing configuration was selected based on the evaluation results and saved for future use.



👩‍💻 Author

Purna Tejitha


