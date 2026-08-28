Customer Support Intent Classification using XGBoost

An XGBoost-based NLP system that classifies customer support queries into six intents using TF-IDF text features and provides prediction confidence.

Overview

This project uses Natural Language Processing (NLP) and Machine Learning to automatically identify the intent behind customer support messages.

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