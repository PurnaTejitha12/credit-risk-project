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
