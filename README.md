# Deep Learning Neural Network using PyTorch

This repository demonstrates how to build and train deep learning models for customer churn prediction using PyTorch. Designed for those with basic machine learning knowledge, it focuses on practical implementation and optimization. Gain hands-on experience while bridging the gap between concepts and real-world applications.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Data Definition](#data-definition)
- [Program Flow](#program-flow)
- [Steps to Run](#steps-to-run)

---

## Project Overview

### Business Case
A telecom company is facing a high rate of customer churn, where existing customers are leaving and switching to competitors. The company's goal is to predict which customers are most likely to churn so they can take targeted actions to retain them before they leave. This prediction can help reduce churn rates and improve customer retention efforts.

### Industry
Telecommunications

### Problem Statement
The company is losing customers to competitors. To mitigate this, they want to predict churn using historical data, including customer demographics, services subscribed, and billing information. By identifying patterns that lead to churn, the company can take proactive steps to improve customer satisfaction and retention rates.

### Objective
The aim is to build a predictive model to forecast customer churn based on key customer features such as:
- Demographics (e.g., age, gender)
- Services Subscribed (e.g., internet, phone, streaming services)
- Monthly Billing and Contract Information (e.g., monthly charges, contract type)
  
By using this predictive model, the telecom company can focus its retention efforts on high-risk customers, offering personalized incentives to encourage them to stay.

> To Learn more about this project using your favourite LLM, [click here](prompts.md)!

---

## Directory Structure

```plaintext
code/
├── __pycache__/                 (directory for compiled Python files)
├── saved_model/                 (directory for saved model files and training scripts)
│   ├── CNN.pkl                     (saved model for the Convolutional Neural Network)
│   ├── FCNN.pkl                    (saved model for the Fully Connected Neural Network)
│   ├── LSTM.pkl                    (saved model for the Long Short-Term Memory network)
│   ├── MLP.pkl                     (saved model for the Multi-Layer Perceptron)
│   ├── RNN.pkl                     (saved model for the Recurrent Neural Network)
├── app.py                      (main application file for the Streamlit web app)
├── classification.py           (script for classification-related functions and utilities)
├── evaluate.py                 (script to evaluate model performance on test data)
├── ingest_transform_mongodb.py (script for ingesting and transforming data into MongoDB)
├── ingest_transform.py         (script for general data ingestion and transformation)
├── train_CNN.py                (script for training the Convolutional Neural Network)
├── train_FCNN.py               (script for training the Fully Connected Neural Network)
├── train_LSTM.py               (script for training the Long Short-Term Memory network)
├── train_MLP.py                (script for training the Multi-Layer Perceptron)
│── train_RNN.py                (script for training the Recurrent Neural Network)
│── Data/
│    └── Master/
│        └── telecom_customer_data.csv   (CSV file containing customer data for analysis)
├── .gitattributes               (file for managing Git attributes)
├── .gitignore                   (specifies files and directories to be ignored by Git)
├── README.md                    (documentation for the project)
└── requirements.txt             (lists the dependencies required for the project)

```
---

## Data Definition

The dataset contains features like:
- Customer demographics (e.g., gender, age, region)
- Customer subscription data (e.g., services like phone, internet, etc.)
- Financial data (e.g., monthly charges, total charges)
- Tenure (e.g., length of subscription)
- Churn status (target variable indicating whether a customer has churned)

### Data Splitting:
- **Training Samples:** 8000
- **Testing Samples:** 2000

---

## Program Flow

1. **Data Ingestion:** Load data from the `Data` directory (e.g., CSV files) and ingest it into a database (MongoDB or SQLite). [`ingest_transform_mongodb.py`, `ingest_transform.py`]
2. **Data Transformation:** Preprocess the data, including encoding categorical features, handling missing values, and normalizing numerical values. The data is then split into training, testing, and validation sets. [`ingest_transform.py`]
3. **Model Training:** Train a deep learning model (e.g., using TensorFlow or PyTorch) with techniques like cross-validation for hyperparameter tuning. [ `train_cnn.py`, `train_FCNN.py`, `train_LSTM.py`, `train_RNN.py`, `train_MLP.py`]
4. **Model Evaluation:** Evaluate the model's performance on the test and validation sets, generating classification reports and confusion matrices. [`evaluate.py`]
5. **Manual Prediction:** Allow users to manually input customer data for churn prediction using the CLI or API. [`classification.py`]
6. **Web Application:** A `Streamlit` app that integrates the entire pipeline and allows users to interact with the model, providing predictions based on customer input. [`app.py`]

---

## Steps to Run

1. Clone the repository from GitHub:
   ```bash
   git clone https://github.com/Preprod-Labs/DIY-DeepLearning-NN-PyTorch
   ```
2. To create and activate a virtual environment (recommended) using Conda, run the following commands:
   ```
   conda create -n env_name python==3.10.11 -y
   conda activate env_name
   ```
   *Note: This project uses Python 3.10.11*

3. Install the dependencies using:
   ```
   pip install -r requirements.txt
   ```

4. Run the following command to launch the Streamlit web application and use the GUI for the entire pipeline:
     ```
     streamlit run code/app.py
     ```