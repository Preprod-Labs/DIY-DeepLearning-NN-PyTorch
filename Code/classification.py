# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Developer details: 
        # Name: Akshat Rastogi, Shubh Gupta and Rupal Mishra
        # Role: Developers
        # Code ownership rights: PreProd Corp
    # Version:
        # Version: V 1.1 (19 October 2024)
            # Developers: Akshat Rastogi, Shubh Gupta and Rupal Mishra
            # Unit test: Pass
            # Integration test: Pass
     
    # Description: This script handles the classification of customer data using trained neural network models.
        # MySQL: Yes
        # MongoDB: Yes

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Dependency: 
    # Environment:     
        # Python 3.10.11
        # torch 2.5.0
        # joblib 1.3.1
        # pandas 1.5.3


from ingest_transform import scale_test  # Import scale_test function for preprocessing input data
import joblib  # Import joblib for loading saved models
import pandas as pd  # Import pandas for data manipulation (not directly used in this snippet)
import torch  # Import PyTorch for handling tensor operations and model evaluations

def classify(algorithm, items):
    """Classify the input data using the specified algorithm model."""
    X_test_tensor = items  # Assign input items to the tensor variable for prediction

    # Load the appropriate model based on the selected algorithm
    if algorithm == "FCNN":
        model = joblib.load('code/saved_model/FCNN.pkl')  # Load FCNN model
    elif algorithm == "CNN":
        model = joblib.load('code/saved_model/CNN.pkl')  # Load CNN model
    elif algorithm == "RNN":
        model = joblib.load('code/saved_model/RNN.pkl')  # Load RNN model
    elif algorithm == "MLP":
        model = joblib.load('code/saved_model/MLP.pkl')  # Load MLP model
    elif algorithm == "LSTM":
        model = joblib.load('code/saved_model/LSTM.pkl')  # Load LSTM model

    model.eval()  # Set the model to evaluation mode to disable dropout and batch normalization
    with torch.no_grad():  # Disable gradient calculations for efficiency during inference
        predictions = model(X_test_tensor)  # Get predictions from the model
        predicted = (predictions.view(-1) > 0.5).float()  # Convert predictions to binary (0 or 1) based on threshold
        return predicted.item()  # Return the predicted class as a Python float
