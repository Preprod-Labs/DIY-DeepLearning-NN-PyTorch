# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
     
# Description: This script handles model evaluation and performance metrics calculation for neural network models.
    # MySQL: Yes
    # MongoDB: Yes

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Dependency: 
    # Environment:     
        # Python 3.10.11
        # torch 2.5.0
        # joblib 1.3.1
        # scikit-learn 1.2.2
        # numpy 1.24.3


import os  # Importing os to interact with the operating system (e.g., file and directory operations)
import torch  # Importing PyTorch for tensor operations and model evaluation
import joblib  # Importing joblib for loading saved models
from sklearn.metrics import classification_report  # Importing classification_report for generating evaluation metrics

def evaluate_model(model, X_test_tensor, y_test_tensor):
    """Evaluate a model using the test tensors and return the accuracy."""
    with torch.no_grad():  # Disable gradient calculation for evaluation to save memory and improve speed
        
        # Obtain predictions from the model
        predictions = model(X_test_tensor)
        
        # Convert predictions to binary (0 or 1) based on a threshold of 0.5
        predicted = (predictions.view(-1) > 0.5).float()
        
        # Calculate accuracy by comparing predicted and actual values
        accuracy = (predicted == y_test_tensor).sum() / y_test_tensor.shape[0]
        return f'{accuracy.item()}%'  # Return accuracy as a percentage

def evaluate_(X_test_tensor, y_test_tensor, model_dir):
    """Evaluate all saved models in the specified directory and return their classification reports."""
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
    reports = {}

    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        try:
            model = joblib.load(model_path)
            model.eval()  # Set the model to evaluation mode to disable dropout and batch normalization
            with torch.no_grad():  # Disable gradient calculation during prediction
                predictions = model(X_test_tensor)  # Get predictions for the test data
                predicted = (predictions.view(-1) > 0.5).float()  # Convert predictions to binary

            # Generate a classification report using sklearn's classification_report
            report = classification_report(y_test_tensor, predicted)
            reports[model_file] = report  # Store the report in the dictionary
        except Exception as e:
            reports[model_file] = f"Error loading model: {str(e)}"

    return reports  # Return the dictionary containing classification reports for all models


