# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
     
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
import os  # Import os for handling file paths

def classify(algorithm, items, model_dir):
    """Classify the input data using the specified algorithm model."""
    X_test_tensor = items

    # Load the model from the specified directory
    model_path = os.path.join(model_dir, f'{algorithm}.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
        
    model = joblib.load(model_path)
    
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor)
        predicted = (predictions.view(-1) > 0.5).float()
        return predicted.item()
