# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Description: This script implements and trains a Fully Connected Neural Network (FCNN) model for customer churn prediction.
        # MySQL: Yes
        # MongoDB: Yes

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Dependency: 
    # Environment:     
        # Python 3.10.11
        # Streamlit 1.40.0
        # torch 2.5.0

import pandas as pd  # For data manipulation and analysis
import joblib  # For saving and loading the trained model
import torch
import torch.nn as nn  # For building neural networks
from ingest_transform import preprocess_data # Custom functions for preprocessing data and storing the model
from evaluate import evaluate_model  # Custom function to evaluate the model
from ingest_transform_mongodb import store_model_to_mongodb
from ingest_transform import store_model_to_mysql
import os

# Define a Fully Connected Neural Network (FCNN) class
class FCNN(nn.Module):
    def __init__(self, input_size, num_layers):
        """
        Initializes the FCNN model.

        Parameters:
        input_size (int): The size of the input data.
        num_layers (int): The number of hidden layers.
        """
        super(FCNN, self).__init__()  # Initialize the parent class
        
        self.num_layers = num_layers
        self.layers_list = nn.ModuleList()  # Use nn.ModuleList to hold the layers
        
        # Set the size of the first layer based on the number of layers
        layer_input = 8 * 2 ** num_layers  # Example scaling for the first layer size
        self.fc1 = nn.Linear(input_size, layer_input)  # First layer
        self.layers_list.append(self.fc1)  # Add the first layer to the layers list
        
        # Hidden layers: continue halving the size until we reach 32
        while layer_input >= 32:
            fc_hidden = nn.Linear(int(layer_input), int(layer_input // 2))  # Create a hidden layer
            self.layers_list.append(fc_hidden)  # Add hidden layer to the layers list
            layer_input = layer_input // 2  # Update layer_input for the next hidden layer
        
        # Final output layer
        self.fc3 = nn.Linear(32, 1)  # Output a single value for binary classification
        
        # Activation functions
        self.relu = nn.ReLU()  # ReLU activation function for hidden layers
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for output layer
        print(self.layers_list[:-1])  # Print hidden layers for debugging

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Parameters:
        x (Tensor): Input tensor.

        Returns:
        Tensor: Output tensor after passing through the model.
        """
        # Apply ReLU activation to all layers except the last output layer
        for layer in self.layers_list[:-1]:  # Iterate through hidden layers
            x = self.relu(layer(x))
        
        # Apply the output layer and sigmoid activation
        x = self.sigmoid(self.fc3(x))  # Final output layer
        return x.squeeze()  # Remove single-dimensional entries from the shape

def train_model(df, n_layers, epochs, db_choice, model_dir):
    """
    Trains the FCNN model.

    Parameters:
    df (DataFrame): Input DataFrame containing the training data.
    n_layers (int): Number of hidden layers.
    epochs (int): Number of training epochs.
    db_choice (str): The selected database ('MongoDB' or 'MySQL').
    model_dir (str): Directory path to save the trained model.

    Returns:
    The result of the model evaluation on the test set.
    """
    # Preprocess the data and create tensors for training and testing
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = preprocess_data(df)

    # Initialize the FCNN model
    model = FCNN(input_size=X_train_tensor.shape[1], num_layers=n_layers)
    
    # Define the loss function and optimizer
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

    # Training loop
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        optimizer.zero_grad()  # Clear gradients for each training step

        # Forward pass: compute outputs
        outputs = model(X_train_tensor)
        
        # Compute the loss
        loss = criterion(outputs.view(-1), y_train_tensor)  # Reshape outputs to match target shape

        # Backward pass: compute gradients
        loss.backward()
        optimizer.step()  # Update model parameters based on gradients

        # Print loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'[FCNN] Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
    
    # Save the trained model using the provided directory
    model_path = os.path.join(model_dir, 'FCNN.pkl')
    joblib.dump(model, model_path)  # Save the model using joblib
    
    # Store model details in the database based on user's choice
    if db_choice == "MongoDB":
        store_model_to_mongodb(
            model_name='FCNN',
            model_path=model_path,
            num_layers=n_layers,
            epochs=epochs,
        )
    else:  # MySQL
        store_model_to_mysql(
            model_name='FCNN',
            model_path=model_path,
            num_layers=n_layers,
            epochs=epochs,
        )
    
    # Evaluate the model on the test set and return the results
    return evaluate_model(model, X_test_tensor, y_test_tensor)




