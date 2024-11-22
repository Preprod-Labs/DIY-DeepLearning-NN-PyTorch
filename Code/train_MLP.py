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
     
    # Description: This Streamlit app allows users to input features and make predictions using Neural Network.
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
from ingest_transform import preprocess_data # Custom functions for preprocessing and storing the model
from evaluate import evaluate_model  # Custom function to evaluate the model
from ingest_transform_mongodb import store_model_to_mongodb
from ingest_transform import store_model_to_mysql
# Define a Multi-Layer Perceptron (MLP) neural network class
class MLP(nn.Module):
    def __init__(self, input_size, num_layers):
        """
        Initializes the MLP model.

        Parameters:
        input_size (int): The size of the input data.
        num_layers (int): The number of hidden layers in the MLP.
        """
        super(MLP, self).__init__()  # Initialize the parent class
        self.num_layers = num_layers  # Number of hidden layers
        
        # Create a list to hold the layers
        self.layers = nn.ModuleList()
        
        # Starting size for the first layer
        in_size = input_size
        
        # Create the hidden layers
        for i in range(num_layers):
            out_size = max(32, in_size // 2)  # Decrease size of layers, ensuring a minimum of 32 units
            self.layers.append(nn.Linear(in_size, out_size))  # Add a linear layer
            in_size = out_size  # Update in_size for the next layer
        
        # Final output layer
        self.output_layer = nn.Linear(in_size, 1)  # Output layer for binary classification
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Parameters:
        x (Tensor): Input tensor.

        Returns:
        Tensor: Output tensor after passing through the model.
        """
        # Pass through each layer
        for layer in self.layers:
            x = self.relu(layer(x))  # Apply ReLU activation
            
        # Output layer with sigmoid activation
        x = self.sigmoid(self.output_layer(x))  # Get probabilities
        return x
def train_evaluate_mlp(df, n_layers, epochs, db_choice):
    """
    Trains the MLP model and evaluates its performance.

    Parameters:
    df (DataFrame): Input DataFrame containing the training data.
    n_layers (int): Number of hidden layers in the MLP.
    epochs (int): Number of training epochs.
    db_choice (str): The selected database ('MongoDB' or 'MySQL').

    Returns:
    The result of the model evaluation on the test set.
    """
    # Preprocess the data and create tensors for training and testing
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = preprocess_data(df)
    
    # Initialize the MLP model
    model = MLP(input_size=X_train_tensor.shape[1], num_layers=n_layers)
    
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
            print(f'[MLP] Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
    
    # Save the trained model
    model_path = 'Code/saved_model/MLP.pkl'
    joblib.dump(model, model_path)  # Save the model using joblib
    
    # Store model details in the database based on user's choice
    if db_choice == "MongoDB":
        store_model_to_mongodb(
            model_name='MLP',  # Corrected the model name here
            model_path=model_path,
            num_layers=n_layers,
            epochs=epochs,
        )
    else:  # MySQL
        store_model_to_mysql(
            model_name='MLP',
            model_path=model_path,
            num_layers=n_layers,
            epochs=epochs,
        )
    
    # Evaluate the model on the test set and return the results
    return evaluate_model(model, X_test_tensor, y_test_tensor)
