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
     
    # Description: This script implements and trains a Recurrent Neural Network (RNN) model for customer churn prediction.
        # MySQL: Yes
        # MongoDB: Yes

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Dependency: 
    # Environment:     
        # Python 3.10.11
        # pandas 1.5.3
        # torch 2.5.0
        # joblib 1.3.1
        # pymongo 4.9.1
        # mysql-connector-python 9.0.0
        # scikit-learn 1.2.2
        # numpy 1.24.3


import pandas as pd  # For data manipulation and analysis
import joblib  # For saving and loading the trained model
import torch
import torch.nn as nn  # For building neural networks
from ingest_transform import preprocess_data  # Custom functions for preprocessing and storing the model
from evaluate import evaluate_model  # Custom function to evaluate the model
from ingest_transform_mongodb import store_model_to_mongodb
from ingest_transform import store_model_to_mysql
# Define a Dynamic Recurrent Neural Network (RNN) class
class DynamicRNN(nn.Module):
    def __init__(self, input_size, num_layers):
        """
        Initializes the Dynamic RNN model.

        Parameters:
        input_size (int): The size of the input data.
        num_layers (int): The number of RNN layers in the model.
        """
        super(DynamicRNN, self).__init__()  # Initialize the parent class
        self.num_layers = num_layers  # Number of RNN layers
        
        # Hidden size for RNN
        self.hidden_size = 64
        
        # Create RNN layer
        self.rnn = nn.RNN(input_size=input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        
        # Fully connected layer for output
        self.fc = nn.Linear(self.hidden_size, 1)  # Output size is 1 for binary classification
        
        # Activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Parameters:
        x (Tensor): Input tensor with shape (batch_size, input_size).

        Returns:
        Tensor: Output tensor after passing through the model, representing probabilities.
        """
        x = x.unsqueeze(1)  # Add time-step dimension for input (batch_size, time_step=1, input_size)
        
        # Pass through the RNN layers
        out, _ = self.rnn(x)
        
        # Use the output from the last time-step
        out = self.fc(out[:, -1, :])  # Get the output from the last time step
        out = self.sigmoid(out)  # Apply sigmoid to get probabilities
        return out

def train_evaluate_rnn(df, n_layers, epochs, db_choice):
    """
    Trains the Dynamic RNN model and evaluates its performance.

    Parameters:
    df (DataFrame): Input DataFrame containing the training data.
    n_layers (int): Number of RNN layers in the model.
    epochs (int): Number of training epochs.
    db_choice (str): The selected database ('MongoDB' or 'MySQL').

    Returns:
    The result of the model evaluation on the test set.
    """
    # Preprocess the data and create tensors for training and testing
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = preprocess_data(df)
    
    # Initialize the RNN model
    model = DynamicRNN(input_size=X_train_tensor.shape[1], num_layers=n_layers)
    
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
            print(f'[RNN] Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
    
    # Save the trained model
    model_path = 'Code/saved_model/RNN.pkl'
    joblib.dump(model, model_path)  # Save the model using joblib
    
    # Store model details in the database based on user's choice
    if db_choice == "MongoDB":
        store_model_to_mongodb(
            model_name='RNN',  # Corrected the model name here
            model_path=model_path,
            num_layers=n_layers,
            epochs=epochs,
        )
    else:  # MySQL
        store_model_to_mysql(
            model_name='RNN',
            model_path=model_path,
            num_layers=n_layers,
            epochs=epochs,
        )
    
    # Evaluate the model on the test set and return the results
    return evaluate_model(model, X_test_tensor, y_test_tensor)
