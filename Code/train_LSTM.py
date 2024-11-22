# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Developer details: 
        # Name: Shubh Gupta and Rupal Mishra
        # Role: Developers
        # Code ownership rights: PreProd Corp
    # Version:
        # Version: V 1.1 (19 October 2024)
            # Developers: Shubh Gupta and Rupal Mishra
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
# Define a Long Short-Term Memory (LSTM) neural network class
class LSTM(nn.Module):
    def __init__(self, input_size, num_layers):
        """
        Initializes the LSTM model.

        Parameters:
        input_size (int): The size of the input data.
        num_layers (int): The number of LSTM layers.
        """
        super(LSTM, self).__init__()  # Initialize the parent class
        self.num_layers = num_layers  # Number of LSTM layers
        
        # Hidden size for LSTM
        self.hidden_size = 64
        
        # Create LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        
        # Fully connected layer to produce output
        self.fc = nn.Linear(self.hidden_size, 1)  # Output a single value for binary classification
        
        # Activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Parameters:
        x (Tensor): Input tensor.

        Returns:
        Tensor: Output tensor after passing through the model.
        """
        x = x.unsqueeze(1)  # Add time-step dimension for input (batch_size, time_step, input_size)
        
        # Pass through the LSTM layers
        out, _ = self.lstm(x)
        
        # Use the output from the last time-step
        out = self.fc(out[:, -1, :])  # Get the output from the last time step
        out = self.sigmoid(out)  # Apply sigmoid to get probabilities
        return out
def train_evaluate_lstm(df, n_layers, epochs, db_choice):
    """
    Trains the LSTM model and evaluates its performance.

    Parameters:
    df (DataFrame): Input DataFrame containing the training data.
    n_layers (int): Number of LSTM layers.
    epochs (int): Number of training epochs.
    db_choice (str): The selected database ('MongoDB' or 'MySQL').

    Returns:
    The result of the model evaluation on the test set.
    """
    # Preprocess the data and create tensors for training and testing
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = preprocess_data(df)
    
    # Initialize the LSTM model
    model = LSTM(input_size=X_train_tensor.shape[1], num_layers=n_layers)
    
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
            print(f'[LSTM] Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
    
    # Save the trained model
    model_path = 'Code/saved_model/LSTM.pkl'
    joblib.dump(model, model_path)  # Save the model using joblib
    
    # Store model details in the database based on user's choice
    if db_choice == "MongoDB":
        store_model_to_mongodb(
            model_name='LSTM',  # Corrected the model name here
            model_path=model_path,
            num_layers=n_layers,
            epochs=epochs,
        )
    else:  # MySQL
        store_model_to_mysql(
            model_name='LSTM',
            model_path=model_path,
            num_layers=n_layers,
            epochs=epochs,
        )
    
    # Evaluate the model on the test set and return the results
    return evaluate_model(model, X_test_tensor, y_test_tensor)
