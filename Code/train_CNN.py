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
import joblib  # For loading the trained model
import torch
import torch.nn as nn  # For building neural networks
from ingest_transform import preprocess_data # Custom functions for data preprocessing and storing model
from evaluate import evaluate_model  # Custom function to evaluate the model
from ingest_transform_mongodb import store_model_to_mongodb
from ingest_transform import store_model_to_mysql
# Define a Convolutional Neural Network (CNN) class
class CNN(nn.Module):
    def __init__(self, input_size, num_layers):
        """
        Initializes the CNN model.

        Parameters:
        input_size (int): The size of the input data.
        num_layers (int): The number of convolutional layers.
        """
        super(CNN, self).__init__()  # Initialize the parent class
        self.num_layers = num_layers
        
        # Starting channel size for the convolutional layers
        in_channels = 1
        out_channels = 16

        # Create a list to hold the convolutional layers
        self.conv_layers = nn.ModuleList()
        
        # Dynamically create convolutional layers based on the number specified
        for i in range(num_layers):
            # Append a 1D convolutional layer to the list
            self.conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3))
            in_channels = out_channels  # Update in_channels for the next layer
            out_channels *= 2  # Double the out_channels for the next layer
        
        # Calculate the size of the output after the convolutional layers
        conv_output_size = input_size - 2 * num_layers  # Each Conv1d reduces the size by 2
        flattened_size = in_channels * conv_output_size  # Calculate the size after flattening
        
        # Define fully connected layers
        self.fc1 = nn.Linear(flattened_size, 64)  # First fully connected layer
        self.fc2 = nn.Linear(64, 1)  # Second fully connected layer for binary classification
        
        # Activation functions
        self.relu = nn.ReLU()  # ReLU activation function
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation function for output layer

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Parameters:
        x (Tensor): Input tensor.

        Returns:
        Tensor: Output tensor after passing through the model.
        """
        x = x.unsqueeze(1)  # Add channel dimension for 1D convolution (batch_size, 1, input_size)
        
        # Pass the input through all convolutional layers with ReLU activation
        for conv in self.conv_layers:
            x = self.relu(conv(x))
        
        # Flatten the output from the convolutional layers for the fully connected layers
        x = x.view(x.size(0), -1)  # Flatten the tensor
        
        # Pass through the fully connected layers
        x = self.relu(self.fc1(x))  # First fully connected layer
        x = self.sigmoid(self.fc2(x))  # Second fully connected layer, sigmoid for binary output
        
        return x  # Return the output

# Function to train and evaluate the CNN model
def train_evaluate_cnn(df, n_layers, epochs, db_choice):
    """
    Trains and evaluates the CNN model.

    Parameters:
    df (DataFrame): The input DataFrame containing customer data.
    n_layers (int): The number of convolutional layers.
    epochs (int): The number of training epochs.
    db_choice (str): The selected database ('MongoDB' or 'MySQL').

    Returns:
    The result of the evaluation.
    """
    # Preprocess the data to get tensors for training and testing
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = preprocess_data(df)
    
    # Initialize the CNN model
    model = CNN(input_size=X_train_tensor.shape[1], num_layers=n_layers)
    
    # Define the loss function and optimizer
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

    # Training loop
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        optimizer.zero_grad()  # Clear the gradients

        # Forward pass: compute model outputs
        outputs = model(X_train_tensor)
        
        # Compute loss
        loss = criterion(outputs.view(-1), y_train_tensor)  # Reshape outputs to match target shape

        # Backward pass: compute gradients
        loss.backward()
        optimizer.step()  # Update model parameters

        # Print loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'[CNN] Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
    
    # Save the trained model
    model_path = 'Code/saved_model/CNN.pkl'
    joblib.dump(model, model_path)  # Save model using joblib
    
    # Store model details in the database based on user's choice
    if db_choice == "MongoDB":
        store_model_to_mongodb(
            model_name='CNN',
            model_path=model_path,
            num_layers=n_layers,
            epochs=epochs,
        )
    else:  # MySQL
        store_model_to_mysql(
            model_name='CNN',
            model_path=model_path,
            num_layers=n_layers,
            epochs=epochs,
        )

    
    # Evaluate the model on the test set and return the results
    return evaluate_model(model, X_test_tensor, y_test_tensor)
