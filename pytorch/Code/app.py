# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
     
# Description: This Streamlit web application provides a user interface for training, evaluating, and making predictions with various neural network models for customer churn prediction.
    # MySQL: Yes
    # MongoDB: Yes

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Dependency: 
    # Environment:     
        # Python 3.10.11
        # streamlit 1.40.0
        # pandas 1.5.3
        # numpy 1.24.3
        # torch 2.5.0
        # pymongo 4.9.1
        # mysql-connector-python 9.0.0
        # python-dotenv 1.0.1
        # scikit-learn 1.2.2
        # joblib 1.3.1

# Import necessary libraries and modules
from classification import classify  # Import classification functions
from evaluate import evaluate_  # Import evaluation functions
import numpy as np  # Import NumPy for numerical operations
import pandas as pd  # Import Pandas for data manipulation
import streamlit as st  # Import Streamlit for creating the web app
from dotenv import load_dotenv  # Import for loading environment variables from a .env file
from train_FCNN import train_model as train_fcnn  # Import training function for Fully Connected Neural Network (FCNN)
from train_CNN import train_evaluate_cnn as train_cnn  # Import training and evaluation function for Convolutional Neural Network (CNN)
from train_RNN import train_evaluate_rnn as train_rnn  # Import training and evaluation function for Recurrent Neural Network (RNN)
from train_LSTM import train_evaluate_lstm as train_lstm  # Import training and evaluation function for Long Short-Term Memory (LSTM)
from train_MLP import train_evaluate_mlp as train_mlp  # Import training and evaluation function for Multi-Layer Perceptron (MLP)
from ingest_transform_mongodb import  store_data_to_mongodb, retrieve_data_from_mongodb  # Import MongoDB functions
from ingest_transform import delabel, preprocess_data, scale_test, store_data_to_mysql, retrieve_data_from_mysql
import os
# Configure Streamlit page settings
st.set_page_config(page_title="Customer Churn Prediction", page_icon=":cash:", layout="centered")

# Display the main title of the web app with custom styling
st.markdown("<h1 style='text-align: center; color: white;'>Customer Churn Prediction</h1>", unsafe_allow_html=True)

# Add a visual divider to separate the title from the rest of the content
st.divider()


# Define tabs in the user interface for organizing different functionalities
tab1, tab2, tab3, tab4 = st.tabs(["Model Config", "Model Training", "Model Evaluation", "Model Prediction"])

with tab1:
    # Dropdown for selecting the database (MySQL or MongoDB)
    db_choice = st.selectbox("Select Database", ["MongoDB", "MySQL"])

    # Input field for the user to enter the file path of the master data
    default_path = "Data\\Master\\telecom_customer_data.csv"
    uploaded_file = st.text_input("Enter the path to the Master data", value=default_path)

    # Add model save directory input
    default_model_dir = "Code/saved_model"
    model_save_dir = st.text_input("Enter the directory path to save models", value=default_model_dir)
    
    if model_save_dir:
        try:
            # Create directory if it doesn't exist
            os.makedirs(model_save_dir, exist_ok=True)
            st.session_state['model_save_dir'] = model_save_dir
            st.success(f"Model save directory set to: {model_save_dir}")
        except Exception as e:
            st.error(f"Error creating directory: {e}")

    # Check if the user has provided a file path
    if uploaded_file:
        try:
            # Attempt to load the CSV file from the provided path
            df = pd.read_csv(uploaded_file)
            # Check if the DataFrame is empty after loading
            if df.empty:
                # If the file is empty, display a warning message to the user
                st.warning("The uploaded file is empty. Please check the file and try again.")
            else:
                # Display a message indicating a preview of the data will be shown
                st.write("Here is a preview of your data:")
                # Display the first few rows of the DataFrame for user inspection
                st.write(df.head())  

                # Update the session state to store the newly provided master data path
                st.session_state.master_data_path = uploaded_file

                # Call the appropriate function to store the uploaded file path based on user's choice
                if db_choice == "MongoDB":
                    message = store_data_to_mongodb(df)
                else:  # MySQL
                    message = store_data_to_mysql(df)  # Use appropriate MySQL function
                
                # Display a success message indicating the path was stored successfully
                st.success(message)

        except FileNotFoundError:
            # Handle the case where the file is not found at the specified path
            st.error(f"File not found at the specified path: {uploaded_file}. Please check the path and try again.")
        except pd.errors.EmptyDataError:
            # Handle the case where the file is empty or not properly formatted
            st.error(f"File is empty or not properly formatted. Please check the file content.")
        except Exception as e:
            # Catch any other exceptions that may occur and display the error message
            st.error(f"An error occurred while loading the file: {e}")

with tab2:
    # Display a subheader for the model training section
    st.subheader("Model Training")
    st.write("This is where you can train the model.")  # Inform users about this tab's purpose
    st.divider()  # Add a visual divider for better layout

    # Retrieve data from the selected database
    if db_choice == "MongoDB":
        df = retrieve_data_from_mongodb()  # Assuming this function retrieves data from MongoDB
    else:  # MySQL
        df = retrieve_data_from_mysql()  # Use appropriate MySQL function
        print(type(df))
    # Continue processing the retrieved data as necessary


    # Check if data was loaded successfully from the database
    if df is None or df.empty:
        # If no data was found or unable to load data, show an error message to the user
        st.error("No data found in the database. Please upload data first.")
    else:
        # Start model training for Fully Connected Neural Network (FCNN)
        model_name = 'FCNN'
        # Display the model name in a centered format
        st.markdown(f"<h3 style='text-align: center; color: white;'>{model_name}</h3>", unsafe_allow_html=True)

        # User input for specifying the number of layers and epochs for training the FCNN
        num_layers = st.number_input('Number of Layers:', min_value=2, max_value=10, value=5, step=1)
        epochs = st.number_input('Number of Epochs:', min_value=10, max_value=1000, value=10, step=10)

        # Button to trigger the training process for the FCNN model
        if st.button(f"Train {model_name} Model", use_container_width=True):
            with st.spinner(f"Training {model_name} Model..."):  # Show a spinner while training
                try:
                    # Call the training function for FCNN and capture the score
                    score = train_fcnn(df, num_layers, epochs, db_choice, st.session_state.model_save_dir)
                    # Display a success message upon successful training
                    st.success(f"{model_name} Trained Successfully!")

                    # Show the training completion message and the model's score
                    st.write(f"Training complete! The score is: {score}")
                    st.write(f"Accuracy: {score}")

                except Exception as e:
                    # If an error occurs during training, show an error message
                    st.error(f"An error occurred during training: {e}")

        st.divider()  # Add a visual divider between different model training sections

        # Model Training for Convolutional Neural Network (CNN)
        model_name = 'CNN'
        st.markdown(f"<h3 style='text-align: center; color: white;'>{model_name}</h3>", unsafe_allow_html=True)

        # User input for the number of layers and epochs for the CNN
        num_layers = st.number_input('Number of Layers:', min_value=2, max_value=6, value=3, step=1, key="num_layers_input")
        epochs = st.number_input('Number of Epochs:', min_value=10, max_value=1000, value=10, step=10, key="epoch_input")
        
        # Button to trigger the training process for the CNN model
        if st.button(f"Train {model_name} Model", use_container_width=True):
            with st.spinner(f"Training {model_name} Model..."):  # Show a spinner during training
                try:
                    # Call the training function for CNN and capture the score
                    score = train_cnn(df, num_layers, epochs, db_choice, st.session_state.model_save_dir)
                    # Display a success message upon successful training
                    st.success(f"{model_name} Trained Successfully!")

                    # Show the training completion message and the model's score
                    st.write(f"Training complete! The score is: {score}")
                    st.write(f"Accuracy: {score}")

                except Exception as e:
                    # If an error occurs during training, show an error message
                    st.error(f"An error occurred during training: {e}")

        st.divider()  # Add a visual divider

        # Model Training for Recurrent Neural Network (RNN)
        model_name = 'RNN'
        st.markdown(f"<h3 style='text-align: center; color: white;'>{model_name}</h3>", unsafe_allow_html=True)

        # User input for the number of layers and epochs for the RNN
        num_layers = st.number_input('Number of Layers:', min_value=2, max_value=10, value=3, step=1, key="num_layers_input2")
        epochs = st.number_input('Number of Epochs:', min_value=10, max_value=1000, value=10, step=10, key="epoch_input2")
        
        # Button to trigger the training process for the RNN model
        if st.button(f"Train {model_name} Model", use_container_width=True):
            with st.spinner(f"Training {model_name} Model..."):  # Show a spinner during training
                try:
                    # Call the training function for RNN and capture the score
                    score = train_rnn(df, num_layers, epochs, db_choice, st.session_state.model_save_dir)
                    # Display a success message upon successful training
                    st.success(f"{model_name} Trained Successfully!")

                    # Show the training completion message and the model's score
                    st.write(f"Training complete! The score is: {score}")
                    st.write(f"Accuracy: {score}")

                except Exception as e:
                    # If an error occurs during training, show an error message
                    st.error(f"An error occurred during training: {e}")

        st.divider()  # Add a visual divider

        # Model Training for Long Short-Term Memory (LSTM)
        model_name = 'LSTM'
        st.markdown(f"<h3 style='text-align: center; color: white;'>{model_name}</h3>", unsafe_allow_html=True)

        # User input for the number of layers and epochs for the LSTM
        num_layers = st.number_input('Number of Layers:', min_value=2, max_value=10, value=3, step=1, key="num_layers_input3")
        epochs = st.number_input('Number of Epochs:', min_value=10, max_value=1000, value=10, step=10, key="epoch_input3")
        
        # Button to trigger the training process for the LSTM model
        if st.button(f"Train {model_name} Model", use_container_width=True):
            with st.spinner(f"Training {model_name} Model..."):  # Show a spinner during training
                try:
                    # Call the training function for LSTM and capture the score
                    score = train_lstm(df, num_layers, epochs, db_choice, st.session_state.model_save_dir)
                    # Display a success message upon successful training
                    st.success(f"{model_name} Trained Successfully!")

                    # Show the training completion message and the model's score
                    st.write(f"Training complete! The score is: {score}")
                    st.write(f"Accuracy: {score}")

                except Exception as e:
                    # If an error occurs during training, show an error message
                    st.error(f"An error occurred during training: {e}")

        st.divider()  # Add a visual divider

        # Model Training for Multi-Layer Perceptron (MLP)
        model_name = 'MLP'
        st.markdown(f"<h3 style='text-align: center; color: white;'>{model_name}</h3>", unsafe_allow_html=True)

        # User input for the number of layers and epochs for the MLP
        num_layers = st.number_input('Number of Layers:', min_value=2, max_value=10, value=3, step=1, key="num_layers_input4")
        epochs = st.number_input('Number of Epochs:', min_value=10, max_value=1000, value=10, step=10, key="epoch_input4")
        
        # Button to trigger the training process for the MLP model
        if st.button(f"Train {model_name} Model", use_container_width=True):
            with st.spinner(f"Training {model_name} Model..."):  # Show a spinner during training
                try:
                    # Call the training function for MLP and capture the score
                    score = train_mlp(df, num_layers, epochs, db_choice, st.session_state.model_save_dir)
                    # Display a success message upon successful training
                    st.success(f"{model_name} Trained Successfully!")

                    # Show the training completion message and the model's score
                    st.write(f"Training complete! The score is: {score}")
                    st.write(f"Accuracy: {score}")

                except Exception as e:
                    # If an error occurs during training, show an error message
                    st.error(f"An error occurred during training: {e}")

with tab3:
    st.subheader("Model Evaluation")
    st.write("This is where you can see the current metrics of the latest saved model.")
    st.divider()
    
    # Retrieve and check data before preprocessing
    if db_choice == "MongoDB":
        df = retrieve_data_from_mongodb()
    else:
        df = retrieve_data_from_mysql()
    
    if df is None or df.empty:
        st.error("No data available in the database. Please upload data first.")
    else:
        try:
            # Preprocess the data to create tensors for training and testing
            X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = preprocess_data(df)

            # List of model filenames to evaluate
            model_list = ['CNN.pkl', 'FCNN.pkl', 'LSTM.pkl', 'MLP.pkl', 'RNN.pkl']

            # Loop through each model in the model list
            for model_name in model_list:
                # Display the model name in a centered white text header
                st.markdown(f"<h3 style='text-align: center; color: white;'>{model_name[:-4]} Model Evaluation</h3>", 
                            unsafe_allow_html=True)
                
                try:
                    # Get evaluation metrics for the current model
                    metrics = evaluate_(X_test_tensor, y_test_tensor, st.session_state.model_save_dir)
                    
                    if model_name in metrics:
                        # Display the classification report in a formatted way
                        st.code(metrics[model_name], language='text')
                    else:
                        st.warning(f"No evaluation metrics found for {model_name[:-4]}. Please train the model first.")
                
                except Exception as e:
                    st.error(f"Error evaluating {model_name[:-4]}: {str(e)}")
                
                st.divider()

        except Exception as e:
            st.error(f"Error during data preprocessing: {str(e)}")
            st.info("Please ensure your data is properly formatted and try again.")

with tab4:  # Create a new tab for model prediction
    # Dropdown to select the classification algorithm
    algorithm = st.selectbox("Select algorithm:", ("FCNN", "CNN", "RNN", "MLP", "LSTM"))

    # Create a form for user input
    with st.form(key="Deep_learning_form"):
        st.subheader("Deep Learning")  # Subheader for the prediction section
        st.write("Enter customer details for classification")  # Instruction for the user

        # Input field for tenure (in months), with a defined range and default value
        tenure = st.number_input('Enter Tenure (1-100 months):', min_value=1, max_value=100, value=12)

        # Radio button for selecting contract type; the selected value is delabeled for processing
        contract_type = delabel(st.radio('Choose Contract Type:', ['Month-to-month', 'One year', 'Two year'], horizontal=True))

        # Input field for monthly charges, with a defined range and formatting for two decimal places
        monthly_charges = st.number_input('Enter Monthly Charges (0.00-1000.00):', min_value=0.00, max_value=1000.00, value=35.65, step=0.01, format="%.2f")

        # Input field for total charges, with a defined range and formatting
        total_charges = st.number_input('Enter Total Charges (0.00-50000.00):', min_value=0.00, max_value=50000.00, value=4091.54, step=0.01, format="%.2f")

        # Radio button for selecting the type of internet service
        internet_service = delabel(st.radio('Choose Internet Service:', ['DSL', 'Fiber optic'], horizontal=True))

        # Radio buttons for various services and support options, processed with delabel
        online_security = delabel(st.radio('Online Security:', ['Yes', 'No'], horizontal=True))
        tech_support = delabel(st.radio('Tech Support:', ['Yes', 'No'], horizontal=True))
        streaming_tv = delabel(st.radio('Streaming TV:', ['Yes', 'No'], horizontal=True))
        streaming_movies = delabel(st.radio('Streaming Movies:', ['Yes', 'No'], horizontal=True))

        # Radio button for payment method, processed with delabel
        payment_method = delabel(st.radio('Payment Method:', ['Bank transfer', 'Credit card', 'Electronic check', 'Mailed check'], horizontal=True))

        # Radio button for indicating if the user is a senior citizen, converted to a boolean
        senior_citizen = st.radio('Senior Citizen:', ['True', 'False'], horizontal=True)
        senior_citizen_bool = True if senior_citizen == 'True' else False  # Convert string to boolean

        # Radio button for paperless billing, also converted to a boolean
        paperless_billing = st.radio('Paperless Billing:', ['True', 'False'], horizontal=True)
        paperless_billing_bool = True if paperless_billing == 'True' else False  # Convert string to boolean

        # Combine all input values into a single NumPy array for classification
        items = np.array([[tenure, contract_type, monthly_charges, total_charges, internet_service, online_security, 
                           tech_support, streaming_tv, streaming_movies, payment_method, senior_citizen_bool, paperless_billing_bool]])

        # Button to submit the form and classify the input data
        if st.form_submit_button("Classify", use_container_width=True):
            # Scale the input data for prediction
            X_test_tensor = scale_test(items)
            
            # Classify the input data using the selected algorithm
            classification = classify(algorithm, X_test_tensor, st.session_state.model_save_dir)

            # Display the result of the classification
            st.write(f"The Result is {'Churn' if classification == 1.0 else 'No Churn'}")
