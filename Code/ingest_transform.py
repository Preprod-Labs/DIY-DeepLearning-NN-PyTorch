# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Developer details: 
        # Name: Akshat Rastogi, Shubh Gupta
        # Role: Developers
        # Code ownership rights: PreProd Corp
    # Version:
        # Version: V 1.1 (19 October 2024)
            # Developers: Akshat Rastogi, Shubh Gupta
            # Unit test: Pass
            # Integration test: Pass
     
    # Description: This script handles data ingestion, preprocessing, and MySQL database operations for the churn prediction system.
        # MySQL: Yes
        # MongoDB: Yes

# Dependency: 
    # Environment:     
        # Python 3.10.11
        # pandas 1.5.3
        # torch 2.5.0
        # mysql-connector-python 9.0.0
        # scikit-learn 1.2.2
        # numpy 1.24.3


# db_connection.py
import mysql.connector
import torch  # Importing PyTorch for tensor operations
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.preprocessing import StandardScaler  # For feature scaling
import pandas as pd  # For data manipulation and analysis

# Mapping categorical values to numerical values for machine learning processing
contract_mapping = {'One year': 0, 'Two year': 1, 'Month-to-month': 2}
internet_service_mapping = {'DSL': 0, 'Fiber optic': 1}
payment_mapping = {'Mailed check': 0, 'Bank transfer': 1, 'Credit card': 2, 'Electronic check': 3}
agree_mapping = {'Yes': 0, 'No': 1}

# Initialize the StandardScaler for feature scaling
scaler = StandardScaler()


def get_mysql_connection():
    """
    Returns a connection object for the MySQL database.
    
    Returns:
    MySQLConnection: A MySQL database connection object.
    """
    try:
        # MySQL database configuration
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='123456',
            database='telecom_db'
        )
        return connection  # Return the connection object
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None  # Return None if connection fails

# Function to label categorical variables in the DataFrame
def labelling(df):
    """
    Maps categorical variables to numerical values in the DataFrame.

    Parameters:
    df (DataFrame): The input DataFrame containing customer data.

    Returns:
    DataFrame: The DataFrame with categorical variables replaced by numerical values.
    """
    global contract_mapping, internet_service_mapping, payment_mapping, agree_mapping

    # Map categorical columns to numerical values using predefined mappings
    df['Contract'] = df['Contract'].map(contract_mapping)
    df['InternetService'] = df['InternetService'].map(internet_service_mapping)
    df['PaymentMethod'] = df['PaymentMethod'].map(payment_mapping)
    df['OnlineSecurity'] = df['OnlineSecurity'].map(agree_mapping)
    df['TechSupport'] = df['TechSupport'].map(agree_mapping)
    df['StreamingTV'] = df['StreamingTV'].map(agree_mapping)
    df['StreamingMovies'] = df['StreamingMovies'].map(agree_mapping)

    return df  # Return the modified DataFrame

# Function to preprocess the data for training and testing
def preprocess_data(df):
    """
    Prepares the DataFrame for machine learning by labeling, splitting, and scaling data.

    Parameters:
    df (DataFrame): The input DataFrame containing customer data.

    Returns:
    Tuple[Tensor, Tensor, Tensor, Tensor]: Tensors for training and testing data (X_train, X_test, y_train, y_test).
    """
    global scaler

    # Drop the 'CustomerID' column as it is not useful for prediction
    df = df.drop(columns=['CustomerID'])

    # Apply labeling to categorical columns
    df = labelling(df)

    # Fill missing values in the 'InternetService' column with the mean of that column
    df['InternetService'] = df['InternetService'].fillna(df['InternetService'].mean())

    # Split features (X) and target variable (y)
    X = df.drop(columns=['Churn'])  # Features
    y = df['Churn']  # Target variable

    # Train-test split, reserving 20% of the data for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling to normalize the feature values
    scaler = StandardScaler()  # Initialize the scaler
    X_train = scaler.fit_transform(X_train)  # Fit to training data and transform
    X_test = scaler.transform(X_test)  # Transform test data using the same scaler

    # Convert numpy arrays to PyTorch tensors for model input
    X_train_tensor = torch.FloatTensor(X_train)  # Training features as tensor
    X_test_tensor = torch.FloatTensor(X_test)  # Testing features as tensor
    y_train_tensor = torch.FloatTensor(y_train.values)  # Training labels as tensor
    y_test_tensor = torch.FloatTensor(y_test.values)  # Testing labels as tensor

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor  # Return the tensors

# Function to decode numerical labels back to original categorical values
def delabel(data):
    """
    Converts numerical labels back to original categorical values.

    Parameters:
    data (str or int): The input data to be converted.

    Returns:
    int: The corresponding numerical label if found; otherwise, returns the input data.
    """
    global contract_mapping, internet_service_mapping, payment_mapping, agree_mapping
    
    # Check if the input data is in any of the mappings and return the corresponding numerical value
    if data in contract_mapping.keys():
        data = contract_mapping[data]
    elif data in internet_service_mapping.keys():
        data = internet_service_mapping[data]
    elif data in payment_mapping.keys():
        data = payment_mapping[data]
    elif data in agree_mapping.keys():
        data = agree_mapping[data]

    return data  # Return the mapped numerical value

# Function to scale new test data using the fitted scaler
def scale_test(df):
    """
    Scales the provided DataFrame using the previously fitted StandardScaler.

    Parameters:
    df (DataFrame): The DataFrame to be scaled.

    Returns:
    Tensor: The scaled data as a PyTorch tensor.
    """
    global scaler
    df = scaler.fit_transform(df)  # Scale the new data
    input_tensor = torch.tensor(df, dtype=torch.float32)  # Convert to PyTorch tensor

    return input_tensor  # Return the scaled tensor


# Function to store the master data path in MySQL
def store_path_to_mysql(path):
    """
    Stores the master data path in a MySQL database.

    Parameters:
    path (str): The file path to be stored.

    Returns:
    str: A success message if the path is stored successfully or an error message.
    """
    connection = get_mysql_connection()
    if connection is None:
        return "Failed to connect to MySQL."

    cursor = connection.cursor()

    try:
        # Create table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS config (
                id INT AUTO_INCREMENT PRIMARY KEY,
                master_data_path VARCHAR(255) NOT NULL
            )
        """)

        # Insert the path into the table
        cursor.execute("INSERT INTO config (master_data_path) VALUES (%s)", (path,))
        connection.commit()
        return "Path stored successfully in MySQL."
    except Exception as e:
        return f"Error storing path to MySQL: {e}"
    finally:
        cursor.close()
        connection.close()

# Function to retrieve the master data path from MySQL and return the DataFrame
def retrieve_path_from_mysql():
    """
    Retrieves the master data path from the MySQL database and loads the CSV into a DataFrame.

    Returns:
    DataFrame or str: The DataFrame read from the CSV file if successful; otherwise, an error message.
    """
    connection = get_mysql_connection()
    if connection is None:
        return "Failed to connect to MySQL."

    cursor = connection.cursor()

    try:
        # Retrieve the latest master data path from the MySQL database
        cursor.execute("SELECT master_data_path FROM config ORDER BY id DESC LIMIT 1")
        result = cursor.fetchone()

        if result:
            csv_path = result[0]  # The master_data_path
            try:
                # Load the CSV file into a DataFrame
                df = pd.read_csv(csv_path)
                return df
            except FileNotFoundError:
                return f"File not found at path: {csv_path}"
            except pd.errors.EmptyDataError:
                return f"No data found in the file at path: {csv_path}"
            except Exception as e:
                return f"Error reading CSV file: {e}"
        else:
            return "No path found in MySQL."
    except Exception as e:
        return f"Error retrieving path from MySQL: {e}"
    finally:
        cursor.close()
        connection.close()


# Function to store model details into MySQL
def store_model_to_mysql(model_name, model_path, num_layers, epochs):
    """
    Stores the model details into a MySQL database.

    Parameters:
    model_name (str): The name of the model.
    model_path (str): The file path where the model is saved.
    num_layers (int): The number of layers in the model.
    epochs (int): The number of epochs used during training.
    """
    connection = get_mysql_connection()
    if connection is None:
        return "Failed to connect to MySQL."

    cursor = connection.cursor()

    # Create the table if it does not already exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            model_name VARCHAR(255),
            model_path VARCHAR(255),
            trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            epochs INT,
            num_layers INT,
            accuracy FLOAT
        )
    """)

    # Insert model details into the table
    try:
        cursor.execute("""
            INSERT INTO model_history (model_name, model_path, trained_at, epochs, num_layers)
            VALUES (%s, %s, NOW(), %s, %s)
        """, (model_name, model_path, epochs, num_layers))
        connection.commit()
        return "Model details stored successfully in MySQL."
    except Exception as e:
        return f"Error storing model details to MySQL: {e}"
    finally:
        cursor.close()
        connection.close()