# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Code ownership rights: PreProd Corp
     
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

# Mapping dictionaries for categorical to numerical conversion
# These mappings ensure consistent encoding across the application
contract_mapping = {'One year': 0, 'Two year': 1, 'Month-to-month': 2}
internet_service_mapping = {'DSL': 0, 'Fiber optic': 1}
payment_mapping = {'Mailed check': 0, 'Bank transfer': 1, 'Credit card': 2, 'Electronic check': 3}
agree_mapping = {'Yes': 0, 'No': 1}

# Global scaler instance for maintaining consistent feature scaling
scaler = StandardScaler()


def get_mysql_connection():
    """
    Establishes and returns a connection to the MySQL database.
    
    The function handles connection configuration and error management for database operations.
    
    Returns:
        MySQLConnection: Active database connection if successful
        None: If connection fails
    
    Database Configuration:
        - host: localhost
        - user: root
        - database: telecom_db
    """
    try:
        # MySQL database configuration
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='enter your password here',# enter your sql db password here
            database='telecom_db'
        )
        return connection  # Return the connection object
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None  # Return None if connection fails

# Function to label categorical variables in the DataFrame
def labelling(df):
    """
    Converts categorical variables to numerical values using predefined mappings.
    
    This function handles the conversion of multiple categorical columns:
    - Contract types
    - Internet service types
    - Payment methods
    - Various service-related boolean fields
    
    Args:
        df (pandas.DataFrame): Raw input DataFrame with categorical values
    
    Returns:
        pandas.DataFrame: Processed DataFrame with numerical values
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
    Comprehensive data preprocessing pipeline for model training.
    
    Preprocessing steps:
    1. Remove unnecessary columns (e.g., CustomerID)
    2. Convert boolean columns to integers
    3. Normalize numeric columns to [0,1] range
    4. Convert categorical columns using mapping dictionaries
    5. Split data into training and testing sets
    6. Convert to PyTorch tensors
    
    Args:
        df (pandas.DataFrame): Raw input DataFrame
    
    Returns:
        tuple: (X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor)
    
    Raises:
        AssertionError: If any column values fall outside [0,1] range after normalization
    """
    df = df.copy()
    
    if 'CustomerID' in df.columns:
        df = df.drop(columns=['CustomerID'])
    
    # Convert boolean columns first
    bool_columns = ['SeniorCitizen', 'PaperlessBilling', 'Churn']
    for col in bool_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # Convert and normalize numeric columns
    numeric_columns = ['Tenure', 'MonthlyCharges', 'TotalCharges']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        max_val = df[col].max()
        min_val = df[col].min()
        if max_val != min_val:
            df[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            df[col] = 0
    
    # Handle categorical columns
    categorical_columns = {
        'Contract': contract_mapping,
        'InternetService': internet_service_mapping,
        'PaymentMethod': payment_mapping,
        'OnlineSecurity': agree_mapping,
        'TechSupport': agree_mapping,
        'StreamingTV': agree_mapping,
        'StreamingMovies': agree_mapping
    }
    
    for col, mapping in categorical_columns.items():
        # Convert categorical to numeric using mapping
        df[col] = df[col].map(mapping).fillna(0).astype(int)
        # Normalize to [0,1] range
        max_val = df[col].max()
        if max_val > 0:
            df[col] = df[col] / max_val
    
    # Verify all values are between 0 and 1
    for col in df.columns:
        if col != 'CustomerID':
            assert df[col].min() >= 0 and df[col].max() <= 1, f"Column {col} has values outside [0,1] range"
    
    # Split features and target
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train.values)
    X_test_tensor = torch.FloatTensor(X_test.values)
    y_train_tensor = torch.FloatTensor(y_train.values)
    y_test_tensor = torch.FloatTensor(y_test.values)
    
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

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



def preprocess_data_for_storage(df):
    """
    Preprocesses DataFrame to ensure data types match database schema.
    """
    df_processed = df.copy()
    
    # Convert categorical columns to proper format
    categorical_mappings = {
        'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
        'InternetService': {'DSL': 0, 'Fiber optic': 1, 'No': 2},
        'PaymentMethod': {'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3},
        'OnlineSecurity': {'No': 0, 'Yes': 1, 'No internet service': 2},
        'TechSupport': {'No': 0, 'Yes': 1, 'No internet service': 2},
        'StreamingTV': {'No': 0, 'Yes': 1, 'No internet service': 2},
        'StreamingMovies': {'No': 0, 'Yes': 1, 'No internet service': 2}
    }
    
    # Apply mappings to categorical columns
    for col, mapping in categorical_mappings.items():
        df_processed[col] = df_processed[col].map(mapping).fillna(-1).astype(int)
    
    # Convert boolean columns
    bool_columns = ['SeniorCitizen', 'PaperlessBilling', 'Churn']
    bool_mapping = {'Yes': 1, 'No': 0, True: 1, False: 0}
    for col in bool_columns:
        df_processed[col] = df_processed[col].map(bool_mapping).fillna(0).astype(int)
    
    # Clean and convert numeric columns
    df_processed['Tenure'] = pd.to_numeric(df_processed['Tenure'], errors='coerce').fillna(0).astype(int)
    df_processed['MonthlyCharges'] = pd.to_numeric(df_processed['MonthlyCharges'], errors='coerce').fillna(0).astype(float)
    df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'].replace(' ', '0'), errors='coerce').fillna(0).astype(float)
    
    return df_processed

def create_mysql_schema():
    """
    Creates the MySQL schema for telecom customer data.
    """
    return """
    CREATE TABLE IF NOT EXISTS customer_data (
        id INT AUTO_INCREMENT PRIMARY KEY,
        CustomerID VARCHAR(255) NOT NULL,
        Contract INT NOT NULL,
        InternetService INT NOT NULL,
        PaymentMethod INT NOT NULL,
        OnlineSecurity INT NOT NULL,
        TechSupport INT NOT NULL,
        StreamingTV INT NOT NULL,
        StreamingMovies INT NOT NULL,
        Tenure INT NOT NULL,
        MonthlyCharges DECIMAL(10,2) NOT NULL,
        TotalCharges DECIMAL(10,2) NOT NULL,
        SeniorCitizen TINYINT(1) NOT NULL,
        PaperlessBilling TINYINT(1) NOT NULL,
        Churn TINYINT(1) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """

def store_data_to_mysql(df):
    """
    Persists customer data to MySQL database with proper data type handling.
    
    Process flow:
    1. Establishes database connection
    2. Creates/recreates table schema
    3. Preprocesses data for storage
    4. Performs bulk insert operation
    5. Verifies data insertion
    
    Args:
        df (pandas.DataFrame): Customer data to store
    
    Returns:
        str: Success/failure message with row count
    
    Error handling:
    - Catches and logs database connectivity issues
    - Handles data type conversion errors
    - Ensures proper connection cleanup
    """
    connection = get_mysql_connection()
    if connection is None:
        return "Failed to connect to MySQL."

    cursor = connection.cursor()

    try:
        # Drop existing table and create new one
        cursor.execute("DROP TABLE IF EXISTS customer_data")
        cursor.execute(create_mysql_schema())
        connection.commit()
        
        # Preprocess data
        df_processed = preprocess_data_for_storage(df)
        
        # Prepare insert query
        insert_sql = """
        INSERT INTO customer_data (
            CustomerID, Contract, InternetService, PaymentMethod,
            OnlineSecurity, TechSupport, StreamingTV, StreamingMovies,
            Tenure, MonthlyCharges, TotalCharges, SeniorCitizen,
            PaperlessBilling, Churn
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        # Convert DataFrame to list of tuples
        values = df_processed[['CustomerID', 'Contract', 'InternetService', 'PaymentMethod',
                             'OnlineSecurity', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                             'Tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen',
                             'PaperlessBilling', 'Churn']].values.tolist()
        
        # Execute bulk insert
        cursor.executemany(insert_sql, values)
        connection.commit()
        
        # Verify insertion
        cursor.execute("SELECT COUNT(*) FROM customer_data")
        count = cursor.fetchone()[0]
        return f"Data stored successfully in MySQL. {count} rows inserted."
        
    except Exception as e:
        print(f"Detailed error: {str(e)}")  # Add detailed error printing
        return f"Error storing data to MySQL: {e}"
    finally:
        cursor.close()
        connection.close()

def retrieve_data_from_mysql():
    """
    Retrieves raw data from MySQL without any preprocessing.
    """
    connection = get_mysql_connection()
    if connection is None:
        return None

    try:
        query = """
        SELECT CustomerID, Contract, InternetService, PaymentMethod,
               OnlineSecurity, TechSupport, StreamingTV, StreamingMovies,
               Tenure, MonthlyCharges, TotalCharges, SeniorCitizen,
               PaperlessBilling, Churn 
        FROM customer_data
        """
        
        df = pd.read_sql(query, connection)
        
        if df.empty:
            print("Warning: No data retrieved from MySQL")
            return None
            
        print(f"Retrieved {len(df)} rows from MySQL")
        return df

    except Exception as e:
        print(f"Error retrieving data from MySQL: {e}")
        return None
    finally:
        connection.close()

# Function to store model details into MySQL
def store_model_to_mysql(model_name, model_path, num_layers, epochs):
    """
    Records model metadata in MySQL for model versioning and tracking.
    
    Stored information includes:
    - Model name and file path
    - Training timestamp
    - Architecture details (number of layers)
    - Training parameters (epochs)
    
    Args:
        model_name (str): Identifier for the model
        model_path (str): Storage location of model file
        num_layers (int): Number of neural network layers
        epochs (int): Training iterations completed
    
    Returns:
        str: Success/failure message
    
    Table Schema:
        - id: Auto-incrementing primary key
        - model_name: Model identifier
        - model_path: File system location
        - trained_at: Timestamp of training completion
        - epochs: Number of training iterations
        - num_layers: Model architecture detail
        - accuracy: Model performance metric
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
        connection.close