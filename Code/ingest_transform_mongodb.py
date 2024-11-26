# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Developer details: 
        # Name: Shubh Gupta
        # Role: Developers
        # Code ownership rights: PreProd Corp
    # Version:
        # Version: V 1.1 (19 October 2024)
            # Developers: Shubh Gupta
            # Unit test: Pass
            # Integration test: Pass
     
    # Description: This script handles data ingestion, preprocessing, and MongoDB database operations for the churn prediction system.
        # MySQL: Yes
        # MongoDB: Yes

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Dependency: 
    # Environment:     
        # Python 3.10.11
        # pymongo 4.9.1
        # pandas 1.5.3
from pymongo import MongoClient  # Import MongoClient to interact with MongoDB
import pandas as pd  # Pandas for handling CSV data

# Connect to MongoDB (local instance)
def get_mongo_client():
    """ 
    Connect to local MongoDB.
    """
    return MongoClient("mongodb://localhost:27017/")  # Local MongoDB URI


# Function to store the master data path in MongoDB
def store_path_to_mongodb(path):
    """
    Stores the master data path in a MongoDB collection.

    Parameters:
    path (str): The file path to be stored.
    """
    try:
        # Connect to the MongoDB client
        client = get_mongo_client()
        db = client["telecom_db"]  # Use or create the database
        config_collection = db["config"]  # Use or create the collection for storing configurations

        # Insert the path into the collection
        config_collection.insert_one({
            "master_data_path": path
        })

        # Close the client connection
        client.close()

        return "Path stored successfully in MongoDB."
    except Exception as e:
        return f"Error storing path to MongoDB: {e}"


# Function to retrieve the master data path from MongoDB
def retrieve_path_from_mongodb():
    """
    Retrieves the master data path from the MongoDB collection.

    Returns:
    DataFrame or None: The DataFrame read from the CSV file if successful; otherwise, None.
    """
    try:
        # Connect to the MongoDB client
        client = get_mongo_client()
        db = client["telecom_db"]  # Access the churn prediction database
        config_collection = db["config"]  # Access the config collection

        # Retrieve the latest master data path (sorted by insertion order)
        latest_record = config_collection.find_one({}, sort=[('_id', -1)])

        client.close()  # Close the connection

        if (latest_record):
            csv_path = latest_record["master_data_path"]
            try:
                # Load the CSV using pandas
                df = pd.read_csv(csv_path)
                return df
            except FileNotFoundError:
                return f"File not found at path: {csv_path}"
            except pd.errors.EmptyDataError:
                return f"No data found in the file at path: {csv_path}"
            except Exception as e:
                return f"Error reading CSV file: {e}"
        else:
            return "No path found in MongoDB."
    except Exception as e:
        return f"Error retrieving path from MongoDB: {e}"
    
from pymongo import MongoClient  # Import MongoClient to interact with MongoDB
from datetime import datetime  # Import datetime to get the current timestamp

# Function to store model details into MongoDB
def store_model_to_mongodb(model_name, model_path, num_layers, epochs):
    """
    Stores the model details into a MongoDB collection.

    Parameters:
    model_name (str): The name of the model.
    model_path (str): The file path where the model is saved.
    num_layers (int): The number of layers in the model.
    epochs (int): The number of epochs used during training.
    """
    # Connect to MongoDB (local instance)
    client = MongoClient("mongodb://localhost:27017/")
    db = client["telecom_db"]  # Use or create the database
    model_collection = db["model_history"]  # Use or create the collection for model history

    # Create a document with model details
    model_details = {
        "model_name": model_name,
        "model_path": model_path,
        "trained_at": datetime.now(),  # Current timestamp
        "epochs": epochs,
        "num_layers": num_layers,
        "accuracy": None  # Placeholder for accuracy; you can update this later
    }

    # Insert model details into the collection
    try:
        model_collection.insert_one(model_details)
        return "Model details stored successfully in MongoDB."
    except Exception as e:
        return f"Error storing model details to MongoDB: {e}"
    finally:
        client.close()  # Ensure the client connection is closed

def store_data_to_mongodb(df):
    """
    Stores the entire DataFrame in MongoDB.
    """
    try:
        client = get_mongo_client()
        db = client["telecom_db"]
        collection = db["customer_data"]

        # Convert DataFrame to dictionary records
        records = df.to_dict('records')

        # Insert data
        collection.delete_many({})  # Clear existing data
        collection.insert_many(records)
        
        client.close()
        return "Data stored successfully in MongoDB."
    except Exception as e:
        return f"Error storing data to MongoDB: {e}"

def retrieve_data_from_mongodb():
    """
    Retrieves the entire dataset from MongoDB.
    """
    try:
        client = get_mongo_client()
        db = client["telecom_db"]
        collection = db["customer_data"]

        # Retrieve all documents
        data = list(collection.find({}, {'_id': 0}))
        
        client.close()
        
        if data:
            df = pd.DataFrame(data)
            return df
        return None
    except Exception as e:
        print(f"Error retrieving data from MongoDB: {e}")
        return None
