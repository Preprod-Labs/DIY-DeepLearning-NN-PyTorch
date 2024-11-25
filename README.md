# DIY-Deep-Learning-NN-PyTorch

This is the **Customer Churn Prediction** branch.

## Customer Churn Prediction

### Business Case

A telecom company is facing a high rate of customer churn, where existing customers are leaving and switching to competitors. The company's goal is to predict which customers are most likely to churn so they can take targeted actions to retain them before they leave. This prediction can help reduce churn rates and improve customer retention efforts.

### Industry

Telecommunications

### Problem Statement

The company is losing customers to competitors. To mitigate this, they want to predict churn using historical data, including customer demographics, services subscribed, and billing information. By identifying patterns that lead to churn, the company can take proactive steps to improve customer satisfaction and retention rates.

### Objective

The aim is to build a predictive model to forecast customer churn based on key customer features such as:

- Demographics (e.g., age, gender)
- Services Subscribed (e.g., internet, phone, streaming services)
- Monthly Billing and Contract Information (e.g., monthly charges, contract type)

By using this predictive model, the telecom company can focus its retention efforts on high-risk customers, offering personalized incentives to encourage them to stay.

> To Learn more about this project using your favourite LLM, [click here](prompts.md)!

---

## Directory Structure

```plaintext
code/
├── __pycache__/                   (directory for compiled Python files)
├── saved_model/                   (directory for saved model files and training scripts)
│   ├── CNN.pkl                    (saved model for the Convolutional Neural Network)
│   ├── FCNN.pkl                   (saved model for the Fully Connected Neural Network)
│   ├── LSTM.pkl                   (saved model for the Long Short-Term Memory network)
│   ├── MLP.pkl                    (saved model for the Multi-Layer Perceptron)
│   ├── RNN.pkl                    (saved model for the Recurrent Neural Network)
│   ├── app.py                     (main application file for the Streamlit web app)
│   ├── classification.py          (script for classification-related functions and utilities)
│   ├── evaluate.py                (script to evaluate model performance on test data)
│   ├── ingest_transform_mongodb.py (script for ingesting and transforming data into MongoDB)
│   ├── ingest_transform.py        (script for general data ingestion and transformation)
│   ├── train_CNN.py               (script for training the Convolutional Neural Network)
│   ├── train_FCNN.py              (script for training the Fully Connected Neural Network)
│   ├── train_LSTM.py              (script for training the Long Short-Term Memory network)
│   ├── train_MLP.py               (script for training the Multi-Layer Perceptron)
│   └── train_RNN.py               (script for training the Recurrent Neural Network)
└── Data/
    └── Master/
        └── telecom_customer_data.csv (CSV file containing customer data for analysis)
.gitattributes                       (file for managing Git attributes)
.gitignore                          (specifies files and directories to be ignored by Git)
readme.md                           (documentation for the project)
requirements.txt                   (lists the dependencies required for the project)
```

---

## Data Definition

The dataset contains features like:

- Customer demographics (e.g., gender, age, region)
- Customer subscription data (e.g., services like phone, internet, etc.)
- Financial data (e.g., monthly charges, total charges)
- Tenure (e.g., length of subscription)
- Churn status (target variable indicating whether a customer has churned)

**Training and Testing Data:**

- **Training Samples:** 8000
- **Testing Samples:** 2000

---

## Program Flow

1. **Data Ingestion:** Load data from the `Data` directory (e.g., CSV files) and ingest it into a database (MongoDB or SQLite). [`ingest_transform_mongodb.py`, `ingest_transform.py`]
2. **Data Transformation:** Preprocess the data, including encoding categorical features, handling missing values, and normalizing numerical values. The data is then split into training, testing, and validation sets. [`ingest_transform.py`]
3. **Model Training:** Train a deep learning model (e.g., using TensorFlow or PyTorch) with techniques like cross-validation for hyperparameter tuning. [ `train_cnn.py`, `train_FCNN.py`, `train_LSTM.py`, `train_RNN.py`, `train_MLP.py`]
4. **Model Evaluation:** Evaluate the model's performance on the test and validation sets, generating classification reports and confusion matrices. [`evaluate.py`]
5. **Manual Prediction:** Allow users to manually input customer data for churn prediction using the CLI or API. [`classification.py`]
6. **Web Application:** A `Streamlit` app that integrates the entire pipeline and allows users to interact with the model, providing predictions based on customer input. [`app.py`]

---

## Database Setup

### MongoDB Setup

1. Download MongoDB:

   Windows:
   - Visit [MongoDB Download Center](https://www.mongodb.com/try/download/community)
   - Select your operating system and download the installer
   - Run the installer and follow the installation wizard

   Mac:
   ```bash
   # Install Homebrew if not already installed
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   
   # Install MongoDB Community Edition
   brew tap mongodb/brew
   brew install mongodb-community
   
   # Start MongoDB service
   brew services start mongodb-community
   ```

2. Start MongoDB:

   - Windows: MongoDB should start automatically as a service
   - Mac: Run `brew services start mongodb-community`
   - Linux: Run `sudo service mongod start`

3. Create Database:
   - Install MongoDB Compass (GUI tool) from [MongoDB Compass](https://www.mongodb.com/products/compass)
   - Open MongoDB Compass
   - Click "New Connection" and use: `mongodb://localhost:27017`
   - Click "Create Database"
   - Enter database name: `telecom_db`

The application will automatically create these collections:

- `config`: Stores configuration settings
- `model_history`: Stores model training data

### MySQL Setup

1. Download MySQL:

   Windows:
   - Visit [MySQL Downloads](https://dev.mysql.com/downloads/mysql/)
   - Choose your operating system
   - Download MySQL Community Server
   - Download MySQL Workbench (GUI tool)

   Mac:
   ```bash
   # Install MySQL using Homebrew
   brew install mysql
   
   # Start MySQL service
   brew services start mysql
   
   # Secure MySQL installation (follow prompts to set root password)
   mysql_secure_installation
   
   # Install MySQL Workbench (optional GUI tool)
   brew install --cask mysqlworkbench
   ```

2. Install MySQL:

   - Run the installer
   - Choose "Typical" installation
   - Make note of the root password you set during installation
   - Complete the installation

3. Set Up Database:

   - Open MySQL Workbench
   - Connect using credentials:
     ```
     Host: localhost
     User: root
     Password: <your-password>
     ```
   - Create database by running:
     ```sql
     CREATE DATABASE telecom_db;
     ```

4. Configure Application:
   - Open `ingest_transform.py` in your project
   - Update database settings:
     ```python
        host='localhost',
        user='root',
        password='your_password',
        database='telecom_db'
     ```

## Steps to Run

1. Clone the repository from GitHub:
   ```bash
   git clone https://github.com/Preprod-Labs/DIY-DeepLearning-NN-PyTorch
   ```
2. To create and activate a virtual environment (recommended) using Conda, run the following commands:
   ```
   conda create -n env_name python==3.10.11 -y
   conda activate env_name
   ```
   *Note: This project uses Python 3.10.11*

3. Install the dependencies using:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch the Streamlit web application:
   ```bash
   streamlit run code/app.py
   ```

5. Using the application:

   a) In the "Model Config" tab:
      - Select your preferred database (MongoDB or MySQL)
      - Enter the path to your data file, for example:
        ```
        Data/Master/telecom_customer_data.csv
        ```
      - You should see a preview of your data if the path is correct

   b) In the "Model Training" tab:
      - Train each model one by one (FCNN, CNN, RNN, LSTM, MLP)
      - For each model:
        1. Set the number of layers (recommended: start with 3)
        2. Set the number of epochs (recommended: start with 10)
        3. Click the "Train [Model] Model" button
        4. Wait for training to complete and view accuracy

   c) In the "Model Evaluation" tab:
      - View detailed metrics for each trained model
      - If you see "Please Run the Model First", go back to the Training tab and train that model

   d) In the "Model Prediction" tab:
      - Select your preferred algorithm (FCNN, CNN, RNN, MLP, or LSTM)
      - Fill in the customer details form
      - Click "Classify" to get the churn prediction

6. Common troubleshooting:
   - If database connection fails, ensure MongoDB/MySQL is running
   - If models fail to load, ensure you've trained them first
   - If data preview fails, double-check the file path

For additional help, refer to:
- MongoDB docs: https://docs.mongodb.com/manual/installation/
- MySQL docs: https://dev.mysql.com/doc/refman/8.0/en/installing.html