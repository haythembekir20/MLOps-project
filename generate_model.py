"""
Bitcoin Price Prediction Pipeline using Prefect, MLflow, and Keras

This script trains an LSTM-based deep learning model to predict Bitcoin prices.
It includes the following stages:
1. Fetching live Bitcoin data using yfinance.
2. Preprocessing the data (e.g., scaling and feature engineering).
3. Building an LSTM neural network model.
4. Training the model and logging results to MLflow.
5. Registering the trained model in the MLflow Model Registry.

Key Features:
- Orchestrated with Prefect for error handling, retries, and task monitoring.
- Integrated with MLflow for model tracking and management.
- Modular, reusable, and extensible design for MLOps workflows.

Author: [Your Name]
Date: [Today's Date]
"""

from prefect import flow, task
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import mlflow
import mlflow.keras
import joblib
import time
import logging

# Constants
LOOK_BACK = 60  # Number of previous days used for prediction
EPOCHS = 20  # Number of training epochs
BATCH_SIZE = 32  # Batch size for model training
REGISTERED_MODEL_NAME = "BitcoinPricePredictor"  # MLflow registered model name

# Task: Fetch Bitcoin data
@task(retries=3, retry_delay_seconds=10)
def fetch_live_bitcoin_data(period="1y", interval="1d"):
    """
    Fetch live Bitcoin data using yfinance.
    
    Parameters:
    - period: The historical period to fetch (default: 1 year).
    - interval: The data interval (e.g., daily).

    Returns:
    - A dictionary of Bitcoin data including timestamp, price, and volume.
    """
    try:
        ticker = yf.Ticker("BTC-USD")
        data = ticker.history(period=period, interval=interval)
        data.reset_index(inplace=True)
        data = data[['Date', 'Close', 'Volume']]
        data.rename(columns={"Date": "timestamp", "Close": "price", "Volume": "volume"}, inplace=True)
        logging.info("Fetched Bitcoin data successfully.")
        return data.to_dict()
    except Exception as e:
        logging.error(f"Error fetching Bitcoin data: {e}")
        raise e

# Task: Preprocess data
@task(retries=2, retry_delay_seconds=5)
def preprocess_data(data, look_back=LOOK_BACK):
    """
    Preprocess raw Bitcoin data for LSTM training.

    Steps:
    - Compute a 5-day moving average.
    - Scale price, volume, and moving average to a range of 0-1 using MinMaxScaler.
    - Generate input sequences (X) and target values (y) for time-series prediction.

    Parameters:
    - data: Raw Bitcoin data as a dictionary.
    - look_back: Number of days to look back for prediction.

    Returns:
    - A tuple (X, y, scaler), where:
        - X: Input sequences for the model.
        - y: Target values.
        - scaler: Fitted MinMaxScaler object.
    """
    try:
        data = pd.DataFrame.from_dict(data)
        data['moving_avg'] = data['price'].rolling(window=5).mean().fillna(method='bfill')
        scaler = MinMaxScaler(feature_range=(0, 1))
        data[['scaled_price', 'scaled_volume', 'scaled_moving_avg']] = scaler.fit_transform(
            data[['price', 'volume', 'moving_avg']]
        )
        features = data[['scaled_price', 'scaled_volume', 'scaled_moving_avg']].values
        X, y = [], []
        for i in range(look_back, len(features)):
            X.append(features[i - look_back:i])
            y.append(features[i, 0])
        logging.info("Data preprocessing completed successfully.")
        return (np.array(X), np.array(y), scaler)
    except Exception as e:
        logging.error(f"Error during data preprocessing: {e}")
        raise e

# Task: Build the LSTM model
@task
def build_lstm_model(input_shape):
    """
    Build an LSTM-based deep learning model.

    Parameters:
    - input_shape: Shape of the input data (look_back, number of features).

    Returns:
    - A compiled LSTM model.
    """
    try:
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(100, return_sequences=False),
            Dropout(0.2),
            Dense(50),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mean_squared_error")
        logging.info("LSTM model built successfully.")
        return model
    except Exception as e:
        logging.error(f"Error building the LSTM model: {e}")
        raise e

# Task: Train the model
@task(retries=1, retry_delay_seconds=10)
def train_model(data, model):
    """
    Train the LSTM model on preprocessed data and log results to MLflow.

    Steps:
    - Split data into training and testing sets.
    - Train the model and log parameters, metrics, and artifacts to MLflow.
    - Calculate accuracy of prediction trends (up/down).
    - Register the trained model in MLflow.

    Parameters:
    - data: A tuple (X, y, scaler) from the preprocess_data task.
    - model: Compiled LSTM model.

    Returns:
    - The trained model.
    """
    try:
        X, y, scaler = data
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        with mlflow.start_run() as run:
            # Log parameters
            mlflow.log_param("look_back", LOOK_BACK)
            mlflow.log_param("epochs", EPOCHS)
            mlflow.log_param("batch_size", BATCH_SIZE)

            # Start timer
            start_time = time.time()

            # Train the model
            history = model.fit(
                X_train, y_train,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_data=(X_test, y_test)
            )

            # Record training time
            training_time = time.time() - start_time
            mlflow.log_metric("training_time_seconds", training_time)

            # Evaluate the model
            predicted_test = model.predict(X_test)
            predicted_test_rescaled = scaler.inverse_transform(
                np.column_stack([predicted_test, np.zeros((len(predicted_test), 2))])
            )[:, 0]
            actual_test_rescaled = scaler.inverse_transform(
                np.column_stack([y_test, np.zeros((len(y_test), 2))])
            )[:, 0]

            mae = mean_absolute_error(actual_test_rescaled, predicted_test_rescaled)
            rmse = np.sqrt(mean_squared_error(actual_test_rescaled, predicted_test_rescaled))

            # Calculate accuracy of trend predictions
            actual_trend = np.sign(np.diff(actual_test_rescaled))  # Actual trend (+1 or -1)
            predicted_trend = np.sign(np.diff(predicted_test_rescaled))  # Predicted trend (+1 or -1)
            accuracy = np.mean(actual_trend == predicted_trend) * 100  # Percentage of correct trends

            # Log metrics
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
           # mlflow.log_metric("trend_accuracy", accuracy)

            # Save and log scaler
            scaler_path = "scaler.pkl"
            joblib.dump(scaler, scaler_path)
            mlflow.log_artifact(scaler_path)

            # Log and register the model
            mlflow.keras.log_model(model, "model", registered_model_name=REGISTERED_MODEL_NAME)

            logging.info(f"Model training complete. MAE: {mae}, RMSE: {rmse}, Trend Accuracy: {accuracy:.2f}%")
        return model
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise e

# Prefect Flow
@flow(name="Bitcoin Price Prediction Pipeline")
def bitcoin_price_prediction_pipeline():
    """
    Prefect flow for orchestrating the Bitcoin price prediction pipeline.

    Steps:
    1. Fetch Bitcoin data.
    2. Preprocess the data.
    3. Build the LSTM model.
    4. Train the model and log results to MLflow.

    Executes all tasks with built-in retries and monitoring.
    """
    try:
        # Fetch data
        raw_data = fetch_live_bitcoin_data()
        # Preprocess data
        processed_data = preprocess_data(raw_data)
        # Build model
        model = build_lstm_model(input_shape=(LOOK_BACK, 3))
        # Train model
        train_model(processed_data, model)
        logging.info("Pipeline executed successfully.")
    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        raise e

# Run the flow
if __name__ == "__main__":
    bitcoin_price_prediction_pipeline()
