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
import os
import logging

# Constants
LOOK_BACK = 60  # Number of previous days used for prediction
EPOCHS = 20  # Number of training epochs
BATCH_SIZE = 32  # Batch size for model training
REGISTERED_MODEL_NAME = "BitcoinPricePredictor"  # MLflow registered model name

# Configure MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5001")  # Update with your MLflow server URI
mlflow.set_experiment("Bitcoin Price Prediction")

# Task: Fetch Bitcoin data
@task(retries=3, retry_delay_seconds=10)
def fetch_live_bitcoin_data(period="1y", interval="1d"):
    """
    Fetch live Bitcoin data using yfinance and log it as an MLflow artifact.
    """
    try:
        ticker = yf.Ticker("BTC-USD")
        data = ticker.history(period=period, interval=interval)
        data.reset_index(inplace=True)
        data = data[['Date', 'Close', 'Volume']]
        data.rename(columns={"Date": "timestamp", "Close": "price", "Volume": "volume"}, inplace=True)

        # Ensure the data directory exists
        os.makedirs("data", exist_ok=True)

        # Save and log data as an artifact
        file_path = "data/bitcoin_prices.csv"
        data.to_csv(file_path, index=False)
        mlflow.log_artifact(file_path)

        # Log details in MLflow
        mlflow.log_param("data_period", period)
        mlflow.log_param("data_interval", interval)
        mlflow.log_metric("data_rows", len(data))
        mlflow.log_metric("data_columns", len(data.columns))

        logging.info("Fetched and logged Bitcoin data successfully.")
        return data.to_dict()
    except Exception as e:
        logging.error(f"Error fetching Bitcoin data: {e}")
        raise e

# Task: Preprocess data
@task(retries=2, retry_delay_seconds=5)
def preprocess_data(data, look_back=LOOK_BACK):
    """
    Preprocess raw Bitcoin data and log the processed data to MLflow.
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

        # Save and log preprocessed data
        file_path = "data/preprocessed_data.pkl"
        joblib.dump((X, y, scaler), file_path)
        mlflow.log_artifact(file_path)

        # Log details in MLflow
        mlflow.log_param("look_back", look_back)
        mlflow.log_metric("training_samples", len(X))
        mlflow.log_metric("training_features", features.shape[1])

        logging.info("Data preprocessing completed successfully.")
        return np.array(X), np.array(y), scaler
    except Exception as e:
        logging.error(f"Error during data preprocessing: {e}")
        raise e

# Task: Build the LSTM model
@task
def build_lstm_model(input_shape):
    """
    Build an LSTM-based deep learning model.
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
        
        # Log details in MLflow
        mlflow.log_param("lstm_units", 100)
        mlflow.log_param("dropout_rate", 0.2)
        mlflow.log_param("dense_units", 50)
        mlflow.log_param("optimizer", "adam")
        mlflow.log_param("loss_function", "mean_squared_error")

        logging.info("LSTM model built successfully.")
        return model
    except Exception as e:
        logging.error(f"Error building the LSTM model: {e}")
        raise e

# Task: Train the model
@task(retries=1, retry_delay_seconds=10)
def train_model(data, model):
    """
    Train the LSTM model and log results to MLflow.
    """
    try:
        X, y, scaler = data
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_test, y_test)
        )

        # Evaluate the model
        predicted_test = model.predict(X_test)
        mae = mean_absolute_error(y_test, predicted_test)
        rmse = np.sqrt(mean_squared_error(y_test, predicted_test))

          # Calculate custom accuracy
        mean_actual = np.mean(y_test)
        accuracy = (1 - (mae / mean_actual)) * 100  # Accuracy as a percentage

        # Log details in MLflow
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("accuracy", accuracy)  # Log accuracy
        mlflow.log_param("epochs", EPOCHS)

        mlflow.log_metric("training_data_split_ratio", 0.8)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_metric("validation_loss", history.history['val_loss'][-1])

        # Log the model
        mlflow.keras.log_model(model, "model", registered_model_name=REGISTERED_MODEL_NAME)

        logging.info(f"Model training completed. MAE: {mae}, RMSE: {rmse}")
        return model
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise e

# Prefect Flow
@flow(name="Bitcoin Price Prediction Pipeline")
def bitcoin_price_prediction_pipeline():
    """
    Prefect flow for orchestrating the Bitcoin price prediction pipeline.

    This flow includes fetching data, preprocessing it, building an LSTM model,
    training the model, and logging results to MLflow.
    """
    try:
        # Start a single MLflow run for the entire pipeline
        with mlflow.start_run(run_name="Bitcoin_Price_Prediction_Pipeline"):
            raw_data = fetch_live_bitcoin_data()
            processed_data = preprocess_data(raw_data)
            model = build_lstm_model(input_shape=(LOOK_BACK, 3))
            train_model(processed_data, model)

            # Additional pipeline-level metrics
            mlflow.log_param("total_pipeline_steps", 4)
            logging.info("Pipeline executed successfully.")
    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        raise e

if __name__ == "__main__":
    bitcoin_price_prediction_pipeline()
