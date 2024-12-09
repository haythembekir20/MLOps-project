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

# Function to fetch Bitcoin data
def fetch_live_bitcoin_data(period='1y', interval='1d'):
    ticker = yf.Ticker("BTC-USD")
    data = ticker.history(period=period, interval=interval)
    data.reset_index(inplace=True)
    data = data[['Date', 'Close', 'Volume']]
    data.rename(columns={'Date': 'timestamp', 'Close': 'price', 'Volume': 'volume'}, inplace=True)
    return data

# Function to preprocess data
def preprocess_data(data, look_back=60):
    data['moving_avg'] = data['price'].rolling(window=5).mean().fillna(method='bfill')
    scaler = MinMaxScaler(feature_range=(0, 1))
    data[['scaled_price', 'scaled_volume', 'scaled_moving_avg']] = scaler.fit_transform(
        data[['price', 'volume', 'moving_avg']]
    )
    features = data[['scaled_price', 'scaled_volume', 'scaled_moving_avg']].values
    X, y = [], []
    for i in range(look_back, len(features)):
        X.append(features[i - look_back:i])
        y.append(features[i, 0])  # Use scaled price as the target
    return np.array(X), np.array(y), scaler

# Function to build the LSTM model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(100, return_sequences=False),
        Dropout(0.2),
        Dense(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Main script with MLflow integration
if __name__ == "__main__":
    # Start an MLflow experiment
    with mlflow.start_run():
        # Fetch and preprocess data
        bitcoin_data = fetch_live_bitcoin_data(period='1y', interval='1d')
        look_back = 60
        X, y, scaler = preprocess_data(bitcoin_data, look_back)

        # Split data into training and testing sets
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Log parameters
        mlflow.log_param("look_back", look_back)
        mlflow.log_param("epochs", 20)
        mlflow.log_param("batch_size", 32)

        # Build and train the model
        model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
        history = model.fit(
            X_train, y_train,
            epochs=20,
            batch_size=32,
            validation_data=(X_test, y_test)
        )

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

        # Log metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)

        # Save the scaler as an artifact
        scaler_path = "scaler.pkl"
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path)

        # Log the trained model
        mlflow.keras.log_model(model, "model")

        print(f"Model training complete. MAE: {mae}, RMSE: {rmse}")
