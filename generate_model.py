import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Fetch Bitcoin Data
def fetch_live_bitcoin_data(period='1y', interval='1d'):
    ticker = yf.Ticker("BTC-USD")
    data = ticker.history(period=period, interval=interval)
    data.reset_index(inplace=True)
    data = data[['Date', 'Close', 'Volume']]
    data.rename(columns={'Date': 'timestamp', 'Close': 'price', 'Volume': 'volume'}, inplace=True)
    return data

# Preprocess Data
def preprocess_data(data, look_back=60):
    data['moving_avg'] = data['price'].rolling(window=5).mean().fillna(method='bfill')
    scaler = MinMaxScaler(feature_range=(0, 1))
    data[['scaled_price', 'scaled_volume', 'scaled_moving_avg']] = scaler.fit_transform(
        data[['price', 'volume', 'moving_avg']]
    )
    features = data[['scaled_price', 'scaled_volume', 'scaled_moving_avg']].values
    X, y = [], []
    for i in range(look_back, len(features)):
        X.append(features[i-look_back:i])
        y.append(features[i, 0])  # Use price as the target
    return np.array(X), np.array(y), scaler

# Build LSTM Model
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

# Main Script
if __name__ == "__main__":
    # Fetch and preprocess data
    bitcoin_data = fetch_live_bitcoin_data(period='1y', interval='1d')
    look_back = 60
    X, y, scaler = preprocess_data(bitcoin_data, look_back)

    # Split into training and testing sets
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train the model
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    # Save the trained model
    model.save("lstm_model.h5")
    print("Model saved as lstm_model.h5")
