from flask import Flask, request, jsonify, render_template
import subprocess
import yfinance as yf
import pandas as pd
import numpy as np
import mlflow.keras
import joblib
import os

# Initialize Flask app
app = Flask(__name__)

# Constants
LOOK_BACK = 60

mlflow.set_tracking_uri("http://127.0.0.1:5001")

# Load pre-trained model from MLflow
MLFLOW_MODEL_URI = "models:/BitcoinPricePredictor/latest"  # Always fetch the latest version
model = mlflow.keras.load_model(MLFLOW_MODEL_URI)

# Load scaler from MLflow artifacts
SCALER_PATH = mlflow.artifacts.download_artifacts("scaler.pkl")
scaler = joblib.load(SCALER_PATH)

# Function to fetch Bitcoin data
def fetch_live_bitcoin_data(period="1y", interval="1d"):
    ticker = yf.Ticker("BTC-USD")
    data = ticker.history(period=period, interval=interval)
    data.reset_index(inplace=True)
    data = data[["Date", "Close", "Volume"]]
    data.rename(columns={"Date": "timestamp", "Close": "price", "Volume": "volume"}, inplace=True)
    return data

# Function to preprocess data
def preprocess_data(data):
    data["moving_avg"] = data["price"].rolling(window=5).mean().fillna(method="bfill")
    data[["scaled_price", "scaled_volume", "scaled_moving_avg"]] = scaler.transform(
        data[["price", "volume", "moving_avg"]]
    )
    return data

# Function to predict the next 7 days
def predict_next_7_days(model, data):
    predictions = []
    input_seq = data[["scaled_price", "scaled_volume", "scaled_moving_avg"]].values[-LOOK_BACK:]
    input_seq = input_seq.reshape(1, input_seq.shape[0], input_seq.shape[1])

    for _ in range(7):
        scaled_prediction = model.predict(input_seq)
        prediction = scaler.inverse_transform([[scaled_prediction[0][0], 0, 0]])[0, 0]
        predictions.append(prediction)

        new_input = np.array([scaled_prediction[0][0], data["scaled_volume"].iloc[-1], data["scaled_moving_avg"].iloc[-1]])
        input_seq = np.append(input_seq[0, 1:], [new_input], axis=0).reshape(1, -1, 3)

    return predictions

@app.route("/")
def home():
    return "Welcome to the Bitcoin Price Prediction API!"

@app.route("/predict", methods=["GET"])
def predict():
    try:
        period = request.args.get("period", "1y")
        interval = request.args.get("interval", "1d")
        bitcoin_data = fetch_live_bitcoin_data(period=period, interval=interval)
        bitcoin_data = preprocess_data(bitcoin_data)

        predictions = predict_next_7_days(model, bitcoin_data)
        return jsonify({"predictions": predictions})

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/chart", methods=["GET"])
def chart():
    try:
        # Fetch historical data
        bitcoin_data = fetch_live_bitcoin_data(period="1y", interval="1d")

        # Prepare chart data
        dates = bitcoin_data["timestamp"].astype(str).tolist()
        prices = bitcoin_data["price"].tolist()

        return render_template("chart.html", dates=dates, prices=prices)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/generate", methods=["GET"])
def generate_model():
    """
    Endpoint to trigger the model generation pipeline.
    """
    try:
        # Run the model generation script as a subprocess
        subprocess.run(["python", "generate_model.py"], check=True)
        return jsonify({"status": "Model generation pipeline executed successfully!"})
    except subprocess.CalledProcessError as e:
        return jsonify({"error": str(e)})
@app.route("/predict_version", methods=["GET"])
def predict_version():
    """
    Predict using a specific version of the Bitcoin price prediction model.
    """
    try:
        # Get version from query parameters
        version = request.args.get("version")
        if not version:
            return jsonify({"error": "Model version is required!"}), 400
        
        # Load the specified version of the model from MLflow
        specific_model_uri = f"models:/BitcoinPricePredictor/{version}"
        specific_model = mlflow.keras.load_model(specific_model_uri)
        
        # Use the same scaler (already loaded) or load a version-specific scaler
        # For this implementation, assuming scaler compatibility

        # Fetch and preprocess live Bitcoin data
        period = request.args.get("period", "1y")
        interval = request.args.get("interval", "1d")
        bitcoin_data = fetch_live_bitcoin_data(period=period, interval=interval)
        bitcoin_data = preprocess_data(bitcoin_data)

        # Make predictions using the specific model
        predictions = predict_next_7_days(specific_model, bitcoin_data)
        return jsonify({"version": version, "predictions": predictions})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, port=5005)
