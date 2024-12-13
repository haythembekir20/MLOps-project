from flask import Flask, request, jsonify, render_template
import subprocess
import yfinance as yf
import pandas as pd
import numpy as np
import mlflow.keras
import joblib
import os
from flasgger import Swagger

# Initialize Flask app
app = Flask(__name__)
swagger = Swagger(app)

# Constants
LOOK_BACK = 60  # Number of previous time steps to use for prediction

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5001")

# Load pre-trained model from MLflow
MLFLOW_MODEL_URI = "models:/BitcoinPricePredictor/latest"  # Always fetch the latest version
model = mlflow.keras.load_model(MLFLOW_MODEL_URI)

# Load scaler from MLflow artifacts
SCALER_PATH = mlflow.artifacts.download_artifacts("scaler.pkl")
scaler = joblib.load(SCALER_PATH)

def fetch_live_bitcoin_data(period="1y", interval="1d"):
    """
    Fetch live Bitcoin data from Yahoo Finance using the yfinance library.

    Args:
        period (str): Historical period to fetch data (e.g., "1y" for 1 year).
        interval (str): Interval for data points (e.g., "1d" for daily data).

    Returns:
        pd.DataFrame: DataFrame containing the Bitcoin price and volume.
    """
    ticker = yf.Ticker("BTC-USD")
    data = ticker.history(period=period, interval=interval)
    data.reset_index(inplace=True)
    data = data[["Date", "Close", "Volume"]]
    data.rename(columns={"Date": "timestamp", "Close": "price", "Volume": "volume"}, inplace=True)
    return data

def preprocess_data(data):
    """
    Preprocess Bitcoin data by scaling it and calculating a moving average.

    Args:
        data (pd.DataFrame): The raw Bitcoin data.

    Returns:
        pd.DataFrame: The preprocessed data with scaled features.
    """
    data["moving_avg"] = data["price"].rolling(window=5).mean().fillna(method="bfill")
    data[["scaled_price", "scaled_volume", "scaled_moving_avg"]] = scaler.transform(
        data[["price", "volume", "moving_avg"]]
    )
    return data

def predict_next_7_days(model, data):
    """
    Predict Bitcoin prices for the next 7 days using the trained model.

    Args:
        model (keras.Model): The trained Keras model.
        data (pd.DataFrame): The preprocessed Bitcoin data.

    Returns:
        list: A list of predicted Bitcoin prices for the next 7 days.
    """
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
    """
    Home route for the Flask app.

    Returns:
        str: A welcome message.
    ---
    responses:
      200:
        description: Welcome message
    """
    return "Welcome to the Bitcoin Price Prediction API!"

@app.route("/predict", methods=["GET"])
def predict():
    """
    Predict Bitcoin prices using the latest model.

    ---
    parameters:
      - name: period
        in: query
        type: string
        default: "1y"
        required: false
        description: Historical period to fetch data.
      - name: interval
        in: query
        type: string
        default: "1d"
        required: false
        description: Interval for data points.
    responses:
      200:
        description: A JSON with predictions for the next 7 days.
        schema:
          type: object
          properties:
            predictions:
              type: array
              items:
                type: number
      500:
        description: An error message.
    """
    try:
        period = request.args.get("period", "1y")
        interval = request.args.get("interval", "1d")
        bitcoin_data = fetch_live_bitcoin_data(period=period, interval=interval)
        bitcoin_data = preprocess_data(bitcoin_data)
        predictions = predict_next_7_days(model, bitcoin_data)
        return jsonify({"predictions": predictions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/chart", methods=["GET"])
def chart():
    """
    Display a chart of historical Bitcoin prices.

    ---
    parameters:
      - name: period
        in: query
        type: string
        default: "1y"
        required: false
        description: Historical period to fetch data.
      - name: interval
        in: query
        type: string
        default: "1d"
        required: false
        description: Interval for data points.
    responses:
      200:
        description: An HTML page displaying a chart.
      500:
        description: An error message.
    """
    try:
        bitcoin_data = fetch_live_bitcoin_data(period="1y", interval="1d")
        dates = bitcoin_data["timestamp"].astype(str).tolist()
        prices = bitcoin_data["price"].tolist()
        return render_template("chart.html", dates=dates, prices=prices)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/generate", methods=["GET"])
def generate_model():
    """
    Trigger the model generation pipeline.

    ---
    responses:
      200:
        description: Status of the model generation pipeline.
      500:
        description: An error message.
    """
    try:
        subprocess.run(["python", "generate_model.py"], check=True)
        return jsonify({"status": "Model generation pipeline executed successfully!"})
    except subprocess.CalledProcessError as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict_version", methods=["GET"])
def predict_version():
    """
    Predict Bitcoin prices using a specific version of the model.

    ---
    parameters:
      - name: version
        in: query
        type: string
        required: true
        description: The version of the model to use.
      - name: period
        in: query
        type: string
        default: "1y"
        required: false
        description: Historical period to fetch data.
      - name: interval
        in: query
        type: string
        default: "1d"
        required: false
        description: Interval for data points.
    responses:
      200:
        description: A JSON with predictions for the next 7 days.
        schema:
          type: object
          properties:
            version:
              type: string
            predictions:
              type: array
              items:
                type: number
      400:
        description: Missing version parameter.
      500:
        description: An error message.
    """
    try:
        version = request.args.get("version")
        if not version:
            return jsonify({"error": "Model version is required!"}), 400
        specific_model_uri = f"models:/BitcoinPricePredictor/{version}"
        specific_model = mlflow.keras.load_model(specific_model_uri)
        period = request.args.get("period", "1y")
        interval = request.args.get("interval", "1d")
        bitcoin_data = fetch_live_bitcoin_data(period=period, interval=interval)
        bitcoin_data = preprocess_data(bitcoin_data)
        predictions = predict_next_7_days(specific_model, bitcoin_data)
        return jsonify({"version": version, "predictions": predictions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5005)
