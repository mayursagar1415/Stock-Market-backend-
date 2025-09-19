from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
CORS(app)  # ✅ Enable CORS to allow frontend requests

# ✅ Define paths for model and scalers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "stock_prediction_model.pkl")
SCALER_X_PATH = os.path.join(BASE_DIR, "scaler_X.pkl")
SCALER_Y_PATH = os.path.join(BASE_DIR, "scaler_y.pkl")

# ✅ Load trained model and scalers
try:
    model = joblib.load(MODEL_PATH)
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    print("✅ Model and scalers loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model or scaler: {e}")
    raise

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ✅ Get input JSON data
        data = request.json

        # Validate all required fields and types
        required_fields = [
            "High", "Low", "Open", "Volume", "Year", "Month", "Day", "DayOfWeek",
            "50_MA", "200_MA", "Daily_Return", "Daily_Range", "Daily_Range_Pct",
            "Close_Lag_1", "Close_Lag_2", "Close_Lag_3", "Close_Lag_4", "Close_Lag_5"
        ]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400
            # Accept int, float, or string that can be converted to float
            value = data[field]
            try:
                float_value = float(value)
            except (TypeError, ValueError):
                return jsonify({"error": f"Invalid value for {field}: {value}"}), 400

        # ✅ Create a DataFrame with the input features
        features_df = pd.DataFrame({
            "High": [float(data["High"])],
            "Low": [float(data["Low"])],
            "Open": [float(data["Open"])],
            "Volume": [float(data["Volume"])],
            "Year": [float(data["Year"])],
            "Month": [float(data["Month"])],
            "Day": [float(data["Day"])],
            "DayOfWeek": [float(data["DayOfWeek"])],
            "50_MA": [float(data["50_MA"])],
            "200_MA": [float(data["200_MA"])],
            "Daily_Return": [float(data["Daily_Return"])],
            "Daily_Range": [float(data["Daily_Range"])],
            "Daily_Range_Pct": [float(data["Daily_Range_Pct"])],
            "Close_Lag_1": [float(data["Close_Lag_1"])],
            "Close_Lag_2": [float(data["Close_Lag_2"])],
            "Close_Lag_3": [float(data["Close_Lag_3"])],
            "Close_Lag_4": [float(data["Close_Lag_4"])],
            "Close_Lag_5": [float(data["Close_Lag_5"])],
        })

        # ✅ Scale input features
        features_scaled = scaler_X.transform(features_df)

        # ✅ Predict using the trained model
        prediction_scaled = model.predict(features_scaled).reshape(-1, 1)

        # ✅ Convert prediction back to actual price
        predicted_price = scaler_y.inverse_transform(prediction_scaled)[0][0]

        return jsonify({"predicted_price": float(predicted_price)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=5000)  # Ensure Flask runs on port 5000
