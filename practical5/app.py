from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

MODEL_PATH = Path("hotel_cancellation_model.pkl")

FEATURES = [
    "lead_time",
    "adults",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "previous_cancellations",
    "booking_changes",
    "adr",
    "total_of_special_requests",
    "required_car_parking_spaces",
    "hotel",
    "deposit_type",
    "market_segment",
]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "hotel_cancellation_model.pkl was not found. "
            "Run the training pipeline first."
        )
    return joblib.load(MODEL_PATH)


@app.get("/")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "hotel-cancellation-prediction"
    })


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "JSON request body is required"}), 400

    missing_fields = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    sample = pd.DataFrame([{
        feature: data[feature]
        for feature in FEATURES
    }])

    model = load_model()
    prediction_code = int(model.predict(sample)[0])
    probability = float(model.predict_proba(sample)[0][1])
    prediction = "CANCELED" if prediction_code == 1 else "NOT_CANCELED"

    return jsonify({
        "prediction": prediction,
        "prediction_code": prediction_code,
        "cancellation_probability": round(probability, 4)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
