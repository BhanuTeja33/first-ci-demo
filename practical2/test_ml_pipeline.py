import json
import os
import unittest
import joblib
import pandas as pd


def make_sample(**overrides):
    sample = {
        "lead_time": 50,
        "adults": 2,
        "stays_in_weekend_nights": 1,
        "stays_in_week_nights": 2,
        "previous_cancellations": 0,
        "booking_changes": 0,
        "adr": 100.0,
        "total_of_special_requests": 1,
        "required_car_parking_spaces": 0,
        "hotel": "City Hotel",
        "deposit_type": "No Deposit",
        "market_segment": "Online TA",
    }
    sample.update(overrides)
    return pd.DataFrame([sample])


class TestMLPipeline(unittest.TestCase):

    def test_dataset_exists(self):
        self.assertTrue(os.path.exists("data/Hotel_Bookings.csv"))

    def test_model_created(self):
        self.assertTrue(os.path.exists("hotel_cancellation_model.pkl"))

    def test_metrics_created(self):
        self.assertTrue(os.path.exists("metrics.json"))

    def test_accuracy_is_valid(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)
        accuracy = metrics["accuracy"]
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_model_prediction(self):
        model = joblib.load("hotel_cancellation_model.pkl")
        prediction = model.predict(make_sample())[0]
        self.assertIn(int(prediction), [0, 1])

    def test_high_risk_booking(self):
        # Long lead time, non-refundable, earlier cancellations, no special requests
        model = joblib.load("hotel_cancellation_model.pkl")
        sample = make_sample(
            lead_time=300,
            previous_cancellations=3,
            deposit_type="Non Refund",
            total_of_special_requests=0,
            market_segment="Groups",
        )
        # TEMPORARY FAILURE: original expected value was 1
        self.assertEqual(int(model.predict(sample)[0]), 0)

    def test_low_risk_booking(self):
        # Short lead time, special requests, parking, booking changes made
        model = joblib.load("hotel_cancellation_model.pkl")
        sample = make_sample(
            lead_time=2,
            previous_cancellations=0,
            booking_changes=2,
            total_of_special_requests=3,
            required_car_parking_spaces=1,
            market_segment="Direct",
        )
        self.assertEqual(int(model.predict(sample)[0]), 0)


if __name__ == "__main__":
    unittest.main()
