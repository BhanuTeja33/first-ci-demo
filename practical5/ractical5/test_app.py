import unittest
from app import app

HIGH_RISK = {
    "lead_time": 300,
    "adults": 2,
    "stays_in_weekend_nights": 1,
    "stays_in_week_nights": 2,
    "previous_cancellations": 3,
    "booking_changes": 0,
    "adr": 100.0,
    "total_of_special_requests": 0,
    "required_car_parking_spaces": 0,
    "hotel": "City Hotel",
    "deposit_type": "Non Refund",
    "market_segment": "Groups",
}

LOW_RISK = {
    "lead_time": 2,
    "adults": 2,
    "stays_in_weekend_nights": 1,
    "stays_in_week_nights": 2,
    "previous_cancellations": 0,
    "booking_changes": 2,
    "adr": 100.0,
    "total_of_special_requests": 3,
    "required_car_parking_spaces": 1,
    "hotel": "City Hotel",
    "deposit_type": "No Deposit",
    "market_segment": "Direct",
}


class TestPredictionApplication(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_high_risk_prediction(self):
        response = self.client.post("/predict", json=HIGH_RISK)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["prediction"], "CANCELED")

    def test_low_risk_prediction(self):
        response = self.client.post("/predict", json=LOW_RISK)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["prediction"], "NOT_CANCELED")

    def test_missing_field_validation(self):
        response = self.client.post(
            "/predict",
            json={"lead_time": 100, "adults": 2}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("missing_fields", response.get_json())


if __name__ == "__main__":
    unittest.main()
