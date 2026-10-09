import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_FILE = "data/Hotel_Bookings.csv"

NUMERIC_FEATURES = [
    "lead_time",
    "adults",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "previous_cancellations",
    "booking_changes",
    "adr",
    "total_of_special_requests",
    "required_car_parking_spaces",
]
CATEGORICAL_FEATURES = ["hotel", "deposit_type", "market_segment"]
TARGET = "is_canceled"  # 1 = CANCELED, 0 = NOT CANCELED


def load_dataset():
    data = pd.read_csv(DATA_FILE)
    return data[NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]]


def train_model():
    print("Loading dataset...")
    data = load_dataset()
    print("Dataset loaded successfully.")
    print("Number of records:", len(data))

    X = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("Training records:", len(X_train))
    print("Testing records :", len(X_test))

    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    print("Training model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)

    print("\nModel Evaluation")
    print("----------------")
    print("Accuracy:", round(accuracy, 4))
    print("\nConfusion Matrix:")
    print(matrix)

    joblib.dump(model, "hotel_cancellation_model.pkl")
    print("\nModel saved as hotel_cancellation_model.pkl")

    metrics = {
        "accuracy": float(accuracy),
        "training_records": len(X_train),
        "testing_records": len(X_test),
    }
    with open("metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)
    print("Metrics saved as metrics.json")
    return accuracy


if __name__ == "__main__":
    train_model()
