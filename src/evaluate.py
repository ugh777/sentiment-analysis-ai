"""Evaluate the trained sentiment model."""
import os
import sys
import joblib
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_data, split_data
from train import load_config


def main():
    cfg = load_config()

    df = load_data(cfg["data"]["raw_path"])
    _, X_test, _, y_test = split_data(
        df, cfg["data"]["test_size"], cfg["data"]["random_state"]
    )

    model = joblib.load(cfg["model"]["save_path"])
    y_pred = model.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
