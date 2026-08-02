"""Run predictions on new text using the trained model."""
import argparse
import os
import sys
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import load_config


def predict(text: str, model_path: str) -> tuple[str, float]:
    model = joblib.load(model_path)
    pred = model.predict([text])[0]
    proba = model.predict_proba([text]).max()
    return pred, proba


def main():
    parser = argparse.ArgumentParser(description="Predict sentiment of a text string.")
    parser.add_argument("--text", type=str, required=True, help="Text to classify")
    args = parser.parse_args()

    cfg = load_config()
    pred, proba = predict(args.text, cfg["model"]["save_path"])

    print(f"Text: \"{args.text}\"")
    print(f"Prediction: {pred.capitalize()} (confidence: {proba:.2f})")


if __name__ == "__main__":
    main()
