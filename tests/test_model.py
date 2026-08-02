"""Basic unit tests for the sentiment analysis pipeline."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import load_data, split_data  # noqa: E402


def test_load_data():
    df = load_data("data/raw/reviews.csv")
    assert not df.empty
    assert "text" in df.columns
    assert "label" in df.columns


def test_split_data():
    df = load_data("data/raw/reviews.csv")
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(X_train) + len(X_test) == len(df)


def test_labels_are_binary():
    df = load_data("data/raw/reviews.csv")
    assert set(df["label"].unique()) <= {"positive", "negative"}
