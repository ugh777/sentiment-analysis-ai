"""Train a TF-IDF + Logistic Regression sentiment classifier."""
import os
import sys
import yaml
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_data, split_data


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_pipeline(cfg: dict) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=cfg["model"]["max_features"],
            ngram_range=tuple(cfg["model"]["ngram_range"]),
        )),
        ("clf", LogisticRegression(C=cfg["model"]["C"], max_iter=1000)),
    ])


def main():
    cfg = load_config()

    df = load_data(cfg["data"]["raw_path"])
    X_train, X_test, y_train, y_test = split_data(
        df, cfg["data"]["test_size"], cfg["data"]["random_state"]
    )

    pipeline = build_pipeline(cfg)
    pipeline.fit(X_train, y_train)

    train_acc = pipeline.score(X_train, y_train)
    test_acc = pipeline.score(X_test, y_test)
    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test accuracy:  {test_acc:.3f}")

    os.makedirs(os.path.dirname(cfg["model"]["save_path"]), exist_ok=True)
    joblib.dump(pipeline, cfg["model"]["save_path"])
    print(f"Model saved to {cfg['model']['save_path']}")


if __name__ == "__main__":
    main()
