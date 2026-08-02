"""Streamlit web app for the sentiment analysis model."""
import os
import sys
import streamlit as st
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from train import load_config, build_pipeline
from data_loader import load_data, split_data

st.set_page_config(page_title="Sentiment Analysis AI", page_icon="💬")
st.title("💬 Sentiment Analysis AI")
st.write("Enter a review below and the model will predict whether it's positive or negative.")

MODEL_PATH = "models/sentiment_model.joblib"


@st.cache_resource
def get_model():
    cfg = load_config()
    if not os.path.exists(MODEL_PATH):
        # Train on the fly if no saved model exists (e.g. first deploy)
        df = load_data(cfg["data"]["raw_path"])
        X_train, X_test, y_train, y_test = split_data(
            df, cfg["data"]["test_size"], cfg["data"]["random_state"]
        )
        pipeline = build_pipeline(cfg)
        pipeline.fit(X_train, y_train)
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)
        return pipeline
    return joblib.load(MODEL_PATH)


model = get_model()

text = st.text_area("Your text:", placeholder="I absolutely loved this product!")

if st.button("Predict"):
    if text.strip():
        pred = model.predict([text])[0]
        proba = model.predict_proba([text]).max()
        emoji = "😊" if pred == "positive" else "😞"
        st.success(f"{emoji} **{pred.capitalize()}** (confidence: {proba:.2f})")
    else:
        st.warning("Please enter some text first.")
