# app.py
import streamlit as st
import joblib

st.set_page_config(page_title="Fake News Detection", layout="centered")
st.title("📰 Fake News Detection with ML + NLP")

# Load model and vectorizer
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# User input
user_input = st.text_area("Enter the News Article Text Below")

if st.button("Predict"):
    if user_input.strip() != "":
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        st.write("### Prediction:", "🟥 **Fake News**" if prediction == 1 else "🟩 **Real News**")
    else:
        st.warning("Please enter some text.")
