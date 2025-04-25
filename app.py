import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = tf.keras.models.load_model("sentiment_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
# Streamlit app
st.title("🎬 Movie Review Sentiment Analyzer")

user_input = st.text_area("Enter your movie review:", height=150)

if st.button("Predict Sentiment"):
    if user_input:
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequence, maxlen=200)
        prediction = model.predict(padded)
        sentiment = "😊 Positive" if prediction[0][0] > 0.5 else "☹️ Negative"
        st.success(f"The sentiment is: **{sentiment}**")
    else:
        st.warning("Please enter a review.")

