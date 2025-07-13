# import sys
# import os

# # Add the root project directory to sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from src.predict import predict_news

# import streamlit as st
# from src.predict import predict_news

# st.title("📰 Fake News Detector")
# st.write("Enter a news article and find out if it's real or fake!")

# user_input = st.text_area("News Text", height=200)

# if st.button("Predict"):
#     prediction = predict_news(user_input)
#     st.success(f"The news is: **{prediction.upper()}**")



import streamlit as st
import joblib
import os

# Load the trained model
MODEL_PATH = "models/model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Trained model not found. Please run `train_model.py` first.")
        return None
    return joblib.load(MODEL_PATH)

# App title
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector")
st.markdown("Check whether a news article is **FAKE** or **REAL** using ML.")

# Load model
model = load_model()

# User input
user_input = st.text_area("✍️ Enter News Content Here", height=200)

# Predict button
if st.button("🔍 Predict"):
    if not user_input.strip():
        st.warning("Please enter some news text.")
    elif model:
        prediction = model.predict([user_input])[0]
        if prediction == "FAKE":
            st.error("🚨 This news is likely **FAKE**.")
        else:
            st.success("✅ This news is likely **REAL**.")
