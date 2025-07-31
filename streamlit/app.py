import streamlit as st
import joblib

vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
model = joblib.load('models/logistic_model.pkl')

st.title("Fake News detector")
st.write("Enter the news text below to check if it's real or fake")

user_input = st.text_area("News Text", height=200)

if st.button("Check News"):
    if user_input.strip():
        transform_user_input = vectorizer.transform([user_input])
        prediction = model.predict(transform_user_input)

        if prediction[0] == 1:
            st.success("This news is REAL!")
        else:   
            st.error("This news is FAKE!")
    else:
        st.warning("Please enter text (news) to analyze")