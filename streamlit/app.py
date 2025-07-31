# import streamlit as st
# import joblib

# vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
# model = joblib.load('models/logistic_model.pkl')

# st.title("Fake News detector")
# st.write("Enter the news text below to check if it's real or fake")

# user_input = st.text_area("News Text", height=200)

# if st.button("Check News"):
#     if user_input.strip():
#         transform_user_input = vectorizer.transform([user_input])
#         prediction = model.predict(transform_user_input)

#         if prediction[0] == 1:
#             st.success("This news is REAL!")
#         else:   
#             st.error("This news is FAKE!")
#     else:
#         st.warning("Please enter text (news) to analyze")

# import streamlit as st
# import joblib

# st.set_page_config(page_title="📰 Fake News Detector", page_icon="🕵️‍♂️", layout="centered")

# # Sidebar
# st.sidebar.header("🕵️‍♂️ Fake News Detector")
# st.sidebar.markdown(
#     """
#     **Instructions:**
#     - Paste or type the news text in the box.
#     - Click **Check News** to see if it's real or fake.
#     - The model uses machine learning to analyze the news.
#     """
# )

# # Main Title and Description
# st.markdown(
#     """
#     <div style="background-color:#f9f9f9;padding:20px;border-radius:10px">
#         <h1 style="color:#1f77b4;text-align:center;">📰 Fake News Detector</h1>
#         <p style="text-align:center;color:#555;">
#             Enter news text below and let AI tell you if it's <b style="color:green;">REAL</b> or <b style="color:red;">FAKE</b>!
#         </p>
#     </div>
#     """,
#     unsafe_allow_html=True
# )

# vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
# model = joblib.load('models/logistic_model.pkl')

# user_input = st.text_area("✍️ Paste or type news text here:", height=200)

# if st.button("🔍 Check News"):
#     if user_input.strip():
#         transform_user_input = vectorizer.transform([user_input])
#         prediction = model.predict(transform_user_input)

#         if prediction[0] == 1:
#             st.success("✅ This news is REAL! 🟢")
#             st.balloons()
#         else:
#             st.error("🚨 This news is FAKE! 🔴")
#             st.snow()
#     else:
#         st.warning("⚠️ Please enter text (news) to analyze.")



import streamlit as st
import joblib


# Page Configuration
st.set_page_config(page_title="🕵️ Fake News Detector", page_icon="📰", layout="centered")

# Custom Dark Style

st.markdown(
    """
    <style>
        body {
            background-color: #121212;
            color: #E0E0E0;
        }
        .main {
            background-color: #1e1e1e;
            padding: 2rem;
            border-radius: 10px;
        }
        h1, h2, h3 {
            color: #00BFFF;
        }
        .stTextArea textarea {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        .stButton>button {
            background-color: #00BFFF;
            color: white;
            border-radius: 8px;
        }
        .stButton>button:hover {
            background-color: #1e90ff;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3064/3064197.png", width=120)
st.sidebar.title("🕵️‍♂️ Fake News Detector")
st.sidebar.markdown(
    """
    ### 📝 Instructions:
    - Paste or type news content.
    - Click **Check News** to analyze.
    - Model predicts using AI/ML logic.
    """
)

# Title

st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown(
    """
    <h1 style='text-align: center;'>📰 AI-Powered Fake News Classifier</h1>
    <p style='text-align: center; font-size: 18px;'>
        Enter a news article and see if it's <span style='color:limegreen;'>REAL</span> or <span style='color:red;'>FAKE</span>
    </p>
    """,
    unsafe_allow_html=True
)

# Load Model and Vectorizer
try:
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    model = joblib.load('models/logistic_model.pkl')
except Exception as e:
    st.error(f"❌ Error loading model or vectorizer: {e}")
    st.stop()


# User Input
user_input = st.text_area("🖋️ Paste or type the news content here:", height=200, placeholder="E.g. The US government passed a new bill today...")


# Prediction Button
if st.button("🔍 Check News"):
    if user_input.strip():
        transform_user_input = vectorizer.transform([user_input])
        prediction = model.predict(transform_user_input)

        if prediction[0] == 1:
            st.success("✅ This news is REAL! 🟢")
            st.markdown("<p style='color:lime;'>🧠 The article shows credible language patterns.</p>", unsafe_allow_html=True)
            st.balloons()
        else:
            st.error("🚨 This news is FAKE! 🔴")
            st.markdown("<p style='color:red;'>⚠️ Warning: The content resembles known disinformation patterns.</p>", unsafe_allow_html=True)
            st.snow()
    else:
        st.warning("⚠️ Please enter some news text.")

st.markdown("</div>", unsafe_allow_html=True)
