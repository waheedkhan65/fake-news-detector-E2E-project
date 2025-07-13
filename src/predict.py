import joblib

def predict_news(text):
    model = joblib.load('models/model.pkl')
    return model.predict([text])[0]
