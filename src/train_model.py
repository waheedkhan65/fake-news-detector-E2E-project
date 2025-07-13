# 

import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import load_and_split_data

def train_and_save_model():
    # Load and split the data
    X_train, X_test, y_train, y_test = load_and_split_data()

    # Define the pipeline
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
        ('model', LogisticRegression())
    ])

    # Train the model
    pipe.fit(X_train, y_train)

    # Save the model
    joblib.dump(pipe, 'models/model.pkl')
    print("✅ Model trained and saved as models/model.pkl")

    # === 🔍 Model Evaluation ===
    y_pred = pipe.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {accuracy:.4f}")

    # Classification Report
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=["FAKE", "REAL"])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["FAKE", "REAL"],
                yticklabels=["FAKE", "REAL"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_and_save_model()
