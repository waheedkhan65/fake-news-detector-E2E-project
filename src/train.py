from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
import os
from src.data_utils import load_data


def save_object(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)
    print(f"✅ Saved: {path}")


def train_model(fake_path, true_path, model_path, vectorizer_path):
    # Load dataset
    data = load_data(fake_path, true_path)
    X = data['text']
    y = data['label']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Initialize and train XGBoost model
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', max_depth=5, learning_rate=0.1, n_estimators=100)
    model.fit(X_train_vectorized, y_train)

    # Save the model and vectorizer
    save_object(model, model_path)
    save_object(vectorizer, vectorizer_path)

    # Predict and evaluate
    y_pred = model.predict(X_test_vectorized)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=['True Fake', 'True Real'], columns=['Pred Fake', 'Pred Real'])
    print("Confusion Matrix:")
    print(cm_df)

    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.show()

    return model, vectorizer


if __name__ == "__main__":
    fake_path = 'data/fake.csv'
    true_path = 'data/true.csv'
    model_path = 'models/xgboost_model.pkl'
    vectorizer_path = 'models/tfidf_vectorizer.pkl'

    train_model(fake_path, true_path, model_path, vectorizer_path)



















# from xgboost import XGBClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.feature_extraction.text import TfidfVectorizer
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import joblib
# import os
# from src.data_utils import load_data


# def save_object(obj, path):
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     joblib.dump(obj, path)
#     print(f"✅ Saved: {path}")

# def train_model(fake_path, true_path, model_path, vectorizer_path):
#     data = load_data(fake_path, true_path)
#     X = data['text']
#     y = data['label']

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#     vectorizer = TfidfVectorizer()
#     X_train_vectorized = vectorizer.fit_transform(X_train)
#     X_test_vectorized = vectorizer.transform(X_test)

#     model = LogisticRegression(max_iter=1000)
#     model.fit(X_train_vectorized, y_train)

#     save_object(model, model_path)
#     save_object(vectorizer, vectorizer_path)

#     y_pred = model.predict(X_test_vectorized)
#     accuracy = accuracy_score(y_test, y_pred)
#     report = classification_report(y_test, y_pred)
#     print(f"Accuracy: {accuracy}")
#     print("Classification Report:")
#     print(report)

#     # Confusion matrix
#     cm = confusion_matrix(y_test, y_pred)
#     cm_df = pd.DataFrame(cm, index=['True Fake', 'True Real'], columns=['Pred Fake', 'Pred Real'])
#     print("Confusion Matrix:")
#     print(cm_df)

    
#     # Plot confusion matrix
#     plt.figure(figsize=(6, 4))
#     sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
#     plt.title("Confusion Matrix")
#     plt.ylabel("Actual")
#     plt.xlabel("Predicted")
#     plt.show()

    
#     return model, vectorizer




# if __name__ == "__main__":
#     fake_path = 'data/fake.csv'
#     true_path = 'data/true.csv'
#     model_path = 'models/logistic_model.pkl'
#     vectorizer_path = 'models/tfidf_vectorizer.pkl'

#     train_model(fake_path, true_path, model_path, vectorizer_path)
