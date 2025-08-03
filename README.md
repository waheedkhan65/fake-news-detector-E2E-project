# 🧠 Fake and Real News Classification Project

## 📰 Overview

This project classifies news articles as either **"fake"** or **"real"** using a machine learning pipeline built with:
- **TF-IDF Vectorization**
- **XGBoost Classifier**
- An interactive **Streamlit web app** frontend


## 📁 Project Structure

```

project-root/
│
├── data/                  # Raw data files
│   ├── fake.csv
│   ├── true.csv
│   └── **init**.py
│
├── models/                # Saved model and vectorizer
│   ├── xgboost\_model.pkl
│   └── tfidf\_vectorizer.pkl
│
├── notebooks/             # Jupyter notebooks for EDA
│   └── data\_info.ipynb
│
├── src/                   # Core training and data utility code
│   ├── **init**.py
│   ├── data\_utils.py
│   └── train.py
│
├── streamlit/             # Streamlit web application
│   ├── **init**.py
│   └── app.py
│
├── tests/                 # Placeholder for unit tests
│
├── .gitignore
├── requirements.txt
├── template.py
└── README.md

````

---

## ✨ Features

- **Text Cleaning & Preprocessing**
- **TF-IDF Feature Extraction**
- **XGBoost Model for Binary Classification**
- **Performance Evaluation**
- **Interactive News Prediction via Streamlit**

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector
````

### 2. Create a Virtual Environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Model Training

### Step 1: Place Data

Ensure your `data/` folder contains:

* `fake.csv`
* `true.csv`

### Step 2: Train the Model

```bash
python src/train.py
```

This will:

* Preprocess data
* Train an XGBoost classifier
* Save the model to `models/`
* Display accuracy, confusion matrix, and a classification report

---

## 🖥 Streamlit Web Interface

### Launch the Web App:

```bash
streamlit run streamlit/app.py
```

### App Features:

* Input any news article text
* See prediction result instantly (`Real News` ✅ or `Fake News` ❌)
* Friendly and responsive UI
* Uses trained model and TF-IDF vectorizer from `models/`

---

## 📊 Example Output

* **Accuracy:** \~93% (XGBoost)
* **Visual Confusion Matrix**
* **Classification Report (Precision, Recall, F1-score)**

---

## 🔧 Customization Options

* Modify text preprocessing in `src/data_utils.py`
* Replace or extend the model in `src/train.py`
* Add advanced UI features in `streamlit/app.py`

---

## 📦 Tech Stack

* Python
* pandas, scikit-learn
* XGBoost
* Streamlit
* seaborn, matplotlib

---

## 📜 License

MIT License — Feel free to use, modify, and distribute.

---

## 🤝 Contributing

Have ideas or improvements? PRs are welcome!

---
