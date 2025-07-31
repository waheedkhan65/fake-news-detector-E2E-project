# Fake and Real News Classification Project

## Overview
This project is a machine learning application that classifies news articles as either "fake" or "real" using logistic regression with TF-IDF vectorization. The system processes text data, trains a classification model, and provides evaluation metrics to assess performance.

## Project Structure
```
PROJECT ML/
├── .venv/                   # Virtual environment
├── data/
│   ├── fake.csv             # Dataset of fake news articles
│   └── true.csv             # Dataset of true news articles
├── models/
│   ├── logistic_model.pkl   # Trained logistic regression model
│   └── tfidf_vectorizer.pkl # TF-IDF vectorizer
├── notebooks/
│   └── data_info.ipynb      # Jupyter notebook for data exploration
├── src/
│   ├── data_utils.py        # Data loading and preprocessing utilities
│   └── train.py            # Model training script
├── tests/                   # Test files
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
└── template.py             # Template file
```

## Features
- Text preprocessing including:
  - Lowercasing
  - Removing brackets, URLs, HTML tags
  - Punctuation removal
  - Digit removal
- TF-IDF vectorization
- Logistic Regression classifier
- Model evaluation with:
  - Accuracy score
  - Classification report
  - Confusion matrix visualization

## Installation
1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd project-ML-new
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Preparation
Place your datasets in the `data/` folder:
- `fake.csv` - Fake news dataset
- `true.csv` - Real news dataset

### Training the Model
Run the training script:
```bash
python src/train.py
```

This will:
1. Load and preprocess the data
2. Train a logistic regression model
3. Save the model and vectorizer to the `models/` folder
4. Print evaluation metrics and show a confusion matrix

### Expected Output
After running the training script, you should see:
- Accuracy score
- Classification report (precision, recall, f1-score)
- Confusion matrix visualization

## Dependencies
- Python 3.x
- pandas
- scikit-learn
- matplotlib
- seaborn
- joblib

All dependencies are listed in `requirements.txt`.

## Customization
To modify the project:
1. Edit `data_utils.py` to change text preprocessing steps
2. Modify `train.py` to:
   - Use a different classifier
   - Change train/test split ratio
   - Add additional evaluation metrics

## License
This project is open-source. Feel free to use and modify it as needed.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.