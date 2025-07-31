# Fake News Detection E2E Project

## Overview

This project is an end-to-end pipeline for detecting fake news using machine learning. It includes data preprocessing, model training, evaluation, and a Streamlit web app for interactive predictions.

## Project Structure

```
.
├── data/
│   ├── fake.csv
│   └── true.csv
├── models/
│   ├── logistic_model.pkl
│   └── tfidf_vectorizer.pkl
├── notebooks/
│   └── data_info.ipynb
├── src/
│   ├── __init__.py
│   ├── data_utils.py
│   └── train.py
├── streamlit/
│   ├── __init__.py
│   └── app.py
├── tests/
├── requirements.txt
├── README.md
└── template.py
```

## Getting Started

### 1. Install Dependencies

```sh
pip install -r requirements.txt
```

### 2. Data

- Place your datasets (`fake.csv`, `true.csv`) in the `data/` folder.

### 3. Training

- Run the training script to preprocess data and train the model:

```sh
python src/train.py 

   or 

python -m src.train

```

### 4. Streamlit App

- Launch the web app for predictions:

```sh
streamlit run streamlit/app.py
```

## Notebooks

- Explore and analyze the data in [`notebooks/data_info.ipynb`](notebooks/data_info.ipynb).

## Code Modules

- [`src/data_utils.py`](src/data_utils.py): Data loading and preprocessing functions.
- [`src/train.py`](src/train.py): Model training and saving.
- [`streamlit/app.py`](streamlit/app.py): Streamlit web app for fake news detection.

## Model Artifacts

- Trained model and vectorizer are saved in the `models/` directory.

## Testing

- Add your unit tests in the `tests/` directory.

## Requirements

- See [`requirements.txt`](requirements.txt) for all dependencies.

## License

This project is for