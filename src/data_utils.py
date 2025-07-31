import re
import string
import pandas as pd

def clean_text(text):
    """Cleans the input text by performing several preprocessing steps"""
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text) # Remove text within brackets
    text = re.sub(r'\\W', ' ', text)     # Remove non-word characters
    text = re.sub(r"https?://\S+|www\.\S+", "", text)     # Remove URLs
    text = re.sub(r'<.*?>+', '', text)    # Remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)    # Remove punctuation
    text = re.sub(r'\n', ' ', text)    # Remove newlines
    text = re.sub(r'\w*\d\w*', '', text)    # Remove words containing digits
    return text

def load_data(fake_path, true_path):
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df['label'] = 0
    true_df['label'] = 1

    data = pd.concat([fake_df, true_df], axis=0)
    data = data.sample(frac=1).reset_index(drop=True)
    data = data.drop(["title", "subject", "date"], axis=1, errors='ignore')
    data["text"] = data["text"].apply(clean_text)

    return data