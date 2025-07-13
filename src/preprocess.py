import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(fake_path="data/fake.csv", true_path="data/true.csv"):
    # Load both datasets
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    # Add labels
    fake_df["label"] = "FAKE"
    true_df["label"] = "REAL"

    # Combine datasets
    df = pd.concat([fake_df, true_df], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

    # Use 'text' column if it exists, else fall back to 'title'
    if 'text' in df.columns:
        df = df[['text', 'label']]
    elif 'title' in df.columns:
        df = df[['title', 'label']].rename(columns={"title": "text"})
    else:
        raise ValueError("Neither 'text' nor 'title' column found in data.")

    return train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
