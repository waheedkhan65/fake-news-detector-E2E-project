import os

# Define folder and file structure
structure = {
    "data": ["fake_or_real_news.csv"],
    "notebooks": ["eda.ipynb"],
    "models": ["model.pkl"],
    "src": ["preprocess.py", "train_model.py", "predict.py"],
    "app": ["streamlit_app.py"],
    ".": ["requirements.txt", "README.md", ".gitignore"],
}

def create_structure(base_path="."):
    for folder, files in structure.items():
        folder_path = os.path.join(base_path, folder)
        if folder != ".":
            os.makedirs(folder_path, exist_ok=True)
        for file in files:
            file_path = os.path.join(base_path if folder == "." else folder_path, file)
            if not os.path.exists(file_path):
                with open(file_path, "w") as f:
                    f.write("")  # create an empty file
    print("✅ Project folder structure created successfully!")

if __name__ == "__main__":
    create_structure()
