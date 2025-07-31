import os

# Define the folder and file structure
structure = {
    ".venv": [],
    "data": [],
    "models": [],
    "notebooks": ["data_info.ipynb"],
    "src": [
        "__init__.py",
        "data_utils.py",
        "train.py"
    ],
    "streamlit_app": [
        "__init__.py",
        "app.py"
    ],
    "tests": [],
    ".": ["requirements.txt", "README.md", "template.py"]  # ✅ fixed
}

# Create folders and files
for folder, files in structure.items():
    os.makedirs(folder, exist_ok=True)
    for file in files:
        open(os.path.join(folder, file), 'a').close()
