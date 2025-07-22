import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# Set Kaggle credentials directly (no need for .env)
os.environ['KAGGLE_USERNAME'] = "abhirajkumar05"
os.environ['KAGGLE_KEY'] = "697637c98bac834ce92cb3a55aa8da4b"

# Initialize and authenticate
api = KaggleApi()
api.authenticate()

# Download and unzip dataset
dataset_name = "manisha717/dataset-of-pdf-files"
download_dir = "./data"
zip_path = os.path.join(download_dir, "./dataset.zip")

print("Downloading dataset...")
api.dataset_download_files(dataset_name, path=download_dir, unzip=False)

# Unzip manually
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(download_dir)

# Locate PDFs
pdf_dir = os.path.join(download_dir, "dataset-of-pdf-files")
pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

# Get only the first 20
pdf_files = pdf_files[:20]

print(f"✅ Found {len(pdf_files)} PDFs:")
for pdf in pdf_files:
    print(" -", pdf)
