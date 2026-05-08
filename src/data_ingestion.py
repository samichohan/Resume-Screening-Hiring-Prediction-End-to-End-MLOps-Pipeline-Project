# ============================================================
# FILE: src/data_ingestion.py
# KYA KARTA HAI: Kaggle se dataset download karta hai aur
#                raw CSV file ko data_and_model/ folder mein save karta hai
#
# SIMPLE MISAAL:
#   Jaise tu market se sabziyan khareedta hai aur ghar la ke
#   ek thele mein rakh deta hai — yahi kaam yeh file karti hai
#   lekin data ke liye.
#
# INPUT:  Kaggle dataset URL / kaggle.json credentials
# OUTPUT: data_and_model/raw_resume_data.csv
# ============================================================

import os
import shutil
import kaggle  # pip install kaggle

# ---- SETTINGS ----
DATASET = "rhythmghai/resume-screening-dataset-200k-candidates"
RAW_DATA_DIR = "data_and_model"
RAW_FILE_NAME = "raw_resume_data.csv"

def download_data():
    """
    Kaggle se data download karta hai.
    
    Pehle ek baar setup karna hoga:
      1. https://www.kaggle.com pe jao
      2. Account > Settings > API > Create New Token
      3. kaggle.json file download hogi
      4. Windows mein C:/Users/TumharaName/.kaggle/ folder mein rakho
    """
    print("=" * 50)
    print("Step 1: Data Ingestion Shuru...")
    print("=" * 50)

    # Folder banao agar nahi hai
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    # Kaggle se download karo
    print(f"Kaggle se dataset download ho raha hai: {DATASET}")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        DATASET,
        path=RAW_DATA_DIR,
        unzip=True
    )

    # Downloaded file ko sahi naam do
    # (Kaggle ke file names change ho sakte hain isliye rename karte hain)
    downloaded_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".csv")]
    if downloaded_files:
        src = os.path.join(RAW_DATA_DIR, downloaded_files[0])
        dst = os.path.join(RAW_DATA_DIR, RAW_FILE_NAME)
        shutil.move(src, dst)
        print(f"Data save ho gaya: {dst}")
    else:
        raise FileNotFoundError("CSV file nahi mili! Kaggle download check karo.")

    print(f"Total files in {RAW_DATA_DIR}/:", os.listdir(RAW_DATA_DIR))
    print("Step 1 Complete!\n")

if __name__ == "__main__":
    download_data()
