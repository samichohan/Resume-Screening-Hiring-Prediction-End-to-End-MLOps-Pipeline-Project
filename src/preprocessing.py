# ============================================================
# FILE: src/preprocessing.py
# KYA KARTA HAI: Raw (ganda) data ko saaf karta hai,
#                numbers mein convert karta hai,
#                aur train/test split karta hai
#
# SIMPLE MISAAL:
#   Jaise tu sabziyan laata hai, phir unhe dhoata hai,
#   kaatta hai, aur pakane ke liye teyar karta hai —
#   yahi kaam yeh file karti hai lekin data ke liye.
#
# INPUT:  data_and_model/raw_resume_data.csv
# OUTPUT: data_and_model/X_train.csv
#         data_and_model/X_test.csv
#         data_and_model/y_train.csv
#         data_and_model/y_test.csv
# ============================================================

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import yaml

# ---- SETTINGS (params.yaml se aati hain) ----
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

TEST_SIZE    = params["data"]["test_size"]      # kitna % test ke liye
RANDOM_STATE = params["data"]["random_state"]   # reproducibility ke liye

RAW_FILE   = "data_and_model/raw_resume_data.csv"
OUTPUT_DIR = "data_and_model"

def preprocess():
    print("=" * 50)
    print("Step 2: Preprocessing Shuru...")
    print("=" * 50)

    # ---- 1. DATA LOAD KARO ----
    print("Raw data load ho raha hai...")
    df = pd.read_csv(RAW_FILE)
    print(f"Dataset shape: {df.shape}")   # (200000, 17) → 200000 rows, 17 columns
    print(f"Columns: {list(df.columns)}")

    # ---- 2. UNNECESSARY COLUMN HATAO ----
    # candidate_id sirf number hai, koi kaam ka nahi model ke liye
    df.drop(columns=["candidate_id"], inplace=True, errors="ignore")
    print("'candidate_id' column hata diya (koi kaam ka nahi)")

    # ---- 3. MISSING VALUES CHECK KARO ----
    missing = df.isnull().sum()
    print(f"\nMissing values:\n{missing[missing > 0]}")

    # Agar koi missing value ho toh median se fill karo (numbers ke liye)
    for col in df.select_dtypes(include="number").columns:
        df[col].fillna(df[col].median(), inplace=True)

    # ---- 4. CATEGORICAL COLUMNS → NUMBERS (Label Encoding) ----
    # ML model sirf numbers samjhta hai, isliye text ko number mein badalna hoga
    #
    # MISAAL:
    #   education_level: "Bachelors"→0, "Masters"→1, "PhD"→2
    #   university_tier: "Tier 1"→0, "Tier 2"→1, "Tier 3"→2
    #   company_type:    "MNC"→0, "Startup"→1, "Government"→2

    categorical_cols = ["education_level", "university_tier", "company_type"]
    le = LabelEncoder()

    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
            print(f"'{col}' → numbers mein convert ho gaya")

    # ---- 5. FEATURES (X) aur TARGET (y) alag karo ----
    # X = input (jo model ko dete hain)
    # y = output (jo model predict karta hai) → hired (0 ya 1)
    X = df.drop(columns=["hired"])
    y = df["hired"]

    print(f"\nFeatures (X) shape: {X.shape}")
    print(f"Target (y) shape:   {y.shape}")
    print(f"Hired distribution:\n{y.value_counts()}")

    # ---- 6. TRAIN / TEST SPLIT ----
    # 80% training ke liye, 20% testing ke liye
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y  # hired/not-hired ratio same rakho dono mein
    )

    print(f"\nTrain set: {X_train.shape[0]} rows")
    print(f"Test set:  {X_test.shape[0]} rows")

    # ---- 7. SAVE KARO ----
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    X_train.to_csv(f"{OUTPUT_DIR}/X_train.csv", index=False)
    X_test.to_csv(f"{OUTPUT_DIR}/X_test.csv",  index=False)
    y_train.to_csv(f"{OUTPUT_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{OUTPUT_DIR}/y_test.csv",  index=False)

    print(f"\nFiles save ho gayi {OUTPUT_DIR}/ mein:")
    print("  X_train.csv, X_test.csv, y_train.csv, y_test.csv")
    print("Step 2 Complete!\n")

if __name__ == "__main__":
    preprocess()
