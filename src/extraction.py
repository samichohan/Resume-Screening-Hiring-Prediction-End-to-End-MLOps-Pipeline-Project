import os
import kagglehub
import pandas as pd

# 1. Project folder set
base_dir = os.getcwd()
data_dir = os.path.join(base_dir, "data", "raw")
os.makedirs(data_dir, exist_ok=True)
print("Using project folder:", base_dir)

# 2. Dataset download
print("\nDownloading Resume dataset...")
path = kagglehub.dataset_download(
    "rhythmghai/resume-screening-dataset-200k-candidates"
)
print("Downloaded at:", path)

# 3. CSV file dhundo aur load karo
csv_file = os.path.join(path, "resume_dataset_200k_enhanced.csv")
df = pd.read_csv(csv_file)

print("Downloaded! Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nFirst 5 records:")
print(df.head())

# 4. Save data to project folder
output_path = os.path.join(data_dir, "data.csv")
df.to_csv(output_path, index=False)
print("\nData saved at:", output_path)

# 5. Final check
print("\nFinal files in data/raw:")
print(os.listdir(data_dir))