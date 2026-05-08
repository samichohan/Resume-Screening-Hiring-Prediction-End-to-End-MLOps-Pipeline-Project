# ============================================================
# FILE: src/prediction.py
# KYA KARTA HAI: Best model load karke ek naye candidate
#                ke liye prediction karta hai
#
# SIMPLE MISAAL:
#   Trained model ko ek naya resume do,
#   woh batata hai: "Yeh banda hire hoga ya nahi?"
#
# INPUT:  Single candidate ka data (dictionary)
# OUTPUT: hired = 1 (hoga) ya 0 (nahi hoga)
#         probability = kitne % chance hai
# ============================================================

import os
import glob
import pickle
import mlflow
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_USER     = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")

if MLFLOW_URI:
    mlflow.set_tracking_uri(MLFLOW_URI)
if MLFLOW_USER:
    os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_USER
if MLFLOW_PASSWORD:
    os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_PASSWORD

EDUCATION_MAP  = {"Bachelors": 0, "Masters": 1, "PhD": 2, "High School": 3}
UNIVERSITY_MAP = {"Tier 1": 0, "Tier 2": 1, "Tier 3": 2}
COMPANY_MAP    = {"Government": 0, "MNC": 1, "Startup": 2}

def load_best_model():
    import pickle
    # Sabse latest model load karo
    model_path = r"mlruns\165309380226320171\models\m-e6f7c6c73f6940fca3e383f397fa8e8a\artifacts\model.pkl"
    print(f"Model load ho raha hai: {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("Model ready!")
    return model

def encode_input(data: dict) -> pd.DataFrame:
    encoded = {
        "age":                   data.get("age", 25),
        "education_level":       EDUCATION_MAP.get(data.get("education_level", "Bachelors"), 0),
        "university_tier":       UNIVERSITY_MAP.get(data.get("university_tier", "Tier 2"), 1),
        "cgpa":                  data.get("cgpa", 7.0),
        "internships":           data.get("internships", 0),
        "projects":              data.get("projects", 1),
        "programming_languages": data.get("programming_languages", 1),
        "certifications":        data.get("certifications", 0),
        "experience_years":      data.get("experience_years", 0),
        "hackathons":            data.get("hackathons", 0),
        "research_papers":       data.get("research_papers", 0),
        "skills_score":          data.get("skills_score", 10.0),
        "soft_skills_score":     data.get("soft_skills_score", 5.0),
        "resume_length_words":   data.get("resume_length_words", 300),
        "company_type":          COMPANY_MAP.get(data.get("company_type", "MNC"), 1),
    }
    return pd.DataFrame([encoded])

def predict(data: dict):
    model       = load_best_model()
    input_df    = encode_input(data)
    prediction  = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    return {
        "hired":       int(prediction),
        "probability": round(float(probability), 4),
        "verdict":     "Hired!" if prediction == 1 else "Not Hired"
    }