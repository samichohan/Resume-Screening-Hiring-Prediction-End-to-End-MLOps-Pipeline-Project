from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import glob
import pickle
import pandas as pd
import os

app = FastAPI(
    title= "Resume Hiring Prediction API",
    description= "Candidate ka resume data do,model batayega hired hoga ya nhi",
    version= '1.0.0' 
)

EDUCATION_MAP = {'Bachelors': 0,'Masters':1,'Phd': 2, "High School": 3}
UNIVERSITY_MAP = {"Tier 1": 0, "Tier 2":1,"Tier 3": 2}
COMPANY_MAP = {"Government":0,"MNC": 1, "Startup": 2,"Mid-size": 3}

def load_model():
    model_path = "models/best_model.pkl"
    if not os.path.exists(model_path):
        raise Exception("Model nahi mila!")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model
class CandidateInput(BaseModel):
    age: int = Field(default=25)
    education_level: Literal["Bachelors", "Masters", "Phd", "High School"] = Field(default="Bachelors")
    university_tier: Literal["Tier 1", "Tier 2", "Tier 3"] = Field(default="Tier 2")
    cgpa: float = Field(default=7.0)
    internships: int = Field(default=0)
    projects: int = Field(default=1)
    programming_languages: int = Field(default=1)
    certifications: int = Field(default=0)
    experience_years: float = Field(default=0.0)
    hackathons: int = Field(default=0)
    research_papers: int = Field(default=0)
    skills_score: float = Field(default=10.0)
    soft_skills_score: float = Field(default=5.0)
    resume_length_words: int = Field(default=300)
    company_type: Literal["MNC", "Startup", "Government", "Mid-size"] = Field(default="MNC")

class PredictionOutput(BaseModel):
    hired: int
    probability: float
    verdict: str

@app.get("/")
def home():
    return {
        "message": "Resume Hiring Prediction API chal rahi hai!",
        "docs": "/docs"
    }

@app.post("/predict", response_model=PredictionOutput)
def get_prediction(candidate: CandidateInput):
    try:
        model = load_model()
        input_data = pd.DataFrame([{
            "age":                   candidate.age,
            "education_level":       EDUCATION_MAP.get(candidate.education_level, 0),
            "university_tier":       UNIVERSITY_MAP.get(candidate.university_tier, 1),
            "cgpa":                  candidate.cgpa,
            "internships":           candidate.internships,
            "projects":              candidate.projects,
            "programming_languages": candidate.programming_languages,
            "certifications":        candidate.certifications,
            "experience_years":      candidate.experience_years,
            "hackathons":            candidate.hackathons,
            "research_papers":       candidate.research_papers,
            "skills_score":          candidate.skills_score,
            "soft_skills_score":     candidate.soft_skills_score,
            "resume_length_words":   candidate.resume_length_words,
            "company_type":          COMPANY_MAP.get(candidate.company_type, 1),
        }])
        prediction  = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        return {
            "hired":       int(prediction),
            "probability": round(float(probability), 4),
            "verdict":     "Hired!" if prediction == 1 else "Not Hired"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}
    
