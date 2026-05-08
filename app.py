# ============================================================
# FILE: app.py  (root folder mein)
# KYA KARTA HAI: FastAPI server — ek link provide karta hai
#                jahan koi bhi apna data bhej ke prediction le sake
#
# SIMPLE MISAAL:
#   Yeh ek counter hai. Tu counter pe jao, apna order do,
#   counter wala result batata hai. Yahan tu link pe request
#   bhejta hai, server prediction return karta hai.
#
# CHALANE KA TARIKA (terminal mein):
#   uvicorn app:app --reload
#
# PHIR BROWSER MEIN JAO:
#   http://localhost:8000/docs   ← Swagger UI (test kar sakte ho)
#   http://localhost:8000/predict ← Prediction endpoint
# ============================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import sys
import os

sys.path.append(os.path.dirname(__file__))
from src.prediction import predict

# ---- APP BANAO ----
app = FastAPI(
    title="Resume Hiring Prediction API",
    description="Candidate ka resume data do, model batayega hired hoga ya nahi",
    version="1.0.0"
)

# ---- INPUT FORMAT DEFINE KARO ----
# Pydantic model — yeh batata hai API ko konsa data aana chahiye
class CandidateInput(BaseModel):
    age: int = Field(default=25, ge=18, le=60,
                     description="Candidate ki umar (18-60)")
    education_level: Literal["Bachelors", "Masters", "PhD", "High School"] = Field(
        default="Bachelors", description="Education level")
    university_tier: Literal["Tier 1", "Tier 2", "Tier 3"] = Field(
        default="Tier 2", description="University ki tier")
    cgpa: float = Field(default=7.0, ge=0.0, le=10.0,
                        description="CGPA (0-10)")
    internships: int = Field(default=0, ge=0,
                             description="Total internships ki ginti")
    projects: int = Field(default=1, ge=0,
                          description="Projects ki ginti")
    programming_languages: int = Field(default=1, ge=0,
                                       description="Programming languages ki ginti")
    certifications: int = Field(default=0, ge=0,
                                description="Certifications ki ginti")
    experience_years: float = Field(default=0.0, ge=0.0,
                                    description="Experience saalon mein")
    hackathons: int = Field(default=0, ge=0,
                            description="Hackathons mein participate kiya")
    research_papers: int = Field(default=0, ge=0,
                                 description="Research papers publish hue")
    skills_score: float = Field(default=10.0, ge=0.0,
                                description="Technical skills score")
    soft_skills_score: float = Field(default=5.0, ge=0.0,
                                     description="Soft skills score")
    resume_length_words: int = Field(default=300, ge=50,
                                     description="Resume ki word count")
    company_type: Literal["MNC", "Startup", "Government"] = Field(
        default="MNC", description="Company ka type")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 24,
                "education_level": "Masters",
                "university_tier": "Tier 1",
                "cgpa": 8.5,
                "internships": 2,
                "projects": 5,
                "programming_languages": 3,
                "certifications": 2,
                "experience_years": 1.5,
                "hackathons": 1,
                "research_papers": 0,
                "skills_score": 25,
                "soft_skills_score": 7.5,
                "resume_length_words": 450,
                "company_type": "MNC"
            }
        }

# ---- OUTPUT FORMAT ----
class PredictionOutput(BaseModel):
    hired: int
    probability: float
    verdict: str

# ---- ENDPOINTS ----

@app.get("/")
def home():
    """API ka home page — check karo API chal rahi hai ya nahi"""
    return {
        "message": "Resume Hiring Prediction API chal rahi hai!",
        "docs": "/docs",
        "predict": "/predict"
    }

@app.post("/predict", response_model=PredictionOutput)
def get_prediction(candidate: CandidateInput):
    """
    Candidate data bhejo, prediction pao.
    
    REQUEST (POST):
    {
        "age": 24,
        "education_level": "Masters",
        "cgpa": 8.5,
        ...
    }
    
    RESPONSE:
    {
        "hired": 1,
        "probability": 0.87,
        "verdict": "Hired!"
    }
    """
    try:
        result = predict(candidate.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
def health_check():
    """API ka health check — CI/CD mein use hota hai"""
    return {"status": "healthy", "message": "API theek chal rahi hai!"}
