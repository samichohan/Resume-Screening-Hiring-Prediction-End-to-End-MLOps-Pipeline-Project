from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal

app = FastAPI(
    title="Resume Hiring Prediction API",
    description="Candidate ka resume data do, model batayega hired hoga ya nahi",
    version="1.0.0"
)

class CandidateInput(BaseModel):
    age: int = Field(default=25)
    education_level: Literal["Bachelors", "Masters", "PhD", "High School"] = Field(default="Bachelors")
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
    message: str

@app.get("/")
def home():
    return {
        "message": "Resume Hiring Prediction API chal rahi hai!",
        "docs": "/docs"
    }

@app.post("/predict", response_model=PredictionOutput)
def get_prediction(candidate: CandidateInput):
    try:
        return {"message": f"Data received! Age: {candidate.age}, CGPA: {candidate.cgpa}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}