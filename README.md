# Resume Hiring Prediction — MLOps Project

A complete end-to-end MLOps pipeline for predicting whether a candidate will be hired based on their resume features, using modern MLOps practices.

## Problem Statement

**Predict if a candidate will be hired (1) or not (0)** based on their resume attributes like education, experience, skills, CGPA, certifications, and more.

**Dataset:** [Resume Screening Dataset — 200K Candidates](https://www.kaggle.com/datasets/rhythmghai/resume-screening-dataset-200k-candidates)  
**Records:** 200,000 candidates  
**Target:** `hired` (0 = Not Hired, 1 = Hired)

---

## Project Structure

```
Project-Resume-MLOps-System/
├── src/
│   ├── data_ingestion.py    # Kaggle se data download
│   ├── preprocessing.py     # Data clean + encode + split
│   ├── train.py             # 4 models train + MLflow tracking
│   ├── best_model.py        # Best model select + DagsHub register
│   └── prediction.py        # Single candidate prediction
├── .github/
│   └── workflows/
│       └── train.yml        # CI/CD GitHub Actions
├── data_and_model/          # Data files (DVC tracked)
├── app.py                   # FastAPI prediction API
├── streamlit_app.py         # Streamlit web UI
├── dvc.yaml                 # DVC pipeline stages
├── dvc.lock                 # DVC pipeline lock
├── params.yaml              # Hyperparameters & config
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variables template
└── README.md
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Scikit-learn | ML models |
| MLflow | Experiment tracking & model registry |
| DVC | Data & pipeline versioning |
| DagsHub | Remote storage + MLflow hosting |
| FastAPI | REST API for predictions |
| Streamlit | Web UI for predictions |
| GitHub Actions | CI/CD automation |

---

## Features Used

| Feature | Type |
|---------|------|
| age | Numerical |
| education_level | Categorical |
| university_tier | Categorical |
| cgpa | Numerical |
| internships | Numerical |
| projects | Numerical |
| programming_languages | Numerical |
| certifications | Numerical |
| experience_years | Numerical |
| hackathons | Numerical |
| research_papers | Numerical |
| skills_score | Numerical |
| soft_skills_score | Numerical |
| resume_length_words | Numerical |
| company_type | Categorical |

---

## MLOps Pipeline

```
Kaggle Dataset
      ↓
data_ingestion.py   →   Raw CSV save
      ↓
preprocessing.py    →   Clean + Encode + Train/Test Split
      ↓
train.py            →   4 Models train + MLflow track
      ↓
best_model.py       →   Best model → Production (DagsHub)
      ↓
FastAPI + Streamlit →   Prediction UI
```

---

## Models Trained

| Model | Description |
|-------|-------------|
| Logistic Regression | Simple linear model |
| Decision Tree | Tree-based decisions |
| Random Forest | Ensemble of trees |
| Gradient Boosting | Boosted ensemble |

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Project-Resume-MLOps-System.git
cd Project-Resume-MLOps-System
```

### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables
```bash
# .env.example ko copy karo
copy .env.example .env       # Windows
cp .env.example .env         # Mac/Linux

# .env file mein apni DagsHub info fill karo
```

### 5. Setup Kaggle API
- https://www.kaggle.com > Account > Settings > API > Create New Token
- `kaggle.json` download karo
- Windows: `C:\Users\YourName\.kaggle\kaggle.json` mein rakho

### 6. Initialize DVC
```bash
dvc init
dvc remote add -d origin https://dagshub.com/YOUR_USERNAME/Project-Resume-MLOps-System.dvc
```

### 7. Run DVC Pipeline (Sab kuch automatic!)
```bash
dvc repro
```

### 8. Run FastAPI
```bash
uvicorn app:app --reload
# Browser: http://localhost:8000/docs
```

### 9. Run Streamlit UI
```bash
streamlit run streamlit_app.py
# Browser: http://localhost:8501
```

---

## CI/CD Pipeline

Every push to `main` branch automatically:

- Checks out code
- Installs dependencies
- Runs DVC pipeline (`dvc repro`)
- Trains all models
- Registers best model on DagsHub
- Updates `dvc.lock`

---

## API Usage

### Predict Endpoint
```
POST http://localhost:8000/predict
```

**Request:**
```json
{
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
```

**Response:**
```json
{
  "hired": 1,
  "probability": 0.87,
  "verdict": "Hired!"
}
```

---

## params.yaml

```yaml
data:
  test_size: 0.2
  random_state: 42

model:
  cv: 5
  scoring: "f1"
```

Change any value and run `dvc repro` — pipeline automatically retrains!

---

## Links

- **DagsHub Repo:** https://dagshub.com/YOUR_USERNAME/Project-Resume-MLOps-System
- **GitHub Repo:** https://github.com/YOUR_USERNAME/Project-Resume-MLOps-System

---

## Author

**[Your Name]**  
GitHub: [@your_username](https://github.com/your_username)  
DagsHub: [@your_username](https://dagshub.com/your_username)
