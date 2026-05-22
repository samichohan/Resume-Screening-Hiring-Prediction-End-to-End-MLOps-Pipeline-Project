# \---

# title: Resume Hiring Prediction

# emoji: 📄

# colorFrom: blue

# colorTo: green

# sdk: docker

# app\_file: app.py

# pinned: false

# \---

# Resume Hiring Prediction — MLOps Project

A complete end-to-end MLOps pipeline for predicting whether a candidate will be hired, using PySpark for ETL and MLflow for experiment tracking.

## Problem Statement

**Predict if a candidate will be hired (1) or not (0)** based on resume attributes like education, experience, skills, CGPA, certifications, and more.

**Dataset:** [Resume Screening Dataset — 200K Candidates](https://www.kaggle.com/datasets/rhythmghai/resume-screening-dataset-200k-candidates)  
**Records:** 200,000 candidates  
**Target:** `hired` (0 = Not Hired, 1 = Hired)

<<<<<<< HEAD
\---
=======




## Links

- **GitHub Repo:** https://github.com/samichohan/Project-Resume-MLOps-System

- **Live App:** https://project-resume-mlops-system.streamlit.app/

---
>>>>>>> 2f08a25415f0e13b2343123bed9fbe4f28f09628


## Project Structure

Project-Resume-MLOps-System/

├── src/
<<<<<<< HEAD
│   ├── extraction.py        # PySpark - Kaggle se data download
│   ├── cleaning.py          # PySpark - Data clean
│   ├── transformation.py    # PySpark - Encode + Split
│   ├── load.py              # PySpark - Data verify
│   ├── train.py             # PySpark ML - 4 models + MLflow
│   ├── best\_model.py        # MLflow - Best model register
│   └── prediction.py        # Single candidate prediction
├── .github/
│   └── workflows/
│       └── train.yml        # CI/CD GitHub Actions
├── data/
│   ├── raw/                 # Raw data
│   ├── processed/           # Cleaned data
│   └── transformed/         # Train/Test Parquet files
├── models/
│   └── pipeline/            # PySpark Pipeline saved
├── app.py                   # FastAPI prediction API
├── streamlit\_app.py         # Streamlit web UI
├── dvc.yaml                 # DVC pipeline stages
├── params.yaml              # Hyperparameters \& config
├── requirements.txt         # Python dependencies
└── README.md
=======

        │   ├── extraction.py        # PySpark - Kaggle se data download

        │   ├── cleaning.py          # PySpark - Data clean

        │   ├── transformation.py    # PySpark - Encode + Split

        │   ├── load.py              # PySpark - Data verify

        │   ├── train.py             # PySpark ML - 4 models + MLflow

        │   ├── best_model.py        # MLflow - Best model register

        │   └── prediction.py        # Single candidate prediction

├── .github/

          │   └── workflows/

                  │       └── train.yml        # CI/CD GitHub Actions

├── data/

          │   ├── raw/                 # Raw data

          │   ├── processed/           # Cleaned data

          │   └── transformed/         # Train/Test Parquet files

├── models/

        │   └── pipeline/            # PySpark Pipeline saved

\

├── app.py                   # FastAPI prediction API

├── streamlit_app.py         # Streamlit web UI

├── dvc.yaml                 # DVC pipeline stages

├── params.yaml              # Hyperparameters & config

├── requirements.txt         # Python dependencies

└── README.md

>>>>>>> 2f08a25415f0e13b2343123bed9fbe4f28f09628



\---

## Tech Stack

<<<<<<< HEAD
|Tool|Purpose|
|-|-|
|Python|Core language|
|PySpark|ETL - Data extraction, cleaning, transformation|
|Scikit-learn|ML models training|
|MLflow|Experiment tracking \& model registry|
|DVC|Data \& pipeline versioning|
|FastAPI|REST API for predictions|
|Streamlit|Web UI for predictions|
|GitHub Actions|CI/CD automation|
=======
| Tool | Purpose |
|------|---------|
| Python | Core language |
| PySpark | ETL - Data extraction, cleaning, transformation |
| Scikit-learn | ML models training |
| MLflow | Experiment tracking & model registry |
| DVC | Data & pipeline versioning |
| FastAPI | REST API for predictions |
| Streamlit | Web UI for predictions |
| GitHub Actions | CI/CD automation |
>>>>>>> 2f08a25415f0e13b2343123bed9fbe4f28f09628

\---

## PySpark Pipeline

Kaggle Dataset
↓
extraction.py    →  PySpark: CSV load → data/raw/
↓
cleaning.py      →  PySpark: Remove nulls, duplicates
↓
transformation.py → PySpark: StringIndexer + OHE + Scaler
↓
load.py          →  PySpark: Verify data
↓
train.py         →  PySpark ML: 4 models + MLflow track
↓
<<<<<<< HEAD
best\_model.py    →  MLflow: Best model register
↓
FastAPI + Streamlit → Prediction UI
=======
best_model.py    →  MLflow: Best model register
↓
FastAPI + Streamlit → Prediction UI

>>>>>>> 2f08a25415f0e13b2343123bed9fbe4f28f09628



\---

## Models Trained

|Model|Description|
|-|-|
|Logistic Regression|Simple linear model|
|Decision Tree|Tree-based decisions|
|Random Forest|Ensemble of trees|
|Gradient Boosting|Boosted ensemble|

\---

## Getting Started

### 1\. Clone the Repository

```bash
git clone https://github.com/samichohan/Project-Resume-MLOps-System.git
cd Project-Resume-MLOps-System
```

<<<<<<< HEAD
### 2\. Install Java 17

Download from: https://www.oracle.com/java/technologies/downloads/#java17-windows

### 3\. Install Dependencies
=======
### 2. Install Java 17
Download from: https://www.oracle.com/java/technologies/downloads/#java17-windows
>>>>>>> 2f08a25415f0e13b2343123bed9fbe4f28f09628

```bash
pip install -r requirements.txt
```

<<<<<<< HEAD
### 4\. Setup Kaggle API

* https://www.kaggle.com > Account > Settings > API > Create New Token
* `kaggle.json` → `C:\\Users\\YourName\\.kaggle\\kaggle.json`

### 5\. Run Pipeline Step by Step

```bash
python src/extraction.py
python src/cleaning.py
python src/transformation.py
python src/load.py
python src/train.py
python src/best\_model.py
```

### 6\. Run FastAPI

=======
### 4. Setup Kaggle API
- https://www.kaggle.com > Account > Settings > API > Create New Token
- `kaggle.json` → `C:\Users\YourName\.kaggle\kaggle.json`

### 5. Run Pipeline Step by Step
```bash
python src/extraction.py
python src/cleaning.py
python src/transformation.py
python src/load.py
python src/train.py
python src/best_model.py
```

### 6. Run FastAPI
>>>>>>> 2f08a25415f0e13b2343123bed9fbe4f28f09628
```bash
uvicorn app:app --reload
```

<<<<<<< HEAD
### 7\. Run Streamlit UI

```bash
streamlit run streamlit\_app.py
```

\---

## Links

* **GitHub Repo:** https://github.com/samichohan/Project-Resume-MLOps-System
* **Live App:** https://project-resume-mlops-system.streamlit.app/

\---

=======
### 7. Run Streamlit UI
```bash
streamlit run streamlit_app.py
```

>>>>>>> 2f08a25415f0e13b2343123bed9fbe4f28f09628
## Author

**Abdul Sami Chohan**  
GitHub: [@samichohan](https://github.com/samichohan)
<<<<<<< HEAD

=======
>>>>>>> 2f08a25415f0e13b2343123bed9fbe4f28f09628
