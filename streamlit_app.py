import streamlit as st

st.set_page_config(
    page_title="Resume Hiring Prediction",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Resume Hiring Prediction")
st.markdown("### Candidate fill Information")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Personal Info")
    age = st.slider("Age", min_value=18, max_value=60, value=25)
    education_level = st.selectbox("Education Level", ["Bachelors", "Masters", "PhD", "High School"])
    university_tier = st.selectbox("University Tier", ["Tier 1", "Tier 2", "Tier 3"])
    cgpa = st.slider("CGPA", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
    company_type = st.selectbox("Company Type", ["MNC", "Startup", "Government", "Mid-size"])

with col2:
    st.subheader("💼 Experience & Skills")
    internships = st.number_input("Total Internships", min_value=0, max_value=20, value=0)
    projects = st.number_input("Total Projects", min_value=0, max_value=50, value=1)
    programming_languages = st.number_input("Programming Languages", min_value=0, max_value=20, value=1)
    certifications = st.number_input("Certifications", min_value=0, max_value=20, value=0)
    experience_years = st.slider("Experience (years)", min_value=0.0, max_value=30.0, value=0.0, step=0.5)
    hackathons = st.number_input("Total Hackathons", min_value=0, max_value=20, value=0)
    research_papers = st.number_input("Research Papers", min_value=0, max_value=20, value=0)

st.markdown("---")
col3, col4 = st.columns(2)

with col3:
    skills_score = st.slider("Technical Skills Score", min_value=0.0, max_value=50.0, value=10.0, step=0.5)
with col4:
    soft_skills_score = st.slider("Soft Skills Score", min_value=0.0, max_value=20.0, value=5.0, step=0.1)

resume_length_words = st.slider("Resume Word Count", min_value=50, max_value=1000, value=300)

st.markdown("---")

if st.button("🔍 Predict!", use_container_width=True):
    import pickle
    import pandas as pd
    import os

    EDUCATION_MAP  = {"Bachelors": 0, "Masters": 1, "PhD": 2, "High School": 3}
    UNIVERSITY_MAP = {"Tier 1": 0, "Tier 2": 1, "Tier 3": 2}
    COMPANY_MAP    = {"Government": 0, "MNC": 1, "Startup": 2, "Mid-size": 3}

    with st.spinner("Model is Running... ⏳"):
        try:
            import glob
            files = glob.glob("mlruns/**/*.pkl", recursive=True)
            if not files:
                st.error("Model nahi mila! Pehle train.py chalao.")
            else:
                latest = max(files, key=os.path.getmtime)
                with open(latest, "rb") as f:
                    model = pickle.load(f)

                input_data = pd.DataFrame([{
                    "age": age,
                    "education_level": EDUCATION_MAP.get(education_level, 0),
                    "university_tier": UNIVERSITY_MAP.get(university_tier, 1),
                    "cgpa": cgpa,
                    "internships": internships,
                    "projects": projects,
                    "programming_languages": programming_languages,
                    "certifications": certifications,
                    "experience_years": experience_years,
                    "hackathons": hackathons,
                    "research_papers": research_papers,
                    "skills_score": skills_score,
                    "soft_skills_score": soft_skills_score,
                    "resume_length_words": resume_length_words,
                    "company_type": COMPANY_MAP.get(company_type, 1),
                }])

                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data)[0][1]

                st.markdown("## 🎯 Prediction Result")
                if prediction == 1:
                    st.success(f"## ✅ Hired!")
                    st.balloons()
                else:
                    st.error(f"## ❌ Not Hired")

                st.markdown(f"### Probability: **{probability*100:.1f}%**")
                st.progress(float(probability))

        except Exception as e:
            st.error(f"Error: {str(e)}")

st.markdown("---")
st.markdown("*Resume Hiring Prediction MLOps Project*")