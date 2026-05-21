import streamlit as st
import requests

st.set_page_config(
    page_title="Resume Hiring Prediction",
    page_icon="📄",
    layout="wide"
)

FASTAPI_URL = "http://localhost:8000"

st.title("📄 Resume Hiring Prediction")
st.markdown("### Candidate Information")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Personal Info")
    age = st.slider("Age", min_value=18, max_value=60, value=25)
    education_level = st.selectbox(
        "Education Level",
        ["Bachelors", "Masters", "Phd", "High School"]
    )
    university_tier = st.selectbox(
        "University Tier",
        ["Tier 1", "Tier 2", "Tier 3"]
    )
    cgpa = st.slider("CGPA", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
    company_type = st.selectbox(
        "Company Type",
        ["MNC", "Startup", "Government", "Mid-size"]
    )

with col2:
    st.subheader("💼 Experience & Skills")
    internships = st.number_input("Total Internships", min_value=0, max_value=20, value=0)
    projects = st.number_input("Total Projects", min_value=0, max_value=50, value=1)
    programming_languages = st.number_input("Programming Languages", min_value=0, max_value=20, value=1)
    certifications = st.number_input("Certifications", min_value=0, max_value=20, value=0)
    experience_years = st.slider("Experience (years)", min_value=0.0, max_value=30.0, value=0.0, step=0.5)
    hackathons = st.number_input("Hackathons", min_value=0, max_value=20, value=0)
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
    with st.spinner("Processing... ⏳"):
        payload = {
            "age": age,
            "education_level": education_level,
            "university_tier": university_tier,
            "cgpa": cgpa,
            "internships": int(internships),
            "projects": int(projects),
            "programming_languages": int(programming_languages),
            "certifications": int(certifications),
            "experience_years": experience_years,
            "hackathons": int(hackathons),
            "research_papers": int(research_papers),
            "skills_score": skills_score,
            "soft_skills_score": soft_skills_score,
            "resume_length_words": int(resume_length_words),
            "company_type": company_type
        }

        try:
            response = requests.post(
                f"{FASTAPI_URL}/predict",
                json=payload
            )

            if response.status_code == 200:
                result = response.json()

                st.markdown("## 🎯 Prediction Result")

                if result["hired"] == 1:
                    st.success(f"## ✅ {result['verdict']}")
                    
                else:
                    st.error(f"## ❌ {result['verdict']}")

                st.markdown(f"### Probability: **{result['probability']*100:.1f}%**")
                st.progress(float(result["probability"]))

                st.markdown("---")
                st.markdown("**Input Data:**")
                st.json(payload)

            else:
                st.error(f"❌ Error: {response.json()}")

        except requests.exceptions.ConnectionError:
            st.error("❌ FastAPI chal nahi rahi! Pehle chalao:")
            st.code("uvicorn app:app --reload")

st.markdown("---")
st.markdown("*Resume Hiring Prediction MLOps Project*")