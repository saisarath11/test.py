import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from transformers import pipeline
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
import os

st.set_page_config(page_title="AI Resume & Portfolio Builder", layout="centered")

st.title("🤖 AI Resume & Portfolio Builder (Full AI Version)")

# ------------------------------
# ML ROLE PREDICTION MODEL
# ------------------------------

data = {
    "skills": [
        "python machine learning data analysis pandas",
        "html css javascript react",
        "aws cloud docker kubernetes",
        "network security ethical hacking cryptography",
        "deep learning neural networks artificial intelligence"
    ],
    "role": [
        "Data Scientist",
        "Web Developer",
        "Cloud Engineer",
        "Cyber Security Analyst",
        "AI Engineer"
    ]
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["skills"])
y = df["role"]

model = MultinomialNB()
model.fit(X, y)

st.success("✅ ML Role Prediction Model Ready")

# ------------------------------
# LOAD FLAN-T5 (INSTRUCTION MODEL)
# ------------------------------

@st.cache_resource
def load_model():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device=-1
    )

generator = load_model()

# ------------------------------
# USER INPUTS
# ------------------------------

name = st.text_input("Enter your name:")
email = st.text_input("Enter email:")
skills_input = st.text_area("Enter your skills:")
project_title = st.text_input("Enter your project title:")
project_desc = st.text_area("Describe your project:")

# ------------------------------
# GENERATE BUTTON
# ------------------------------

if st.button("Generate Resume & Portfolio"):

    if not name or not email or not skills_input:
        st.warning("⚠ Please fill all required fields.")
    else:

        # Predict Role
        skills_vector = vectorizer.transform([skills_input])
        predicted_role = model.predict(skills_vector)[0]

        st.subheader("🎯 Predicted Job Role")
        st.success(predicted_role)

        # ------------------------------
        # FLAN-T5 PROMPTS
        # ------------------------------

        objective_prompt = f"""
Write a professional 3-4 line career objective for a {predicted_role}
with skills in {skills_input}.
Keep it formal and concise.
"""

        bio_prompt = f"""
Write a professional short bio for {name}, who is an aspiring
{predicted_role} with skills in {skills_input}.
Keep it under 5 lines.
"""

        project_prompt = f"""
Write a professional project description for a project titled
'{project_title}'. The project involves {project_desc}.
Keep it clear and technical.
"""

        # Generate outputs
        objective = generator(objective_prompt, max_length=150)[0]["generated_text"]
        bio = generator(bio_prompt, max_length=200)[0]["generated_text"]
        project_text = generator(project_prompt, max_length=250)[0]["generated_text"]

        # ------------------------------
        # DISPLAY OUTPUT
        # ------------------------------

        st.subheader("📝 AI Career Objective")
        st.write(objective)

        st.subheader("👤 AI Professional Bio")
        st.write(bio)

        st.subheader("🚀 AI Project Description")
        st.write(project_text)

        # ------------------------------
        # BUILD RESUME TEXT
        # ------------------------------

        resume_text = f"""
{name}
Email: {email}

Predicted Role: {predicted_role}

Career Objective:
{objective}

Skills:
{skills_input}

Project:
{project_text}
"""

        st.subheader("📄 Generated Resume")
        st.text(resume_text)

        portfolio_text = f"""
Name: {name}
Email: {email}

Role: {predicted_role}

Professional Bio:
{bio}

Skills:
{skills_input}

Project Summary:
{project_text}
"""

        st.subheader("🌐 Generated Portfolio")
        st.text(portfolio_text)

        # ------------------------------
        # PDF GENERATION
        # ------------------------------

        file_name = "AI_Resume_and_Portfolio.pdf"
        doc = SimpleDocTemplate(file_name, pagesize=A4)
        elements = []
        styles = getSampleStyleSheet()
        style = styles["Normal"]

        full_text = resume_text + "\n\n" + portfolio_text

        for line in full_text.split("\n"):
            elements.append(Paragraph(line, style))
            elements.append(Spacer(1, 8))

        doc.build(elements)

        with open(file_name, "rb") as f:
            st.download_button(
                "⬇ Download Resume & Portfolio PDF",
                f,
                file_name=file_name,
                mime="application/pdf"
            )

        os.remove(file_name)
