import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
import torch
import os

st.set_page_config(page_title="AI Resume & Portfolio Builder", layout="centered")

st.title("🤖 AI Resume & Portfolio Builder")

# --------------------------------------------------
# ML ROLE PREDICTION MODEL
# --------------------------------------------------

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

model_ml = MultinomialNB()
model_ml.fit(X, y)

st.success("✅ ML Role Prediction Model Ready")

# --------------------------------------------------
# LOAD FLAN-T5 SMALL (STABLE)
# --------------------------------------------------

@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model

tokenizer, model = load_model()

# --------------------------------------------------
# TEXT GENERATION FUNCTION
# --------------------------------------------------
def generate_text(prompt, max_tokens=300):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
        **inputs,
        max_length=max_tokens,
        num_beams=6,
        temperature=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

name = st.text_input("Enter your name:")
email = st.text_input("Enter your email:")
skills_input = st.text_area("Enter your skills:")
project_title = st.text_input("Enter your project title:")
project_desc = st.text_area("Describe your project:")

# --------------------------------------------------
# GENERATE BUTTON
# --------------------------------------------------

if st.button("Generate Resume & Portfolio"):

    if not name or not email or not skills_input:
        st.warning("⚠ Please fill all required fields.")
    else:
        skills_vector = vectorizer.transform([skills_input])
        predicted_role = model_ml.predict(skills_vector)[0]

        st.subheader("🎯 Predicted Job Role")
        st.success(predicted_role)

        objective_prompt = f"""
Write a professional and impactful career objective for a {predicted_role}
who has skills in {skills_input}.

The objective should sound confident, highlight technical strengths,
and express interest in contributing to an organization.
Write it as a smooth paragraph.
"""

        bio_prompt = f"""
Write a professional third-person bio for {name},
an aspiring {predicted_role} skilled in {skills_input}.

The bio should describe technical expertise, analytical ability,
and commitment to professional growth.
Write it as a smooth and natural paragraph.
"""

        project_prompt = f"""
Write a professional project description.

Project Title: {project_title}
Project Details: {project_desc}

Explain the purpose of the project, technologies involved,
and the impact or benefits of the solution.
Write it clearly as a detailed paragraph.
"""

        objective = generate_text(objective_prompt, 150)
        bio = generate_text(bio_prompt, 200)
        project_text = generate_text(project_prompt, 250)

        st.subheader("📝 AI Career Objective")
        st.write(objective)

        st.subheader("👤 AI Professional Bio")
        st.write(bio)

        st.subheader("🚀 AI Project Description")
        st.write(project_text)

        # Build Resume Text
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

        # Build Portfolio Text
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

        # --------------------------------------------------
        # PDF GENERATION
        # --------------------------------------------------

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













