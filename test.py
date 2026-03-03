import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import pipeline

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="AI Resume & Portfolio Builder",
    page_icon="🚀",
    layout="wide"
)

# -----------------------------------
# PREMIUM UI STYLE
# -----------------------------------
st.markdown("""
<style>
.main { background-color: #0e1117; }
h1, h2, h3 { color: #4CAF50; }
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

st.title("🚀 AI Resume & Portfolio Builder")

# -----------------------------------
# SIMPLE ML MODEL (ROLE PREDICTION)
# -----------------------------------
data = [
    ("python machine learning data analysis pandas numpy", "Data Scientist"),
    ("html css javascript react ui ux", "Frontend Developer"),
    ("java spring boot api backend database", "Backend Developer"),
    ("c c++ embedded systems microcontroller", "Embedded Engineer")
]

texts = [x[0] for x in data]
labels = [x[1] for x in data]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

ml_model = LogisticRegression()
ml_model.fit(X, labels)

# -----------------------------------
# LOAD GENERATIVE MODEL
# -----------------------------------
@st.cache_resource
def load_model():
    return pipeline(
        task="text2text-generation",
        model="google/flan-t5-base",
        device=-1
    )

generator = load_model()

# -----------------------------------
# GENERATION FUNCTION
# -----------------------------------
def generate_text(prompt):
    response = generator(
        prompt,
        max_new_tokens=160,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.4,
        no_repeat_ngram_size=3
    )
    return response[0]["generated_text"]

# -----------------------------------
# INPUT SECTION
# -----------------------------------
col1, col2 = st.columns(2)

with col1:
    name = st.text_input("Full Name")
    email = st.text_input("Email")
    skills_input = st.text_area("Enter Your Skills")

with col2:
    project_title = st.text_input("Project Title")
    project_desc = st.text_area("Project Description")

# -----------------------------------
# GENERATE BUTTON
# -----------------------------------
if st.button("Generate Portfolio"):

    # Predict Role
    skills_vector = vectorizer.transform([skills_input])
    predicted_role = ml_model.predict(skills_vector)[0]

    st.success(f"🎯 Predicted Job Role: {predicted_role}")

    # -----------------------------------
    # PROMPTS
    # -----------------------------------
    objective_prompt = f"""
    Generate a professional career objective for a college student
    aspiring to become a {predicted_role}.
    Skills: {skills_input}
    Write 2-3 professional sentences highlighting learning,
    technical ability, and growth mindset.
    Output:
    """

    bio_prompt = f"""
    Generate a professional third-person bio for {name},
    a college student aspiring to become a {predicted_role}.
    Skills: {skills_input}
    Do not mention any company, university, or location.
    Write 2-3 professional sentences.
    Output:
    """

    project_prompt = f"""
    Generate a professional project description.

    Project Name: {project_title}
    Details: {project_desc}

    Explain the purpose, technologies used,
    and the impact of the project.
    Output:
    """

    # ✅ PORTFOLIO PROMPT (ADDED)
    portfolio_prompt = f"""
    Generate a professional portfolio summary for {name},
    a college student aspiring to become a {predicted_role}.
    Skills: {skills_input}
    Project: {project_title}
    Do not mention any company or location.
    Write 3-4 professional sentences.
    Output:
    """

    # -----------------------------------
    # GENERATE CONTENT
    # -----------------------------------
    objective = generate_text(objective_prompt)
    bio = generate_text(bio_prompt)
    project_text = generate_text(project_prompt)
    portfolio_text = generate_text(portfolio_prompt)

    # -----------------------------------
    # DISPLAY OUTPUT
    # -----------------------------------
    st.markdown("## 📝 Career Objective")
    st.info(objective)

    st.markdown("## 👤 Professional Bio")
    st.success(bio)

    st.markdown("## 🚀 Project Description")
    st.warning(project_text)

    # ✅ DISPLAY PORTFOLIO
    st.markdown("## 🌐 Portfolio Summary")
    st.success(portfolio_text)

    # -----------------------------------
    # GENERATED RESUME TEXT
    # -----------------------------------
    resume_text = f"""
{name}
Email: {email}

Predicted Role: {predicted_role}

Career Objective:
{objective}

Professional Bio:
{bio}

Skills:
{skills_input}

Project Description:
{project_text}

Portfolio Summary:
{portfolio_text}
"""

    st.markdown("## 📄 Generated Resume")
    st.text_area("Resume Preview", resume_text, height=350)

    st.download_button(
        label="📥 Download Resume",
        data=resume_text,
        file_name="Resume.txt",
        mime="text/plain"
    )





















