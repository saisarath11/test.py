import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import pipeline

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="AI Resume & Portfolio Builder",
    page_icon="🚀",
    layout="wide"
)

# -------------------------------
# CUSTOM CSS (Premium UI)
# -------------------------------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
h1, h2, h3 {
    color: #4CAF50;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

st.title("🚀 AI Resume & Portfolio Builder")

# -------------------------------
# SIMPLE TRAINING DATA (AI Role Prediction)
# -------------------------------
data = [
    ("python machine learning data analysis pandas numpy", "Data Scientist"),
    ("html css javascript react frontend ui ux", "Frontend Developer"),
    ("java spring boot backend api database", "Backend Developer"),
    ("c c++ embedded systems hardware microcontroller", "Embedded Engineer")
]

texts = [x[0] for x in data]
labels = [x[1] for x in data]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model_ml = LogisticRegression()
model_ml.fit(X, labels)

# -------------------------------
# LOAD GENERATIVE MODEL
# -------------------------------
@st.cache_resource
def load_model():
    return pipeline(
        task="text2text-generation",
        model="google/flan-t5-base",
        device=-1
    )

generator = load_model()

# -------------------------------
# GENERATION FUNCTION
# -------------------------------
def generate_text(prompt, max_tokens=150):
    response = generator(
        prompt,
        max_length=max_tokens,
        temperature=0.8,
        num_beams=4,
        repetition_penalty=1.2
    )
    return response[0]["generated_text"]

# -------------------------------
# USER INPUT SECTION
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    name = st.text_input("Full Name")
    email = st.text_input("Email")
    skills_input = st.text_area("Enter Your Skills")

with col2:
    project_title = st.text_input("Project Title")
    project_desc = st.text_area("Project Description")

# -------------------------------
# GENERATE BUTTON
# -------------------------------
if st.button("Generate Portfolio"):

    # Predict Role
    skills_vector = vectorizer.transform([skills_input])
    predicted_role = model_ml.predict(skills_vector)[0]

    st.success(f"🎯 Predicted Job Role: {predicted_role}")

    # PROMPTS
    objective_prompt = f"""
Generate a professional career objective in 2-3 sentences.

Role: {predicted_role}
Skills: {skills_input}

Make it sound confident, highlight technical ability,
and mention contributing to an organization.
Output:
"""

    bio_prompt = f"""
 Generate a professional third-person bio in 2-3 sentences.

Name: {name}
Role: {predicted_role}
Skills: {skills_input}

Mention technical expertise, analytical thinking,
and passion for innovation.
Output:
"""
    project_prompt = f"""
  Generate a professional project description.

Project Name: {project_title}
Details: {project_desc}

Output:
"""

    # Generate Content
    objective = generate_text(objective_prompt)
    bio = generate_text(bio_prompt)
    project_text = generate_text(project_prompt)

    # Display Sections
    st.markdown("## 📝 Career Objective")
    st.info(objective)

    st.markdown("## 👤 Professional Bio")
    st.success(bio)

    st.markdown("## 🚀 Project Description")
    st.warning(project_text)

    # -------------------------------
    # GENERATED RESUME TEXT
    # -------------------------------
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

    st.markdown("## 📄 Generated Resume")
    st.text_area("Resume Preview", resume_text, height=300)

    st.download_button(
        label="📥 Download Resume",
        data=resume_text,
        file_name="Resume.txt",
        mime="text/plain"
    )

















