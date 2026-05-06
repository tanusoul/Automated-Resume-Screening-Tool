import streamlit as st
import pandas as pd
import re
import pdfplumber
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Resume Screening Tool",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# MODERN CSS
# ----------------------------
st.markdown("""
<style>
    .main {
        background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .hero {
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
        padding: 24px 28px;
        border-radius: 18px;
        color: white;
        box-shadow: 0 10px 30px rgba(37, 99, 235, 0.18);
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 800;
        color: white;
    }
    .hero p {
        margin: 0.4rem 0 0 0;
        opacity: 0.95;
        font-size: 1rem;
    }
    .metric-card {
        background: white;
        padding: 18px 20px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    }
    .section-card {
        background: white;
        padding: 18px 20px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    }
    .stTextArea textarea {
        border-radius: 14px;
        border: 1px solid #dbe3f0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.65rem 1.1rem;
        font-weight: 600;
        box-shadow: 0 8px 18px rgba(99, 102, 241, 0.22);
    }
    .stButton > button:hover {
        opacity: 0.95;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# FUNCTIONS
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s+#.-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_text(uploaded_file):
    file_name = uploaded_file.name.lower()
    try:
        if file_name.endswith(".pdf"):
            text = ""
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    text += (page.extract_text() or "") + " "
            return text.strip()
        elif file_name.endswith(".docx"):
            doc = Document(uploaded_file)
            return " ".join([para.text for para in doc.paragraphs]).strip()
        elif file_name.endswith(".txt"):
            return uploaded_file.read().decode("utf-8", errors="ignore").strip()
    except:
        return ""
    return ""

def rank_resumes(job_desc, resumes, names):
    documents = [job_desc] + resumes
    tfidf = TfidfVectorizer(stop_words="english")
    vectors = tfidf.fit_transform(documents)
    scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    df = pd.DataFrame({
        "Resume": names,
        "Score": [round(float(s), 4) for s in scores]
    })
    return df.sort_values(by="Score", ascending=False).reset_index(drop=True)

# ----------------------------
# UI
# ----------------------------
st.markdown("""
<div class="hero">
    <h1>📄 Automated Resume Screening Tool</h1>
    <p>Upload resumes, compare with job description, get ranked results instantly.</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-card"><h4>📁 Supported Files</h4><p>PDF, DOCX, TXT</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card"><h4>🎯 Matching</h4><p>TF-IDF + Cosine</p></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card"><h4>📊 Output</h4><p>Ranking + CSV</p></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-card"><h4>⚡ Use Case</h4><p>ATS Screening</p></div>', unsafe_allow_html=True)

st.write("")

left, right = st.columns([1.1, 0.9])

with left:
    st.markdown("### Job Description")
    job_desc = st.text_area(
        "Paste the job description",
        height=260,
        placeholder="Example: Python Developer with Pandas, NumPy, SQL, Excel, Machine Learning..."
    )

with right:
    st.markdown("### Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload multiple resumes",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    st.markdown("### Threshold")
    threshold = st.slider("Shortlist Threshold", 0.0, 1.0, 0.30, 0.01)

analyze = st.button("🚀 Analyze Resumes", use_container_width=True)

if analyze:
    if not job_desc.strip() or not uploaded_files:
        st.warning("Please provide job description and resumes.")
        st.stop()

    resumes = []
    names = []

    progress = st.progress(0)
    for i, file in enumerate(uploaded_files):
        text = extract_text(file)
        resumes.append(clean_text(text))
        names.append(file.name)
        progress.progress((i + 1) / len(uploaded_files))

    job_desc_clean = clean_text(job_desc)
    results = rank_resumes(job_desc_clean, resumes, names)

    results["Status"] = results["Score"].apply(
        lambda x: "✅ Shortlisted" if x >= threshold else "❌ Rejected"
    )

    shortlisted = sum(results["Status"].str.contains("✅"))
    avg_score = round(results["Score"].mean(), 4)

    st.markdown("### Results")
    st.dataframe(
        results.style.background_gradient(subset=["Score"], cmap="Blues")
        .format({"Score": "{:.4f}"}),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Download")
    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download CSV",
        data=csv,
        file_name="resume_results.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.success(f"✅ Top candidate: {results.iloc[0]['Resume']}")
    st.info(f"📊 {shortlisted}/{len(uploaded_files)} shortlisted, avg score: {avg_score}")