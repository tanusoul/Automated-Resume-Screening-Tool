import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# TEXT EXTRACTION
# ----------------------------

def extract_text_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# ----------------------------
# CLEANING
# ----------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ----------------------------
# LOAD RESUMES
# ----------------------------

def load_resumes(folder):
    resumes = []
    names = []

    for file in os.listdir(folder):
        if file.endswith(".txt"):
            path = os.path.join(folder, file)
            text = extract_text_from_txt(path)
            resumes.append(clean_text(text))
            names.append(file)

    return names, resumes

# ----------------------------
# LOAD JOB DESCRIPTION
# ----------------------------

def load_job_description(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return clean_text(f.read())

# ----------------------------
# RANKING
# ----------------------------

def rank_resumes(job_desc, resumes, names):
    documents = [job_desc] + resumes

    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform(documents)

    scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    results = pd.DataFrame({
        "Resume": names,
        "Score": scores
    })

    return results.sort_values(by="Score", ascending=False)

# ----------------------------
# SHORTLISTING
# ----------------------------

def shortlist_candidates(results, threshold=0.3):
    results["Status"] = results["Score"].apply(
        lambda x: "Shortlisted" if x >= threshold else "Rejected"
    )
    return results

# ----------------------------
# MAIN
# ----------------------------

if __name__ == "__main__":
    resume_folder = "resumes"
    job_desc_path = "data/job_description.txt"

    names, resumes = load_resumes(resume_folder)
    job_desc = load_job_description(job_desc_path)

    results = rank_resumes(job_desc, resumes, names)

    # Apply shortlist logic
    final_results = shortlist_candidates(results, threshold=0.3)

    print("\n📊 FINAL SCREENING RESULTS:\n")
    print(final_results)

    # Save CSV
    output_path = "outputs/results.csv"
    final_results.to_csv(output_path, index=False)

    print(f"\n✅ Results saved to: {output_path}")