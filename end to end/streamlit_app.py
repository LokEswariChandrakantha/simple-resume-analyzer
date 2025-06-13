import streamlit as st
import pdfplumber
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from fpdf import FPDF
import unicodedata
import re

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

st.title("AI-Powered Resume Analyzer")

# Upload Resume
resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

# Upload Job Description
job_desc_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])

if resume_file and job_desc_file:
    # Extract Resume Text
    with pdfplumber.open(resume_file) as pdf:
        resume_text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    
    # Extract Job Description Text
    with pdfplumber.open(job_desc_file) as pdf:
        job_description = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

    st.write("### Extracted Resume Text:")
    st.text(resume_text[:500])  # Show first 500 characters

    st.write("### Extracted Job Description Text:")
    st.text(job_description[:500])  # Show first 500 characters

    # Compute Similarity Score
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    job_embedding = model.encode(job_description, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(resume_embedding, job_embedding).item() * 100
    similarity_score = round(similarity_score, 2)

    st.write(f"### Matching Score: {similarity_score}/100")

    # ✅ Fix Unicode Issues
    def clean_text(text):
        text = text.replace("\u2013", "-").replace("\u2014", "--")  # Replace en dash & em dash
        text = unicodedata.normalize("NFKD", text).encode("latin-1", "ignore").decode("latin-1")  # Remove unsupported characters
        return text

    resume_text = clean_text(resume_text)
    job_description = clean_text(job_description)

    # ✅ Identify Missing Keywords in Resume
    job_keywords = re.findall(r'\b\w+\b', job_description.lower())
    resume_words = set(re.findall(r'\b\w+\b', resume_text.lower()))
    missing_keywords = [word for word in job_keywords if word not in resume_words]
    missing_keywords = list(set(missing_keywords))
    missing_keywords_text = ", ".join(missing_keywords[:50])

    # ✅ Determine Candidate Fit
    if similarity_score > 80:
        fit_assessment = "The candidate is an excellent fit for the role."
    elif 60 <= similarity_score <= 80:
        fit_assessment = "The candidate is a moderate fit, with some improvements needed."
    else:
        fit_assessment = "The candidate requires significant improvements to be a good fit for the role."

    # ✅ Areas of Improvement (Expanded)
    improvement_areas = """
    1. **Technical Skills**: The candidate should focus on enhancing technical competencies that are explicitly required in the job description. Consider gaining expertise in missing programming languages, frameworks, or tools.
    2. **Soft Skills**: Improving soft skills such as communication, leadership, and teamwork can be beneficial for collaborative roles. Engaging in workshops or practical experiences can help.
    3. **Relevant Experience**: There may be gaps in relevant work experience related to the job description. The candidate should highlight past projects, internships, or freelance work that align with job expectations.
    4. **Certifications & Education**: Obtaining relevant certifications or further education in key areas mentioned in the job description can strengthen the resume.
    5. **Resume Optimization**: The resume can be further optimized by using industry-specific keywords, making it more ATS-friendly and improving overall visibility to recruiters.
    """

    # ✅ Generate PDF Report
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Resume Analysis Report", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Matching Score:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"{similarity_score}/100")
    pdf.ln(5)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Candidate Fit Assessment:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, fit_assessment)
    pdf.ln(10)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Key Job Requirements Missing in Resume:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"{missing_keywords_text if missing_keywords else 'No major mismatches detected!'}")
    pdf.ln(10)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Areas for Improvement:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, improvement_areas)
    pdf.ln(10)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Job Description (Preview):", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"{job_description[:1000]}...")
    pdf.ln(5)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Resume Content (Preview):", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"{resume_text[:1000]}...")
    pdf.ln(5)

    pdf_filename = "resume_report.pdf"
    pdf.output(pdf_filename)

    # Provide Download Button
    with open(pdf_filename, "rb") as file:
        st.download_button("Download Report", file, file_name=pdf_filename, mime="application/pdf")

    st.success("✅ Analysis complete! Download your report.")