# streamlit_app.py

import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from groq import Groq
import logging
import pdfplumber
from io import StringIO

# Setup
logging.basicConfig(level=logging.INFO)
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# Helper

def call_llm(prompt: str, model: str = "mixtral-8x7b-32768") -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a professional VC data room analyst."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

# File parsing

def extract_text_from_pdf(file) -> str:
    try:
        with pdfplumber.open(file) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as e:
        return f"Error reading PDF: {e}"

def extract_text_from_csv(file) -> str:
    try:
        df = pd.read_csv(file)
        return df.to_string(index=False)
    except Exception as e:
        return f"Error reading CSV: {e}"

# Modules

def generate_summary_from_files(file_texts: str, founder_name: str) -> str:
    prompt = (
        f"You are a due diligence assistant for founder {founder_name}. Based on the following documents, create a summary of traction, risks, and investor-readiness.\n"
        f"Documents: {file_texts[:6000]}"
    )
    return call_llm(prompt)

def investor_qa(question: str, file_texts: str) -> str:
    prompt = (
        f"A VC asks: {question}\n"
        f"Answer using the following startup documentation: {file_texts[:6000]}"
    )
    return call_llm(prompt)

# Streamlit UI

def main():
    st.set_page_config("DataRoom AI", page_icon="🗂️", layout="wide")
    st.title("🗂️ DataRoom AI")
    st.write("Upload your startup documents. Let AI answer due diligence questions and generate investor summaries.")

    founder = st.text_input("Your name (for personalization)")
    uploaded_files = st.file_uploader("Upload Documents (PDF or CSV)", type=["pdf", "csv"], accept_multiple_files=True)

    file_texts = ""
    if uploaded_files:
        for file in uploaded_files:
            if file.name.endswith(".pdf"):
                file_texts += extract_text_from_pdf(file) + "\n"
            elif file.name.endswith(".csv"):
                file_texts += extract_text_from_csv(file) + "\n"
        st.success("Files processed.")

    tab1, tab2 = st.tabs(["📄 Generate Deal Summary", "💬 Investor Q&A"])

    with tab1:
        st.subheader("📄 Auto-Generated Deal Summary")
        if st.button("Generate Summary"):
            if not file_texts:
                st.error("Please upload documents first.")
            else:
                summary = generate_summary_from_files(file_texts, founder or "Founder")
                st.text_area("Deal Summary", value=summary, height=400)

    with tab2:
        st.subheader("💬 Ask Investor-Style Questions")
        question = st.text_input("Ask a due diligence question")
        if st.button("Answer"):
            if not question:
                st.error("Enter a question.")
            elif not file_texts:
                st.error("Please upload documents first.")
            else:
                response = investor_qa(question, file_texts)
                st.markdown(f"**AI:** {response}")

if __name__ == "__main__":
    main()
