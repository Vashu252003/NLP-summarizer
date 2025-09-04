import streamlit as st
def local_css(css: str):
    import streamlit as st
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
local_css("""
    /* App background - soft gray */
    /* App background - dark blue */
.stApp {
    background-color: #1E3A8A;  /* Dark blue */
    color: #FFFFFF;             /* Light text for contrast */
    font-family: 'Segoe UI', sans-serif;
}

/* Main title */
h1 {
    color: #FFFFFF;
    text-align: center;
    font-weight: 700;
    margin-bottom: 20px;
}

/* Subtitles */
h2, h3 {
    color: #E0E7FF;  /* Lighter blue for headings */
    font-weight: 600;
}

/* Radio buttons */
.stRadio label {
    font-size: 17px;
    color: #FFFFFF !important;
}

/* Textarea */
textarea {
    background-color: #FFFFFF !important;  /* Keep input white for readability */
    color: #0F172A !important;             /* Dark text inside textarea */
    border-radius: 12px !important;
    border: 1px solid #CBD5E1 !important;
    padding: 14px !important;
    font-size: 15px !important;
    line-height: 1.5em !important;
}

/* Placeholder text in textarea */
textarea::placeholder {
    color: #94A3B8 !important;  /* Light gray */
}

/* Buttons */
.stButton>button {
    background-color: #3B82F6;
    color: white;
    font-size: 16px;
    border-radius: 10px;
    padding: 10px 24px;
    transition: all 0.3s ease;
    border: none;
}
.stButton>button:hover {
    background-color: #2563EB;
    transform: scale(1.03);
}

/* Summary output box */
.stMarkdown {
    background: #1E40AF;   /* Slightly lighter dark blue */
    border: 1px solid #2563EB;
    border-radius: 12px;
    padding: 18px;
    font-size: 16px;
    color: #FFFFFF;
    line-height: 1.6em;
}

""")

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize

# PDF text extraction
from PyPDF2 import PdfReader

# Try importing transformers for abstractive summarization
try:
    from transformers import pipeline
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False


# ----------------------------
# Extractive summarizer (TF-IDF)
# ----------------------------
def summarize_extractive(text: str, n_sentences: int = 3) -> str:
    """Simple extractive summarizer using TF-IDF scoring of sentences."""
    if not text or not text.strip():
        return ""

    try:
        nltk.data.find("tokenizers/punkt")
    except Exception:
        nltk.download("punkt")

    sentences = sent_tokenize(text)
    if len(sentences) <= n_sentences:
        return "\n".join(sentences)

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(sentences)

    import numpy as np
    scores = tfidf_matrix.sum(axis=1).A1
    top_n_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_sentences]
    top_n_idx.sort()

    selected = [sentences[i] for i in top_n_idx]
    return "\n".join(selected)


# ----------------------------
# Abstractive summarizer (HF)
# ----------------------------
class AbstractiveSummarizer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn", device: int = -1):
        if not HF_AVAILABLE:
            raise RuntimeError("transformers not available. Install `transformers` and `torch`.")
        self.pipeline = pipeline("summarization", model=model_name, device=device)

    def summarize(self, text: str, max_length: int = 150, min_length: int = 40, do_sample: bool = False) -> str:
        if not text or not text.strip():
            return ""

        if len(text) <= 3000:
            res = self.pipeline(text, max_length=max_length, min_length=min_length, do_sample=do_sample)
            return res[0]["summary_text"].strip()
        else:
            paras = [p for p in text.split("\n\n") if p.strip()]
            chunks, curr = [], ""
            for p in paras:
                if len(curr) + len(p) < 3000:
                    curr += "\n\n" + p
                else:
                    chunks.append(curr.strip())
                    curr = p
            if curr.strip():
                chunks.append(curr.strip())

            partial_summaries = []
            for chunk in chunks:
                out = self.pipeline(chunk, max_length=max_length, min_length=min_length, do_sample=do_sample)
                partial_summaries.append(out[0]["summary_text"].strip())

            concat = "\n".join(partial_summaries)
            out = self.pipeline(concat, max_length=max_length, min_length=min_length, do_sample=do_sample)
            return out[0]["summary_text"].strip()


# ----------------------------
# Helper: Read file content
# ----------------------------
def read_file(uploaded_file):
    if uploaded_file.name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")
    elif uploaded_file.name.endswith(".pdf"):
        pdf = PdfReader(uploaded_file)
        text = ""
        for page in pdf.pages:
            try:
                text += page.extract_text() or ""
            except Exception:
                continue  # skip problematic pages
            text += "\n"
        return text
    else:
        st.error("Unsupported file type. Please upload a .txt or .pdf file.")
        return ""

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📝 AI Text Summarizer")
st.subheader("Choose how you want your summary: Quick 📌 or Detailed 📖")

st.write("Summarize text from **paste box**, **.txt file**, or **.pdf file**")

# Option: paste text OR upload file
tab1, tab2 = st.tabs(["✍️ Paste Text", "📂 Upload File"])

input_text = ""
with tab1:
    input_text = st.text_area("Enter your text:", height=300)

with tab2:
    uploaded_file = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])
    if uploaded_file is not None:
        input_text = read_file(uploaded_file)

# Method choice
# Show simpler options to user
choice = st.radio("Choose summary style:", ["Quick Summary", "Detailed Summary"])

# Map user-friendly labels to internal methods
if choice == "Quick Summary":
    method = "Extractive"
else:
    method = "Abstractive"


# Extractive: ask for number of sentences
num_sentences = None
if method == "Extractive":
    num_sentences = st.slider("Number of sentences:", min_value=1, max_value=10, value=3)

# Summarize button
if st.button("Summarize"):
    if not input_text.strip():
        st.warning("Please provide text either by pasting or uploading a file!")
    else:
        with st.spinner("Summarizing..."):
            if method == "Extractive":
                summary = summarize_extractive(input_text, n_sentences=num_sentences)
            else:
                summarizer = AbstractiveSummarizer()
                summary = summarizer.summarize(input_text)

        st.subheader("📌 Summary:")
        st.write(summary)

        st.download_button(
            label="⬇️ Download Summary",
            data=summary,
            file_name="summary.txt",
            mime="text/plain"
        )
