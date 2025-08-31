# NLP Text Summarizer 📝

A versatile Python command-line tool that provides both **extractive** and **abstractive** text summarization.  
The script is designed to be simple and interactive, allowing users to define summary length on the fly.

---

## ✨ Key Features

- **Dual Summarization Methods**

  - **Extractive**: Uses TF-IDF to identify and extract the most important sentences. Lightweight and fast.
  - **Abstractive**: Uses a Hugging Face Transformer model (`facebook/bart-large-cnn`) to generate human-like summaries.

- **Interactive Mode**  
  User-friendly prompts for selecting summary length (extractive mode). No complex arguments needed.

- **Robust & Self-Contained**  
  Single Python script with minimal dependencies, managed via `requirements.txt`.

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/nlp-summarizer.git
cd nlp-summarizer
```
