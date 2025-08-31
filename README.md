# NLP Text Summarizer

A versatile Python command-line tool that provides both **extractive** and **abstractive** text summarization. The script is designed to be simple and interactive, allowing users to define summary length on the fly.

## ✨ Key Features

### Dual Summarization Methods

- **Extractive** → Uses TF-IDF to identify and extract the most important sentences. Lightweight and fast.
- **Abstractive** → Uses a Hugging Face Transformer model (`facebook/bart-large-cnn`) to generate human-like summaries.

### Interactive Mode

User-friendly prompts for selecting summary length (extractive mode). No complex arguments needed.

### Robust & Self-Contained

Single Python script with minimal dependencies, managed via `requirements.txt`.

## 🛠️ Installation

1. **Clone the Repository**

```bash
git clone https://github.com/your-username/nlp-summarizer.git
cd nlp-summarizer
```

2. **Create and Activate a Virtual Environment**

- **Windows:**

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

- **macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. **Download NLTK Data (One-time setup)**
   If you encounter a `LookupError`, run the following in a Python interpreter:

```python
import nltk
nltk.download('punkt_tab')
```

## 🚀 Usage

The script is configured for interactive use.

1. **Prepare Your Input File**
   Ensure a text file (e.g., `article.txt`) is in the project directory. Update the `input_file_path` variable in the script to point to your file.

2. **Run the Script**

```bash
python summarizer_interactive.py
```

3. **Follow the Prompt**
   The script will ask for the number of sentences for the summary.

```
Please enter the number of sentences for the summary: 3
```

4. **View the Output**
   The summary will be printed directly to your console.

## 📜 License

This project is licensed under the MIT License.
