📝 NLP Text Summarizer
A versatile Python command-line tool that provides both extractive and abstractive text summarization. The script is designed to be simple and interactive, allowing users to define summary length on the fly.

✨ Key Features
Dual Summarization Methods

Extractive → Uses TF-IDF to identify and extract the most important sentences. Lightweight and fast.

Abstractive → Uses a Hugging Face Transformer model (facebook/bart-large-cnn) to generate human-like summaries.

Interactive Mode

User-friendly prompts for selecting summary length (extractive mode). No complex arguments needed.

Robust & Self-Contained

Single Python script with minimal dependencies, managed via requirements.txt.

🛠️ Installation
Clone the Repository
git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
cd your-repository-name

Create and Activate a Virtual Environment
Windows:

python -m venv .venv
.\.venv\Scripts\activate

macOS / Linux:

python3 -m venv .venv
source .venv/bin/activate

Install Dependencies
pip install -r requirements.txt

Download NLTK Data (One-time setup)
If you encounter a LookupError, run the following in a Python interpreter:

# Run this in a Python interpreter if needed

import nltk
nltk.download('punkt_tab')

🚀 Usage
The script is configured to be interactive.

Prepare Your Input File
Ensure a text file (e.g., article.txt) is in the project directory.

Update the input_file_path variable in the script to point to your file.

Run the Script
python summarizer_interactive.py

Follow the Prompt
The script will ask for the number of sentences for the summary.

Please enter the number of sentences for the summary: 3

View the Output
The summary will be printed directly to your console.

📜 License
This project is licensed under the MIT License.
