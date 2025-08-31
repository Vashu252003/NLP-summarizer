## NLP Text Summarizer 📝

A versatile Python command-line tool that provides both extractive and abstractive text summarization. This script is designed to be easy to use, handling various text inputs and allowing for user-defined summary lengths through an interactive prompt.

Key Features
Dual Summarization Methods:

Extractive: Uses TF-IDF to identify and extract the most important sentences directly from the source text. Lightweight and fast.

Abstractive: Leverages a pre-trained Hugging Face Transformer model (facebook/bart-large-cnn) to generate new, human-like summaries.

Interactive Mode: No need for complex command-line arguments. The script interactively prompts the user for the desired summary length.

Flexible Input: The original code supports input from files, direct text arguments, or piped stdin (though the current interactive version is configured for file input).

Robust and Self-Contained: The project is a single, easy-to-run script with minimal external dependencies, all managed via requirements.txt.

Installation
Follow these steps to set up the project environment.

1. Clone the Repository

git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
cd your-repository-name

2. Create and Activate a Virtual Environment (Recommended)

Windows:

python -m venv .venv
.\.venv\Scripts\activate

macOS / Linux:

python3 -m venv .venv
source .venv/bin/activate

3. Install Dependencies
   Install all the required Python libraries using the requirements.txt file.

pip install -r req.txt

4. Download NLTK Data (One-time setup)
   The first time you run the script, nltk may need to download the punkt and punkt_tab data packages for sentence tokenization. The script handles the punkt download, but you may need to fetch punkt_tab manually if you encounter an error.

# Run this in a Python interpreter if needed

import nltk
nltk.download('punkt_tab')

Usage
The script is configured to be interactive.

1. Prepare Your Input File

Make sure you have a text file in the project directory. The script is currently hard-coded to look for article.txt.

You can change the target file by editing this line in the script: input_file_path = 'article.txt'.

2. Run the Script
   Execute the script from your terminal:

python summarizer_interactive.py

3. Follow the Prompt
   The script will ask you to enter the number of sentences you want in your summary.

Successfully loaded 'article.txt'.
Please enter the number of sentences for the summary: 3

4. View the Output
   The generated summary will be printed directly to your console.

Example Output
--- SUMMARY ---

Rising greenhouse gas emissions, primarily from human activities such as burning fossil fuels and deforestation, are leading to unprecedented shifts in global temperatures.
Governments and international organizations have begun to address the crisis through agreements such as the Paris Climate Accord, which aims to limit global warming to well below 2 degrees Celsius above pre-industrial levels.
By rethinking energy systems, protecting forests, and fostering innovation, humanity still has the opportunity to mitigate the worst effects of climate change and secure a more sustainable future for generations to come.

License
This project is licensed under the MIT License - see the LICENSE.md file for details.
