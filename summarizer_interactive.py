
from typing import List
import argparse
import sys
import math
import os

# Abstractive summarization (transformers)
try:
    from transformers import pipeline
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# Extractive summarization (TF-IDF)
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import sent_tokenize


def summarize_extractive(text: str, n_sentences: int = 3) -> str:
    """Simple extractive summarizer using TF-IDF scoring of sentences.

    - Tokenizes text into sentences
    - Calculates TF-IDF across sentences
    - Scores each sentence by summing TF-IDF values for terms in the sentence
    - Returns top-n sentences in original order
    """
    if not text or not text.strip():
        return ""

    # Ensure the sentence tokenizer is available
    try:
        nltk.data.find('tokenizers/punkt')
    except Exception:
        nltk.download('punkt')

    # Split into sentences
    sentences = sent_tokenize(text)
    if len(sentences) <= n_sentences:
        return "\n".join(sentences)

    # Vectorize sentences with TF-IDF (treat each sentence as a document)
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Score sentences by summing tf-idf values
    import numpy as np
    scores = tfidf_matrix.sum(axis=1).A1  # convert to 1D numpy array

    # Pick top N sentence indices
    top_n_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_sentences]
    top_n_idx.sort()  # preserve original order

    selected = [sentences[i] for i in top_n_idx]
    return "\n".join(selected)


class AbstractiveSummarizer:
    """A small wrapper around HF's summarization pipeline.

    If transformers isn't available, attempting to instantiate will raise an error.
    """
    def __init__(self, model_name: str = 'facebook/bart-large-cnn', device: int = -1):
        if not HF_AVAILABLE:
            raise RuntimeError('transformers library not available. Install `transformers` and `torch`.')
        # device = -1 -> CPU, >=0 -> GPU
        self.pipeline = pipeline('summarization', model=model_name, device=device)

    def summarize(self, text: str, max_length: int = 150, min_length: int = 40, do_sample: bool = False) -> str:
        if not text or not text.strip():
            return ""

        # Rough chunking by characters (not tokens) to avoid heavy tokenization here.
        if len(text) <= 3000:
            res = self.pipeline(text, max_length=max_length, min_length=min_length, do_sample=do_sample)
            return res[0]['summary_text'].strip()
        else:
            # Chunking approach: split into paragraphs and build chunks under approx size
            paras = [p for p in text.split('\n\n') if p.strip()]
            chunks = []
            curr = ''
            for p in paras:
                if len(curr) + len(p) < 3000:
                    curr += '\n\n' + p
                else:
                    chunks.append(curr.strip())
                    curr = p
            if curr.strip():
                chunks.append(curr.strip())

            partial_summaries = []
            for chunk in chunks:
                out = self.pipeline(chunk, max_length=max_length, min_length=min_length, do_sample=do_sample)
                partial_summaries.append(out[0]['summary_text'].strip())

            # Summarize concatenated partial summaries to form final summary
            concat = '\n'.join(partial_summaries)
            out = self.pipeline(concat, max_length=max_length, min_length=min_length, do_sample=do_sample)
            return out[0]['summary_text'].strip()


def safe_read_stdin() -> str:
    """Safely read from stdin.

    Many sandboxed environments (including some CI systems and restricted shells)
    do not provide a readable stdin, or calling sys.stdin.read() raises OSError.

    This helper handles those cases and returns an empty string if stdin cannot be
    read. It also checks sys.stdin.isatty() and will not attempt to read when the
    process has no piped input.
    """
    # If stdin is not available or is a TTY (interactive), return empty string
    try:
        if sys.stdin is None:
            return ""
        # sys.stdin may not have isatty in some environments; guard with getattr
        is_tty = getattr(sys.stdin, 'isatty', lambda: True)()
        if is_tty:
            return ""
        # Try to read; catch OSError specifically
        try:
            data = sys.stdin.read()
            return data if data is not None else ""
        except OSError as e:
            print(f"Warning: unable to read from stdin: {e}", file=sys.stderr)
            return ""
    except Exception as e:
        # Extremely defensive: if anything unexpected happens, log and return empty
        print(f"Warning: unexpected error while checking stdin: {e}", file=sys.stderr)
        return ""

def main():
    # --- 1. SET YOUR CONFIGURATION HERE ---
    # Put the name of the file you want to summarize
    input_file_path = 'article.txt'
    
    # Choose your method: 'extractive' or 'abstractive'
    summarization_method = 'extractive'
    
    # Set an output file, or set it to None to print to the console
    output_file_path = None 
    # ------------------------------------


    # --- 2. THE SCRIPT RUNS FROM HERE ---
    text = ""
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"Successfully loaded '{input_file_path}'.")
    except FileNotFoundError:
        print(f"Error: The file '{input_file_path}' was not found.", file=sys.stderr)
        sys.exit(1)

    # === NEW: INTERACTIVE PROMPT FOR SENTENCE COUNT ===
    num_sentences = 0
    if summarization_method == 'extractive':
        while True:
            try:
                # Ask the user for input
                user_input = input("Please enter the number of sentences for the summary: ")
                num_sentences = int(user_input)
                if num_sentences > 0:
                    break  # Exit the loop if the input is a valid positive number
                else:
                    print("Please enter a number greater than 0.")
            except ValueError:
                print("That's not a valid number. Please try again.")
    # =================================================

    # Ensure nltk punkt is available before tokenization
    try:
        nltk.data.find('tokenizers/punkt')
    except Exception:
        nltk.download('punkt')

    # Generate the summary
    summary = ""
    if summarization_method == 'extractive':
        summary = summarize_extractive(text, n_sentences=num_sentences)
    elif summarization_method == 'abstractive':
        # Abstractive method doesn't use sentence count, so we skip the prompt for it
        if not HF_AVAILABLE:
            print('Error: transformers not installed for abstractive method.', file=sys.stderr)
            sys.exit(2)
        summarizer = AbstractiveSummarizer() 
        summary = summarizer.summarize(text)
    else:
        print(f"Error: Unknown method '{summarization_method}'.", file=sys.stderr)
        sys.exit(1)

    # Save to file or print to console
    if output_file_path:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"Summary successfully saved to: {output_file_path}")
    else:
        print('\n--- SUMMARY ---\n')
        print(summary)
        


# ------------------
# Unit tests
# ------------------
import unittest

class ExtractiveTests(unittest.TestCase):
    def setUp(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except Exception:
            nltk.download('punkt')

    def test_empty_text(self):
        self.assertEqual(summarize_extractive('', n_sentences=3), '')

    def test_short_text_returns_all_sentences(self):
        text = 'Hello world. This is a test.'
        expected = 'Hello world.\nThis is a test.'
        self.assertEqual(summarize_extractive(text, n_sentences=3), expected)

    def test_selects_top_sentences(self):
        text = 'Cats are cute. Dogs are loyal. Fish swim. Birds can fly.'
        out = summarize_extractive(text, n_sentences=2)
        # Should return exactly 2 sentences separated by newline
        lines = [l for l in out.splitlines() if l.strip()]
        self.assertEqual(len(lines), 2)


class AbstractiveTests(unittest.TestCase):
    @unittest.skipUnless(HF_AVAILABLE, 'transformers not installed')
    def test_abstractive_basic(self):
        # This test only runs when transformers is available locally.
        summarizer = AbstractiveSummarizer(model_name='t5-small', device=-1)
        text = 'This is a test. ' * 20
        out = summarizer.summarize(text, max_length=30, min_length=5)
        self.assertIsInstance(out, str)
        self.assertTrue(len(out) > 0)


if __name__ == '__main__':
    main()