import re
import numpy as np
import spacy
from textblob import TextBlob

# Load spaCy English model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    import spacy.cli
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

def title_len(title):
    """Length of the title (number of characters)."""
    return len(title) if isinstance(title, str) else 0

def num_exclamations(text):
    """Number of exclamation marks in the text."""
    return text.count('!') if isinstance(text, str) else 0

def num_uppercase_words(text):
    """Number of fully uppercase words in the text."""
    if not isinstance(text, str):
        return 0
    return sum(1 for word in text.split() if word.isupper() and len(word) > 1)

def named_entity_count(text):
    """Count the number of named entities in the text using spaCy."""
    if not isinstance(text, str):
        return 0
    doc = nlp(text)
    return len(doc.ents)

def sentiment_subjectivity(text):
    """Return sentiment polarity and subjectivity using TextBlob."""
    if not isinstance(text, str):
        return 0.0, 0.0
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

def count_capitals(text):
    """Count the number of capital letters in the text."""
    return sum(1 for c in text if c.isupper()) if isinstance(text, str) else 0

def text_length(text):
    """Return the length of the text."""
    return len(text) if isinstance(text, str) else 0

def extract_features(title, text, ner_count=None, use_ner=False):
    """Extract all custom features and return as a numpy array."""
    pol, subj = sentiment_subjectivity(text)
    features = [
        title_len(title),
        num_exclamations(text),
        num_uppercase_words(text),
    ]
    if use_ner:
        if ner_count is None:
            ner_count = named_entity_count(text)
        features.append(ner_count)
    features += [
        pol,
        subj,
        count_capitals(text),
        text_length(text)
    ]
    return np.array(features) 