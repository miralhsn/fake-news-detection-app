import numpy as np
from model import load_model
from text_cleaner import clean_text
from features import extract_features
from scipy.sparse import hstack

def predict(title, text, model=None, vectorizer=None):
    if model is None or vectorizer is None:
        model, vectorizer = load_model()
    cleaned = clean_text(text)
    X_tfidf = vectorizer.transform([cleaned])
    X_custom = extract_features(title, text).reshape(1, -1)
    X = hstack([X_tfidf, X_custom])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]
    return 'REAL' if pred == 1 else 'FAKE', prob 