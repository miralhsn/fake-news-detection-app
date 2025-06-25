import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from data_loader import load_data
from text_cleaner import clean_text
from features import extract_features, named_entity_count
import pandas as pd
import spacy

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False

def get_classifiers():
    classifiers = {
        'LogisticRegression': LogisticRegression(max_iter=5000, random_state=42),
        'MultinomialNB': MultinomialNB(),
        'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=42),
        'LinearSVC': LinearSVC(random_state=42, max_iter=5000)
    }
    if xgb_available:
        classifiers['XGBoostClassifier'] = XGBClassifier(n_estimators=20, max_depth=3, use_label_encoder=False, eval_metric='logloss', random_state=42)
    return classifiers

def prepare_features(df):
    vectorizer = TfidfVectorizer(max_features=1000)
    X_tfidf = vectorizer.fit_transform(df['text_clean'])
    # Use spaCy's nlp.pipe for fast NER
    nlp = spacy.load('en_core_web_sm')  # use the same nlp object as in features.py
    texts = df['text'].tolist()
    titles = df['title'].tolist() if 'title' in df.columns else [''] * len(df)
    # Get named entity counts in batch
    ner_counts = [len(doc.ents) for doc in nlp.pipe(texts, batch_size=32, disable=['tagger', 'parser', 'lemmatizer'])]
    # Now build features row by row, but use precomputed ner_counts
    X_custom = np.vstack([
        extract_features(title, text, ner_count=ner_counts[i])
        for i, (title, text) in enumerate(zip(titles, texts))
    ])
    from scipy.sparse import hstack
    X = hstack([X_tfidf, X_custom])
    return X, vectorizer

def evaluate_classifiers(cv=5, save_results_path='models/model_comparison.csv'):
    df = load_data()
    df['text_clean'] = df['text'].apply(clean_text)
    X, vectorizer = prepare_features(df)
    y = df['label']
    classifiers = get_classifiers()
    results = []
    for name, clf in classifiers.items():
        print(f"Evaluating {name}...")
        acc = cross_val_score(clf, X, y, cv=cv, scoring='accuracy').mean()
        f1 = cross_val_score(clf, X, y, cv=cv, scoring='f1').mean()
        recall = cross_val_score(clf, X, y, cv=cv, scoring='recall').mean()
        precision = cross_val_score(clf, X, y, cv=cv, scoring='precision').mean()
        results.append({
            'Classifier': name,
            'Accuracy': acc,
            'F1': f1,
            'Recall': recall,
            'Precision': precision
        })
    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv(save_results_path, index=False)
    return results_df

def train_best_model(best_model_name='RandomForestClassifier', save_path='models/best_model.pkl', vectorizer_path='models/vectorizer.pkl'):
    df = load_data()
    df['text_clean'] = df['text'].apply(clean_text)
    X, vectorizer = prepare_features(df)
    y = df['label']
    classifiers = get_classifiers()
    model = classifiers[best_model_name]
    model.fit(X, y)
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    return model, vectorizer

def load_model(model_path='models/best_model.pkl', vectorizer_path='models/vectorizer.pkl'):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

if __name__ == "__main__":
    print("Training the best model (default: RandomForestClassifier)...")
    train_best_model()
    print("Model training complete. Model saved to models/best_model.pkl") 