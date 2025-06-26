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

def prepare_features(df, use_ner=False):
    vectorizer = TfidfVectorizer(max_features=1000)
    X_tfidf = vectorizer.fit_transform(df['text_clean'])
    titles = df['title'].tolist() if 'title' in df.columns else [''] * len(df)
    texts = df['text'].tolist()
    if use_ner:
        import features
        nlp = features.nlp
        ner_counts = [len(doc.ents) for doc in nlp.pipe(texts, batch_size=32, disable=['tagger', 'parser', 'lemmatizer'])]
    else:
        ner_counts = [None] * len(texts)
    X_custom = np.vstack([
        extract_features(title, text, ner_count=ner_counts[i], use_ner=use_ner)
        for i, (title, text) in enumerate(zip(titles, texts))
    ])
    from scipy.sparse import hstack
    X = hstack([X_tfidf, X_custom])
    return X, vectorizer

def evaluate_classifiers(cv=5, save_results_path=None):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if save_results_path is None:
        save_results_path = os.path.join(base_dir, 'models', 'model_comparison.csv')
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

def train_best_model(best_model_name='RandomForestClassifier', save_path=None, vectorizer_path=None, use_ner=False):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if save_path is None:
        save_path = os.path.join(base_dir, 'models', 'best_model.pkl')
    if vectorizer_path is None:
        vectorizer_path = os.path.join(base_dir, 'models', 'vectorizer.pkl')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = load_data()
    df['text_clean'] = df['text'].apply(clean_text)
    X, vectorizer = prepare_features(df, use_ner=use_ner)
    y = df['label']
    classifiers = get_classifiers()
    model = classifiers[best_model_name]
    model.fit(X, y)
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    return model, vectorizer

def load_model(model_path=None, vectorizer_path=None):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if model_path is None:
        model_path = os.path.join(base_dir, 'models', 'best_model.pkl')
    if vectorizer_path is None:
        vectorizer_path = os.path.join(base_dir, 'models', 'vectorizer.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

if __name__ == "__main__":
    print("Training the best model (default: RandomForestClassifier) without NER features...")
    train_best_model(use_ner=False)
    print("Model training complete. Model saved to models/best_model.pkl") 