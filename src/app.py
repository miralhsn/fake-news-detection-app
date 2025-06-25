import streamlit as st
import pickle
import os
from text_cleaner import clean_text
import eli5
from predict import predict
from model import load_model
import numpy as np
import requests
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

NEWSAPI_KEY = st.secrets["NEWSAPI_KEY"] if "NEWSAPI_KEY" in st.secrets else "YOUR_NEWSAPI_KEY"
NEWSAPI_URL = "https://newsapi.org/v2/top-headlines?country=us&pageSize=5&apiKey=" + NEWSAPI_KEY

def load_model_and_vectorizer():
    """Load the trained model and vectorizer."""
    try:
        with open('models/model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model files not found. Please train the model first by running train_model.py")
        return None, None

def predict_news(text, model, vectorizer):
    """
    Predict if the news is fake or real.
    
    Args:
        text (str): Input news text
        model: Trained classifier
        vectorizer: Fitted TF-IDF vectorizer
    
    Returns:
        str: Prediction result ('REAL' or 'FAKE')
    """
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Vectorize the text
    text_vectorized = vectorizer.transform([cleaned_text])
    
    # Make prediction
    prediction = model.predict(text_vectorized)[0]
    probability = model.predict_proba(text_vectorized)[0]
    
    return 'REAL' if prediction == 1 else 'FAKE', probability

def fetch_live_headlines():
    try:
        response = requests.get(NEWSAPI_URL)
        data = response.json()
        if data.get("status") == "ok":
            return data.get("articles", [])
        else:
            return []
    except Exception as e:
        st.warning(f"Could not fetch live news: {e}")
        return []

def main():
    st.set_page_config(
        page_title="Fake News Detector",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Fake News Detector")
    st.write("""
    This application uses machine learning to detect whether a news article is likely to be fake or real.
    Enter the text of a news article below to get started!
    """)
    
    # Load model and vectorizer
    model, vectorizer = load_model()
    
    if model is None or vectorizer is None:
        st.stop()
    
    # Text input
    news_title = st.text_input(
        "Enter news title here:",
        placeholder="Paste the news article title here..."
    )
    news_text = st.text_area(
        "Enter news text here:",
        height=200,
        placeholder="Paste the news article text here..."
    )
    
    if st.button("Analyze"):
        if not news_text.strip():
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing..."):
                result, probability = predict(news_title, news_text, model, vectorizer)
                
                # Display result with probability
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"### Prediction: {result}")
                
                with col2:
                    prob_real = probability[1] if result == 'REAL' else probability[0]
                    st.markdown(f"### Confidence: {prob_real:.2%}")
                
                # Add a visual indicator
                color = "green" if result == "REAL" else "red"
                st.markdown(f"""
                    <div style="
                        padding: 20px;
                        border-radius: 10px;
                        background-color: {color};
                        color: white;
                        text-align: center;
                        margin: 10px 0;">
                        This news article appears to be {result}
                    </div>
                """, unsafe_allow_html=True)
                
                # Explainability with eli5
                st.markdown("💬 Top keywords influencing this result:")
                try:
                    # Prepare input for eli5
                    cleaned = news_text.lower()
                    vectorized_input = vectorizer.transform([cleaned])
                    explanation = eli5.format_as_text(eli5.explain_prediction(model, vectorized_input))
                    st.code(explanation)
                except Exception as e:
                    st.warning(f"Could not generate explanation: {e}")
    
    # Add information about the model
    with st.expander("ℹ️ About the Model"):
        st.write("""
        This fake news detector uses:
        - TF-IDF (Term Frequency-Inverse Document Frequency) for text vectorization
        - Logistic Regression, Naive Bayes, Random Forest, Linear SVC, and XGBoost for classification
        - NLTK, spaCy, and TextBlob for text preprocessing and feature engineering
        - ELI5 for model explainability
        The model has been trained on a dataset of labeled real and fake news articles.
        """)

    st.header("🌍 Real-Time News Validation")
    if NEWSAPI_KEY == "YOUR_NEWSAPI_KEY":
        st.info("Set your NewsAPI key in Streamlit secrets to enable live news.")
    else:
        articles = fetch_live_headlines()
        for i, article in enumerate(articles):
            headline = article.get("title", "")
            content = article.get("content", "") or article.get("description", "") or ""
            st.markdown(f"**{i+1}. {headline}**")
            if st.button(f"Classify Headline {i+1}"):
                with st.spinner("Classifying..."):
                    result, probability = predict(headline, content, model, vectorizer)
                    st.write(f"Prediction: **{result}** | Confidence: **{max(probability):.2%}**")
                    if st.button(f"Show Explanation {i+1}"):
                        try:
                            cleaned = content.lower()
                            vectorized_input = vectorizer.transform([cleaned])
                            explanation = eli5.format_as_text(eli5.explain_prediction(model, vectorized_input))
                            st.code(explanation)
                        except Exception as e:
                            st.warning(f"Could not generate explanation: {e}")

    st.header("🎛️ Batch Prediction & Analysis")
    uploaded_file = st.file_uploader("Upload a .txt or .csv file for batch prediction", type=["txt", "csv"])
    batch_df = None
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            batch_df = pd.read_csv(uploaded_file)
        else:
            # Assume .txt: one article per line
            lines = uploaded_file.read().decode('utf-8').splitlines()
            batch_df = pd.DataFrame({'text': lines, 'title': ['']*len(lines)})
        st.write(f"Loaded {len(batch_df)} articles.")
        results = []
        for idx, row in batch_df.iterrows():
            title = row['title'] if 'title' in row else ''
            text = row['text']
            result, prob = predict(title, text, model, vectorizer)
            pol, subj = 0.0, 0.0
            try:
                from features import sentiment_subjectivity
                pol, subj = sentiment_subjectivity(text)
            except:
                pass
            results.append({
                'Title': title,
                'Text': text,
                'Prediction': result,
                'Confidence': f"{max(prob)*100:.1f}%",
                'Polarity': pol,
                'Subjectivity': subj
            })
        results_df = pd.DataFrame(results)
        st.dataframe(results_df)
        # Word cloud for fake/real
        for label in ['FAKE', 'REAL']:
            texts = results_df[results_df['Prediction'] == label]['Text'].str.cat(sep=' ')
            if texts.strip():
                st.subheader(f"Word Cloud for {label} News")
                wc = WordCloud(width=600, height=300, background_color='white').generate(texts)
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
        # Sentiment chart
        st.subheader("Sentiment Polarity & Subjectivity Distribution")
        if not results_df.empty:
            fig, ax = plt.subplots()
            ax.scatter(results_df['Polarity'], results_df['Subjectivity'],
                       c=(results_df['Prediction'] == 'FAKE').astype(int), cmap='coolwarm', alpha=0.6)
            ax.set_xlabel('Polarity')
            ax.set_ylabel('Subjectivity')
            st.pyplot(fig)
        # --- Analytics Dashboard ---
        st.header("📊 Analytics Dashboard")
        # Confidence distribution
        st.subheader("Confidence Distribution")
        conf_vals = results_df['Confidence'].str.rstrip('%').astype(float)
        fig, ax = plt.subplots()
        sns.histplot(conf_vals, bins=10, kde=True, ax=ax, color='skyblue')
        ax.set_xlabel('Confidence (%)')
        ax.set_ylabel('Count')
        st.pyplot(fig)
        # Class balance pie chart
        st.subheader("Class Balance")
        class_counts = results_df['Prediction'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
        ax.axis('equal')
        st.pyplot(fig)
        # Top 10 fake keywords
        st.subheader("Top 10 Fake News Keywords")
        fake_texts = results_df[results_df['Prediction'] == 'FAKE']['Text'].str.cat(sep=' ')
        if fake_texts.strip():
            tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
            X_fake = tfidf.fit_transform([fake_texts])
            feature_array = np.array(tfidf.get_feature_names_out())
            tfidf_sorting = np.argsort(X_fake.toarray()).flatten()[::-1]
            top_n = feature_array[tfidf_sorting][:10]
            st.write(', '.join(top_n))
        # Topic modeling (NMF)
        st.subheader("Topic Modeling (NMF)")
        all_texts = results_df['Text'].dropna().tolist()
        if len(all_texts) > 5:
            tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
            X = tfidf.fit_transform(all_texts)
            n_topics = min(5, X.shape[0] // 2)
            nmf = NMF(n_components=n_topics, random_state=42)
            W = nmf.fit_transform(X)
            H = nmf.components_
            feature_names = tfidf.get_feature_names_out()
            for topic_idx, topic in enumerate(H):
                st.markdown(f"**Topic {topic_idx+1}:** " + ', '.join([feature_names[i] for i in topic.argsort()[:-6:-1]]))
    # Model evaluation charts
    st.header("📊 Model Evaluation Charts")
    if os.path.exists('models/model_comparison.png'):
        st.image('models/model_comparison.png', caption='Model Comparison')
    if os.path.exists('models/confusion_matrix.png'):
        st.image('models/confusion_matrix.png', caption='Confusion Matrix')

if __name__ == "__main__":
    main() 