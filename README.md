# Fake News Detection Project

This project implements an advanced machine learning pipeline to detect fake news articles using Natural Language Processing (NLP) and modern explainability tools. It features a modular backend and a rich Streamlit web interface for single, batch, and real-time news validation.

## Dataset

- Download the dataset from Kaggle: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- Place `Fake.csv` and `Real.csv` in the `data/` directory at the project root.

## Features

- **Modular codebase**: Clean separation for data loading, text cleaning, feature engineering, model training, and prediction.
- **Multiple classifiers**: Logistic Regression, MultinomialNB, Random Forest, Linear SVC, XGBoost (if installed).
- **Advanced feature engineering**: Includes TF-IDF, title length, exclamation/uppercase counts, named entity count (spaCy), sentiment polarity/subjectivity (TextBlob), and more.
- **Batch prediction**: Upload `.csv` or `.txt` files for bulk analysis.
- **Real-time news validation**: Fetch and classify live headlines using NewsAPI.
- **Explainability**: ELI5 for top keywords, ready for SHAP/LIME integration.
- **Analytics dashboard**: Confidence distribution, class balance, word clouds, top fake keywords, and topic modeling (NMF).
- **Model evaluation**: Cross-validation, comparison charts, and confusion matrix.

## Project Structure

```
.
├── data/                # Place Fake.csv and Real.csv here (not tracked by git)
├── models/              # Trained models, vectorizers, and evaluation charts
├── src/                 # All source code
│   ├── app.py           # Streamlit web app
│   ├── data_loader.py   # Data loading utilities
│   ├── text_cleaner.py  # Text preprocessing
│   ├── features.py      # Feature engineering
│   ├── model.py         # Model training, evaluation, and comparison
│   └── predict.py       # Prediction logic
├── requirements.txt     # Python dependencies
└── README.md
```

## Setup

1. Clone the repository
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install streamlit wordcloud matplotlib seaborn spacy textblob eli5 scikit-learn
   python -m spacy download en_core_web_sm
   ```

## Data Preparation

1. Download the dataset from Kaggle: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
2. Place `Fake.csv` and `Real.csv` in the `data` directory at the project root.

## Training the Model

1. From the `src` directory, run:
   ```bash
   python model.py
   ```
   - The model and vectorizer will be saved in the `../models/` directory.
   - You can change the model by editing the `train_best_model` call in `model.py`.

## Running the Web Application

1. From the `src` directory, run:
   ```bash
   streamlit run app.py
   ```
2. Open your browser to the URL shown in the terminal (typically `http://localhost:8501`).

## Web App Features

- **Single Article Prediction**: Enter a title and article text for instant prediction and explanation.
- **Batch Prediction**: Upload `.csv` (with `title` and `text` columns) or `.txt` (one article per line) for batch analysis, confidence scores, word clouds, and sentiment plots.
- **Real-Time News**: Fetch and classify live headlines (requires NewsAPI key in `.streamlit/secrets.toml`).
- **Explainability**: See top keywords influencing each prediction (ELI5).
- **Analytics Dashboard**: Visualize confidence, class balance, fake keywords, and topic modeling.
- **Model Evaluation**: View model comparison and confusion matrix charts.

## Requirements

- Python 3.8+
- See `requirements.txt` for core dependencies
- Additional: `streamlit`, `wordcloud`, `matplotlib`, `seaborn`, `spacy`, `textblob`, `eli5`, `scikit-learn`

## License

This project is licensed under the MIT License - see the LICENSE file for details. 