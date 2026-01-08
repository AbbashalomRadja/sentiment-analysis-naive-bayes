import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.strip()

class SentimentModel:
    def __init__(self, model_path="models/nb_model.pkl", vec_path="models/tfidf.pkl"):
        self.model_path = model_path
        self.vec_path = vec_path
        self.model = None
        self.vectorizer = None

    def train_from_csv(self, file_path):
        df = pd.read_csv(file_path)
        df['text'] = df['text'].apply(clean_text)
        X = df['text']
        y = df['label']

        self.vectorizer = TfidfVectorizer()
        X_vec = self.vectorizer.fit_transform(X)

        self.model = MultinomialNB()
        self.model.fit(X_vec, y)

        joblib.dump(self.model, self.model_path)
        joblib.dump(self.vectorizer, self.vec_path)

        return len(df)  # jumlah data train
