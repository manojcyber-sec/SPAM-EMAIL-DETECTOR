# preprocess_and_train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'emails.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'nb_model.joblib')
VEC_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'tfidf_vectorizer.joblib')
def train():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=['label', 'text'])
    X = df['text'].astype(str)
    y = df['label'].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    vec = TfidfVectorizer(max_features=2000, stop_words='english')
    X_train_tfidf = vec.fit_transform(X_train)
    X_test_tfidf = vec.transform(X_test)
    clf = MultinomialNB()
    clf.fit(X_train_tfidf, y_train)
    y_pred = clf.predict(X_test_tfidf)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(vec, VEC_PATH)
    print("Saved model and vectorizer.")
if __name__ == "__main__":
    train()
