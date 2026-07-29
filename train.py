"""Train a Multinomial Naive Bayes SMS classifier (ham / promotion / fraud).

Data: base Indonesian SMS dataset (label 0=ham, 1=fraud, 2=promotion) unioned
with any extra fraud rows in data/sms_fraud.csv. Run: python train.py
"""
import csv
import re

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# ponytail: Sastrawi stopwords if installed, else skip — TF-IDF copes without.
try:
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    STOPWORDS = set(StopWordRemoverFactory().get_stop_words())
except ImportError:
    STOPWORDS = set()

LABELS = {"0": "ham", "1": "fraud", "2": "promotion"}


def clean(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # urls
    text = re.sub(r"\d+", " ", text)                        # numbers/phones
    tokens = re.findall(r"[a-z]+", text)
    return " ".join(t for t in tokens if t not in STOPWORDS and len(t) > 1)


def load():
    texts, labels = [], []
    seen = set()
    with open("data/base_sms.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = row["Teks"].strip()
            texts.append(t); labels.append(LABELS[row["label"]]); seen.add(t)
    # union extra fraud examples not already in the base set
    extra = 0
    with open("data/sms_fraud.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = row["text"].strip()
            if t and t not in seen:
                texts.append(t); labels.append("fraud"); seen.add(t); extra += 1
    print(f"loaded {len(texts)} messages (+{extra} extra fraud from sms_fraud.csv)")
    return texts, labels


def main():
    texts, labels = load()
    X = [clean(t) for t in texts]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    clf = MultinomialNB()
    clf.fit(vec.fit_transform(X_tr), y_tr)

    pred = clf.predict(vec.transform(X_te))
    order = ["ham", "promotion", "fraud"]
    print("\n" + classification_report(y_te, pred, labels=order, digits=3, zero_division=0))
    print("confusion matrix (rows=true, cols=pred):", order)
    print(confusion_matrix(y_te, pred, labels=order))

    joblib.dump({"vectorizer": vec, "model": clf}, "model.joblib")
    print("\nsaved model.joblib")


if __name__ == "__main__":
    main()
