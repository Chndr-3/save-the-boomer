"""Retrain Lambda: weekly EventBridge trigger. Reads dataset from S3,
rebuilds the NB model, uploads model.joblib back to S3.

ponytail: the fit below (TF-IDF 1-2gram min_df=2 + MultinomialNB) must stay
identical to train.py::main — if you tune one, tune both, or the weekly model
diverges from what you validated locally.
"""
import csv
import os

import boto3
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from train import clean  # shared preprocessing

s3 = boto3.client("s3")
BUCKET = os.environ["MODEL_BUCKET"]
FRAUD_THRESHOLD = float(os.environ.get("FRAUD_THRESHOLD", "0.30"))
LABELS = {"0": "ham", "1": "fraud", "2": "promotion"}


def _load():
    texts, labels, seen = [], [], set()
    base = "/tmp/base_sms.csv"
    s3.download_file(BUCKET, "data/base_sms.csv", base)
    with open(base, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = row["Teks"].strip()
            texts.append(t); labels.append(LABELS[row["label"]]); seen.add(t)
    try:
        extra = "/tmp/sms_fraud.csv"
        s3.download_file(BUCKET, "data/sms_fraud.csv", extra)
        with open(extra, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                t = row["text"].strip()
                if t and t not in seen:
                    texts.append(t); labels.append("fraud"); seen.add(t)
    except s3.exceptions.ClientError:
        pass  # extra fraud file is optional
    return texts, labels


def lambda_handler(event, context):
    texts, labels = _load()
    X = [clean(t) for t in texts]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    clf = MultinomialNB()
    clf.fit(vec.fit_transform(X), labels)

    out = "/tmp/model.joblib"
    joblib.dump(
        {"vectorizer": vec, "model": clf, "fraud_threshold": FRAUD_THRESHOLD}, out
    )
    s3.upload_file(out, BUCKET, "model.joblib")
    return {"trained_on": len(texts), "model": f"s3://{BUCKET}/model.joblib"}
