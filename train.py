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


# Flag fraud if P(fraud) >= threshold; otherwise pick the best of the rest.
# Lower threshold = catch more fraud (higher recall), at the cost of precision.
FRAUD_THRESHOLD = 0.30


def predict_biased(clf, Xv, threshold):
    proba = clf.predict_proba(Xv)
    fraud_i = list(clf.classes_).index("fraud")
    out = []
    for p in proba:
        if p[fraud_i] >= threshold:
            out.append("fraud")
        else:
            best = max(range(len(p)), key=lambda i: p[i] if i != fraud_i else -1)
            out.append(clf.classes_[best])
    return out


def top_words(clf, vec, n=12):
    """Most indicative tokens per class, straight from the trained NB."""
    feats = vec.get_feature_names_out()
    for ci, cls in enumerate(clf.classes_):
        logp = clf.feature_log_prob_[ci]
        top = sorted(range(len(feats)), key=lambda i: logp[i], reverse=True)[:n]
        print(f"  {cls:>9}: " + ", ".join(feats[i] for i in top))


def main():
    texts, labels = load()
    X = [clean(t) for t in texts]
    idx = list(range(len(X)))

    tr, te, y_tr, y_te = train_test_split(
        idx, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_tr = [X[i] for i in tr]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    clf = MultinomialNB()
    clf.fit(vec.fit_transform(X_tr), y_tr)
    Xte = vec.transform([X[i] for i in te])
    order = ["ham", "promotion", "fraud"]

    # sweep to show the trade-off (0.5 == plain argmax baseline)
    from sklearn.metrics import precision_recall_fscore_support
    print("\nfraud class vs threshold:")
    print(f"{'thresh':>7} {'precision':>10} {'recall':>8} {'f1':>7}")
    for th in [0.50, 0.40, 0.30, 0.20, 0.15, 0.10]:
        p = predict_biased(clf, Xte, th)
        pr, rc, f1, _ = precision_recall_fscore_support(
            y_te, p, labels=["fraud"], zero_division=0
        )
        print(f"{th:>7.2f} {pr[0]:>10.3f} {rc[0]:>8.3f} {f1[0]:>7.3f}")

    pred = predict_biased(clf, Xte, FRAUD_THRESHOLD)
    print(f"\n=== chosen threshold {FRAUD_THRESHOLD} ===")
    print(classification_report(y_te, pred, labels=order, digits=3, zero_division=0))
    print("confusion matrix (rows=true, cols=pred):", order)
    print(confusion_matrix(y_te, pred, labels=order))

    print("\n=== most indicative words per class ===")
    top_words(clf, vec)

    print("\n=== misclassified messages ===")
    wrong = [(y_te[j], pred[j], texts[te[j]]) for j in range(len(te)) if y_te[j] != pred[j]]
    print(f"{len(wrong)}/{len(te)} wrong")
    for true, got, msg in wrong:
        print(f"  [true={true:>9} pred={got:>9}] {msg[:90]}")

    joblib.dump(
        {"vectorizer": vec, "model": clf, "fraud_threshold": FRAUD_THRESHOLD},
        "model.joblib",
    )
    print("\nsaved model.joblib")


if __name__ == "__main__":
    main()
