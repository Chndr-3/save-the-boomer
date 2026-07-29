"""Classify an SMS. Usage: python predict.py "your message here" """
import sys

import joblib

from train import FRAUD_THRESHOLD, clean


def classify(text, bundle):
    vec, clf = bundle["vectorizer"], bundle["model"]
    th = bundle.get("fraud_threshold", FRAUD_THRESHOLD)
    proba = clf.predict_proba(vec.transform([clean(text)]))[0]
    prob = dict(zip(clf.classes_, proba))
    label = "fraud" if prob["fraud"] >= th else max(
        (c for c in clf.classes_ if c != "fraud"), key=lambda c: prob[c]
    )
    return label, prob


def main():
    if len(sys.argv) < 2:
        sys.exit('usage: python predict.py "message text"')
    bundle = joblib.load("model.joblib")
    label, prob = classify(" ".join(sys.argv[1:]), bundle)
    print(f"\n  {label.upper()}")
    for c in ("fraud", "promotion", "ham"):
        print(f"    {c:>9}: {prob[c]:.1%}")


if __name__ == "__main__":
    main()
