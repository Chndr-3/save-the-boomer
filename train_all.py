"""Fit all four classifiers on the same TF-IDF/split and save them together
into models.joblib for the side-by-side frontend. Run: python train_all.py

Mirrors compare.py settings (class_weight balanced). LinearSVC is wrapped in
CalibratedClassifierCV so it exposes predict_proba like the rest.
"""
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from train import FRAUD_THRESHOLD, clean, load

MODELS = {
    "naive_bayes": MultinomialNB(),
    "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "linear_svm": CalibratedClassifierCV(LinearSVC(class_weight="balanced")),
    "random_forest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42
    ),
}


def main():
    texts, labels = load()
    X = [clean(t) for t in texts]
    X_tr, _, y_tr, _ = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    Xtr = vec.fit_transform(X_tr)

    fitted = {}
    for name, clf in MODELS.items():
        clf.fit(Xtr, y_tr)
        fitted[name] = clf
        print(f"  fitted {name}")

    joblib.dump(
        {"vectorizer": vec, "models": fitted, "fraud_threshold": FRAUD_THRESHOLD},
        "models.joblib",
    )
    print("\nsaved models.joblib")


if __name__ == "__main__":
    main()
