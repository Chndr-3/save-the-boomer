"""Bake-off: same TF-IDF features + split, four classifiers, ranked by fraud F1.

Fair comparison — features built once, split once (seeded), plain argmax
predictions (no threshold biasing; that's a tuning step for the winner in
train.py). Run: python compare.py
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from train import load, clean

# class_weight='balanced' where supported — fraud is the minority and we want its recall.
MODELS = {
    "naive_bayes": MultinomialNB(),
    "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "linear_svm": LinearSVC(class_weight="balanced"),
    "random_forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
}


def evaluate():
    texts, labels = load()
    X = [clean(t) for t in texts]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    Xtr, Xte = vec.fit_transform(X_tr), vec.transform(X_te)

    rows = []
    for name, clf in MODELS.items():
        clf.fit(Xtr, y_tr)
        pred = clf.predict(Xte)
        pr, rc, f1, _ = precision_recall_fscore_support(
            y_te, pred, labels=["fraud"], zero_division=0
        )
        rows.append({
            "model": name,
            "acc": accuracy_score(y_te, pred),
            "fraud_P": pr[0], "fraud_R": rc[0], "fraud_F1": f1[0],
            "macro_F1": f1_score(y_te, pred, average="macro", zero_division=0),
        })
    # rank by fraud F1, recall as tiebreaker
    rows.sort(key=lambda r: (r["fraud_F1"], r["fraud_R"]), reverse=True)
    return rows


def main():
    rows = evaluate()
    hdr = f"{'model':>14} {'acc':>6} {'fraud_P':>8} {'fraud_R':>8} {'fraud_F1':>9} {'macro_F1':>9}"
    print("\n" + hdr)
    for r in rows:
        print(f"{r['model']:>14} {r['acc']:>6.3f} {r['fraud_P']:>8.3f} "
              f"{r['fraud_R']:>8.3f} {r['fraud_F1']:>9.3f} {r['macro_F1']:>9.3f}")
    print(f"\nwinner (fraud F1): {rows[0]['model']}")


if __name__ == "__main__":
    main()
