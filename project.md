# Project Spec: Save the Boomer — SMS Fraud Classifier (Phase 1)

## 1. Project Name
**Save the Boomer** — open source elder-fraud SMS protection system for Indonesia.

## 2. Phase 1 Scope
This phase is deliberately narrow: build and validate the **dataset and detection model only**. Nothing else.

**In scope:**
- Sourcing / assembling an Indonesian-language SMS dataset (spam, fraud, normal/ham)
- Preprocessing pipeline for Bahasa Indonesia text
- Naive Bayes classifier: train, evaluate, tune

**Explicitly out of scope for this phase (no work, no design, no placeholders):**
- Any client — no Android app, no UI, no mobile spec of any kind
- API / endpoint of any kind
- Call screening
- On-device inference
- Family notification/sync features
- WhatsApp integration
- Hosting/deployment decisions

This phase produces: a labeled dataset + a trained, evaluated classifier (as a script/notebook + serialized model file). Nothing further.

## 3. Dataset
### Primary source
- `Andikazidanef15/Sentiment-Analysis-on-Indonesian-SMS-Dataset` (GitHub) — ~1,145 labeled Indonesian SMS entries (`dataset_sms_spam_v1.csv`), originally sourced from `kmkurn/id-nlp-resource`
- Note: this dataset skews toward **promotional spam** (telco offers, subscriptions), not fraud specifically

### Known limitation
- The `kmkurn/id-nlp-resource` README references a corpus of 1,143 sentences labeled as normal/fraud/promotion — worth digging into directly if the fraud-labeled subset can be isolated
- Public Indonesian **fraud-specific** SMS datasets are scarce; most available data is general spam

### Plan to close the gap
- Use available dataset as the v1 training base (ham vs. promotional spam vs. fraud, if labels allow)
- **Build our own fraud-labeled dataset** — this becomes an open-source contribution in itself:
  - Hand-collect real scam SMS examples (community sourcing, forums, personal reports)
  - Document a clear labeling schema: `ham`, `promotion`, `fraud`
  - Publish the growing dataset alongside the model in the same open source repo
  - Later: add a "report as scam" feedback loop once the client exists, to keep growing the dataset from real users

## 4. Classifier
- **Algorithm:** Multinomial Naive Bayes (standard choice for text/word-frequency classification)
- **Preprocessing pipeline:**
  - Lowercasing
  - Tokenization (Bahasa Indonesia — watch for slang/abbreviations common in scam texts, e.g. "sgra", "byr", "kirim")
  - Stopword removal (Indonesian stopword list)
  - Feature extraction: Bag-of-Words or TF-IDF vectorization
- **Training/eval:**
  - Train/test split (e.g. 80/20)
  - Metrics: accuracy, precision, recall, F1 — precision/recall matter more than raw accuracy here since false negatives (missed scams) are more costly than false positives
  - Confusion matrix to see where fraud vs. promotion vs. ham get confused
- **Output:** serialized model (e.g. pickle/joblib) + the fitted vectorizer

## 5. Tooling / Stack
- Python 3.x
- `scikit-learn` (Naive Bayes, TF-IDF vectorizer, train/test split, metrics)
- `pandas` for dataset handling
- Indonesian NLP support: check `Sastrawi` (Indonesian stemmer/stopword library) for preprocessing

## 6. Open Source Plan
- Public repo from day one, MIT license
- Repo contains: dataset (with labeling schema documented), preprocessing scripts, training script, evaluation notebook/report
- Clear README explaining the fraud-vs-promotion-vs-ham labeling approach and inviting community contributions to the dataset

## 7. Immediate Next Steps
- [ ] Pull down `Andikazidanef15` dataset and audit label quality/distribution
- [ ] Investigate `kmkurn/id-nlp-resource` for the fraud-labeled subset specifically
- [ ] Start hand-collecting real Indonesian fraud SMS examples (target: 100–200 for v1)
- [ ] Set up preprocessing pipeline with Sastrawi or equivalent
- [ ] Train baseline Naive Bayes model, evaluate
- [ ] Push initial repo structure to GitHub (public, MIT license)
