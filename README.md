# 🛡️ Save the Boomer

**An open-source SMS fraud detector for Indonesia — built to help protect the people most likely to fall for scam texts.**

Elderly and less tech-savvy people in Indonesia get hit hard by fraudulent SMS: fake prize notifications, "your package is held" scams, bank impersonation, and urgent money-transfer requests. Save the Boomer is a community project to detect those messages automatically, and to build the open dataset that makes such detection possible in Bahasa Indonesia.

> **Status: Phase 1 — dataset + model only.** No app, no API, no fancy features yet. Right now this repo is about one thing: assembling good training data and a solid classifier. Everything else comes later.

## What this project does

It classifies an Indonesian SMS message into one of three buckets:

| Label | Meaning | Example vibe |
|-------|---------|-------------|
| `ham` | Normal, legitimate message | "Bu, aku pulang telat ya" |
| `promotion` | Marketing / promo spam | "Nikmati diskon 50% pulsa hari ini!" |
| `fraud` | Scam / fraud attempt | "Selamat! No Anda menang 50jt, transfer biaya admin ke..." |

The goal is to catch **fraud** with high recall — missing a real scam is far more costly than mislabeling a harmless promo.

## Why it matters

Public Indonesian SMS datasets that are labeled *specifically for fraud* barely exist — most available data is generic promotional spam. So part of this project's mission is to **build and openly publish a fraud-labeled Indonesian SMS dataset** as a contribution the whole community can use.

## Tech

- **Python 3.x**
- **scikit-learn** — Multinomial Naive Bayes classifier + TF-IDF features
- **pandas** — dataset handling
- **Sastrawi** — Indonesian stopword removal / stemming

The preprocessing pipeline handles Bahasa Indonesia text: lowercasing, tokenization (with an eye on scam-text slang like *sgra*, *byr*, *kirim*), stopword removal, and TF-IDF vectorization.

## Data sources

- **Base dataset:** [`Andikazidanef15/Sentiment-Analysis-on-Indonesian-SMS-Dataset`](https://github.com/Andikazidanef15/Sentiment-Analysis-on-Indonesian-SMS-Dataset) (~1,145 labeled SMS, originally from [`kmkurn/id-nlp-resource`](https://github.com/kmkurn/id-nlp-resource)). Note: it skews toward promotional spam.
- **Our contribution:** a hand-collected, growing set of real Indonesian **fraud** SMS examples, with a documented labeling schema (`ham` / `promotion` / `fraud`).

## Roadmap (Phase 1)

- [ ] Audit the base dataset's label quality and distribution
- [ ] Isolate the fraud-labeled subset from `id-nlp-resource` if possible
- [ ] Hand-collect 100–200 real fraud SMS examples for v1
- [ ] Build the Bahasa Indonesia preprocessing pipeline (Sastrawi)
- [ ] Train and evaluate a baseline Naive Bayes model (accuracy, precision, recall, F1, confusion matrix)
- [ ] Publish dataset + serialized model

See [`project.md`](./project.md) for the full technical spec.

## Contributing

**We especially need real scam SMS examples.** If you've received fraudulent texts in Bahasa Indonesia, contributions to the dataset are hugely valuable. Labeling guidelines will live alongside the dataset. Issues and PRs welcome.

## License

MIT — see [`LICENSE`](./LICENSE).
