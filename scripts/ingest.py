"""Parse a scam-report issue body (env ISSUE_BODY) and append it to
data/sms_fraud.csv, deduped. Exits non-zero with a reason on invalid input,
so the workflow can post it back as a comment. Run by ingest-scam.yml.
"""
import csv
import os
import re
import sys

CSV = "data/sms_fraud.csv"
CATEGORIES = {
    "fake_prize_lottery", "fake_courier", "impersonate_official",
    "fake_family_emergency", "fake_charity", "romance_scam", "other",
}


def parse_issue(body):
    """GitHub issue-form bodies are '### Header\n\nvalue' blocks."""
    fields, header = {}, None
    for line in body.splitlines():
        if line.startswith("### "):
            header = line[4:].strip().lower()
            fields[header] = ""
        elif header is not None:
            fields[header] += line + "\n"
    return {k: v.strip() for k, v in fields.items()}


def main():
    body = os.environ.get("ISSUE_BODY", "")
    f = parse_issue(body)
    text = f.get("scam sms text", "")
    category = f.get("category", "").lower()

    if not text or text == "_No response_":
        sys.exit("No SMS text provided.")
    if category not in CATEGORIES:
        sys.exit(f"Invalid category '{category}'. Allowed: {sorted(CATEGORIES)}")
    text = re.sub(r"\s+", " ", text).strip()  # collapse whitespace/newlines

    with open(CSV, encoding="utf-8") as fh:
        existing = {row["text"].strip() for row in csv.DictReader(fh)}
    if text in existing:
        sys.exit("Duplicate — this message is already in the dataset.")

    with open(CSV, "a", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerow([text, category])
    print(f"Added [{category}]: {text[:80]}")


if __name__ == "__main__":
    main()
