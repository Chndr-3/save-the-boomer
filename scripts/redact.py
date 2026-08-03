"""Redact Indonesian phone numbers from SMS text -> [PHONE].

Used by ingest.py (scrub incoming reports) and runnable directly to clean the
existing CSVs in place: `python scripts/redact.py`.
"""
import csv
import re
import sys

# Indonesian mobile: +62 / 62 / 0 prefix, then 8, then 8-12 more digits,
# tolerating single space/dash separators (0821-5104-2542, 082-33333-783).
# ponytail: won't touch short codes (*123*66#) or dates (08-2016) — too few digits.
PHONE = re.compile(r"(?:\+?62|0)8\d(?:[\s\-]?\d){7,11}")

# Account / virtual-account / ref numbers: any run of 8+ digits (with optional
# single separators). Threshold 8 keeps Rupiah prices (35000, 100000) as signal.
ACCT = re.compile(r"\d(?:[\s\-.]?\d){7,}")


def redact(text):
    text = PHONE.sub("[PHONE]", text)  # phones first — they'd also match ACCT
    return ACCT.sub("[ACCT]", text)


def _clean_csv(path):
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    text_col = 0  # both CSVs keep the message in the first column
    n = 0
    for row in body:
        new = redact(row[text_col])
        if new != row[text_col]:
            n += 1
            row[text_col] = new
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(body)
    print(f"{path}: redacted {n} rows")


def _selfcheck():
    assert redact("Hub ATY 082134976289(wa)") == "Hub ATY [PHONE](wa)"
    assert redact("call +6287820008358 now") == "call [PHONE] now"
    assert redact("byr 0821-5104-2542 ok") == "byr [PHONE] ok"
    assert redact("aktifkan di *123*66# ya") == "aktifkan di *123*66# ya"
    assert redact("promo 08-2016 diskon") == "promo 08-2016 diskon"
    assert redact("transfer ke rek 5500248418 BCA") == "transfer ke rek [ACCT] BCA"
    assert redact("diskon 50000 rupiah") == "diskon 50000 rupiah"  # price kept


if __name__ == "__main__":
    _selfcheck()
    for p in sys.argv[1:] or ["data/base_sms.csv", "data/sms_fraud.csv"]:
        _clean_csv(p)
