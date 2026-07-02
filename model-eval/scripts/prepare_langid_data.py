#!/usr/bin/env python3
"""
prepare_langid_data.py — build a language-ID test CSV.

Classes: english, urdu, roman_urdu, mixed.

Sources actually used (reuse files already downloaded for sentiment, mirfan899/Urdu):
  - roman_urdu : datasets/sentiment/raw/sentiment.csv  (Roman Urdu, `sentence`), sampled
  - urdu       : datasets/sentiment/raw/urdu_v1.tsv     (Urdu script, `Tweet`)

Sources attempted but NOT used (reported honestly):
  - ERUPD (arXiv 2412.17562, English<->Roman Urdu, 75k pairs) -> Kaggle-gated
    ("ERUPD_NMT"), no HuggingFace/GitHub mirror found -> login wall, skipped.
    (It would have supplied the `english` class at scale.)

Because ERUPD is gated, the public combined.csv here covers roman_urdu + urdu only.
The `english` and `mixed` classes live in the balanced gold set
(gold_sets/langid_gold.csv), which is the PRIMARY 4-class confusion-matrix benchmark.

Output: datasets/langid/combined.csv  columns [text, true_language, source]
"""
import csv
import io
import os
import random

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SENT_RAW = os.path.join(HERE, "datasets/sentiment/raw")
OUT = os.path.join(HERE, "datasets/langid/combined.csv")
ROMAN_SAMPLE = 1000    # balance against ~1000 urdu rows

FAILED = [
    ("ERUPD (arXiv 2412.17562, ERUPD_NMT)",
     "Kaggle-gated, no HF/GitHub mirror found -> login wall; would have provided `english`"),
]


def main():
    rows = []
    counts = {}

    roman_path = os.path.join(SENT_RAW, "sentiment.csv")
    if os.path.exists(roman_path):
        df = pd.read_csv(roman_path)
        texts = [str(t).strip() for t in df["sentence"].tolist()
                 if str(t).strip() and str(t).strip().lower() != "nan"]
        random.seed(42)
        random.shuffle(texts)
        for t in texts[:ROMAN_SAMPLE]:
            rows.append({"text": t, "true_language": "roman_urdu", "source": "mirfan899_roman"})
        counts["roman_urdu"] = min(len(texts), ROMAN_SAMPLE)
    else:
        print(f"[MISSING] {roman_path}")

    urdu_path = os.path.join(SENT_RAW, "urdu_v1.tsv")
    if os.path.exists(urdu_path):
        df = pd.read_csv(urdu_path, sep="\t")
        n = 0
        for t in df["Tweet"].tolist():
            t = str(t).strip()
            if t and t.lower() != "nan":
                rows.append({"text": t, "true_language": "urdu", "source": "mirfan899_urdu"})
                n += 1
        counts["urdu"] = n
    else:
        print(f"[MISSING] {urdu_path}")

    random.seed(7)
    random.shuffle(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with io.open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["text", "true_language", "source"])
        for r in rows:
            w.writerow([r["text"], r["true_language"], r["source"]])

    print("=== langid dataset (public) ===")
    for k, v in counts.items():
        print(f"  {k:12s}: {v}")
    print(f"  TOTAL       : {len(rows)} -> {OUT}")
    print("  (english + mixed are gold-set only — see note below)")
    print("\n  Sources attempted but NOT used:")
    for name, why in FAILED:
        print(f"    - {name}: {why}")


if __name__ == "__main__":
    main()
