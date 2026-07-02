#!/usr/bin/env python3
"""
prepare_sentiment_data.py — normalize public sentiment corpora into one CSV.

Sources actually used (from the mirfan899/Urdu GitHub repo, sentiment/ folder):
  - roman.csv -> sentiment.csv   Roman Urdu, cols [sentence, sentiment]  (~20k)
  - urdu.tsv  -> urdu_v1.tsv      Urdu script, cols [Tweet, Class=P/N/O]  (~1k)

Sources attempted but NOT used (reported honestly, not fabricated):
  - UCI "Roman Urdu Data Set" id=458 via ucimlrepo -> DatasetNotFoundError
    ("exists but not available for import"). Its data is the same as roman.csv above.
  - HuggingFace `roman_urdu` dataset -> same underlying data as roman.csv; skipped
    (and the HF `datasets` lib is shadowed by this repo's local datasets/ dir).
  - "A Precisely Xtreme..." (arXiv 2003.05443, 3,241 rows) -> no public data
    repo/mirror found for the dataset file. Skipped.

Output: datasets/sentiment/combined.csv  columns [text, label, lang, source]
label in {positive, negative, neutral}. Rows are shuffled (seed=42) so a --limit
sample in the eval is representative (the raw files are sorted by label).
"""
import csv
import io
import os
import random

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "datasets/sentiment/raw")
OUT = os.path.join(HERE, "datasets/sentiment/combined.csv")

LABEL_MAP = {
    "positive": "positive", "pos": "positive", "p": "positive", "1": "positive",
    "negative": "negative", "neg": "negative", "n": "negative", "neative": "negative", "-1": "negative",
    "neutral": "neutral", "neu": "neutral", "o": "neutral", "0": "neutral",
}

SOURCES = [
    {"name": "roman_urdu_mirfan899", "file": "sentiment.csv", "sep": ",",
     "text": "sentence", "label": "sentiment", "lang": "roman_urdu"},
    {"name": "urdu_mirfan899", "file": "urdu_v1.tsv", "sep": "\t",
     "text": "Tweet", "label": "Class", "lang": "urdu"},
]

FAILED_ATTEMPTS = [
    ("UCI Roman Urdu Data Set (ucimlrepo id=458)",
     "DatasetNotFoundError: exists but not available for programmatic import"),
    ("HuggingFace `roman_urdu` dataset",
     "same data as roman.csv; skipped (HF datasets lib shadowed by local datasets/ dir)"),
    ("Precisely Xtreme (arXiv 2003.05443, 3,241 rows)",
     "no public data file/mirror found"),
]


def norm_label(x):
    if x is None:
        return None
    s = str(x).strip().lower()
    return LABEL_MAP.get(s)


def main():
    rows = []
    per_source = {}
    for src in SOURCES:
        path = os.path.join(RAW, src["file"])
        if not os.path.exists(path):
            print(f"[MISSING] {src['name']}: {path}")
            continue
        df = pd.read_csv(path, sep=src["sep"])
        kept = 0
        for _, r in df.iterrows():
            text = str(r.get(src["text"], "")).strip()
            label = norm_label(r.get(src["label"]))
            if not text or text.lower() == "nan" or label is None:
                continue
            rows.append({"text": text, "label": label,
                         "lang": src["lang"], "source": src["name"]})
            kept += 1
        per_source[src["name"]] = kept

    random.seed(42)
    random.shuffle(rows)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with io.open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["text", "label", "lang", "source"])
        for r in rows:
            w.writerow([r["text"], r["label"], r["lang"], r["source"]])

    print("=== sentiment dataset normalization ===")
    for name, n in per_source.items():
        print(f"  {name:24s}: {n:6d} rows")
    print(f"  TOTAL                   : {len(rows):6d} rows -> {OUT}")
    dist = pd.Series([r["label"] for r in rows]).value_counts()
    print("  label distribution     :", dist.to_dict())
    lang = pd.Series([r["lang"] for r in rows]).value_counts()
    print("  lang distribution      :", lang.to_dict())
    print("\n  Sources attempted but NOT used:")
    for name, why in FAILED_ATTEMPTS:
        print(f"    - {name}: {why}")


if __name__ == "__main__":
    main()
