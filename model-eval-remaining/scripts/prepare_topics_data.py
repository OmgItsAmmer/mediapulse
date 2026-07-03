#!/usr/bin/env python3
"""
prepare_topics_data.py — build the topic-modeling text pool (Prompt 5, Task 1).

Pools the `text` column already collected for the three text tasks into a single
corpus for unsupervised topic modeling / clustering:
  - datasets/ner/ner_public.csv        (source_module = ner)
  - datasets/sentiment/combined.csv    (source_module = sentiment)
  - datasets/langid/combined.csv       (source_module = langid)

Dedupe: on a normalized key (strip + lowercase + whitespace-collapsed). The FIRST
occurrence wins, in the order ner -> sentiment -> langid, and keeps its module tag.

Guard: if the deduped pool is under MIN_POOL rows, this prints a clear message and
exits WITHOUT writing a pool — clustering on a tiny corpus is not meaningful, and
we don't fake a larger set (per the prompt).

Output: datasets/topics/pool.csv   columns [text, source_module]

Runs in the eval venv (pandas only):
  venv/bin/python scripts/prepare_topics_data.py
"""
import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "datasets/topics/pool.csv")
MIN_POOL = 500        # below this, stop rather than cluster a too-small corpus

# (csv path, source_module) in priority order for dedup (first occurrence wins).
SOURCES = [
    ("datasets/ner/ner_public.csv", "ner"),
    ("datasets/sentiment/combined.csv", "sentiment"),
    ("datasets/langid/combined.csv", "langid"),
]

_WS = re.compile(r"\s+")


def norm_key(text):
    return _WS.sub(" ", text.strip().lower())


def main():
    rows = []            # (text, source_module) kept, in insertion order
    seen = set()
    per_module = {}
    dupes = 0

    for rel, module in SOURCES:
        path = os.path.join(HERE, rel)
        if not os.path.exists(path):
            print(f"[MISSING] {rel} — skipping (source_module={module})", file=sys.stderr)
            per_module[module] = 0
            continue
        kept = 0
        with open(path, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                text = (r.get("text") or "").strip()
                if not text or text.lower() == "nan":
                    continue
                key = norm_key(text)
                if key in seen:
                    dupes += 1
                    continue
                seen.add(key)
                rows.append((text, module))
                kept += 1
        per_module[module] = kept

    total = len(rows)
    print("=== topic-modeling pool ===")
    for _, module in SOURCES:
        print(f"  {module:10s}: {per_module.get(module, 0)}")
    print(f"  duplicates dropped: {dupes}")
    print(f"  TOTAL (deduped)   : {total}")

    if total < MIN_POOL:
        print(
            f"\nSTOP: pool has {total} rows (< {MIN_POOL}). This is too small for "
            "meaningful topic modeling / clustering.\n"
            "Collect more text (extend the NER/sentiment/langid sets) before running "
            "eval_topics.py. Not writing a pool — refusing to fake a larger corpus.",
            file=sys.stderr,
        )
        sys.exit(1)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["text", "source_module"])
        for text, module in rows:
            w.writerow([text, module])

    print(f"\n  wrote {total} rows -> {os.path.relpath(OUT, HERE)}")
    dom = max(per_module, key=per_module.get) if per_module else None
    if dom and per_module.get(dom, 0) > 0.5 * total:
        print(f"  NOTE: '{dom}' dominates the pool ({per_module[dom]}/{total}). The full "
              "pool is used for the latency benchmark; ARI/NMI is measured on the\n"
              "        balanced gold set (gold_sets/topics_gold.csv), not this pool.")


if __name__ == "__main__":
    main()
