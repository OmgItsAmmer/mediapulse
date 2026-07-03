#!/usr/bin/env python3
"""
prepare_summarization_data.py — build the summarization input clusters (Prompt 6, Task 1).

Reuses the topic clusters from gold_sets/topics_gold.csv: posts are grouped by
`story_id` and concatenated into ONE text blob per cluster. This keeps the pipeline
consistent end to end (4.3 topics -> 4.5 summaries) — each summary input is exactly
one story's worth of posts.

Guard: if fewer than MIN_CLUSTERS usable clusters exist, this prints a clear message
and exits WITHOUT writing — we don't summarize a single post as if it were a cluster.

Output: datasets/summarization/clusters.csv
  columns [story_id, story_label, n_posts, text]   (text = posts joined by newline)

Runs in the eval venv (stdlib only):
  venv/bin/python scripts/prepare_summarization_data.py
"""
import csv
import os
import sys
from collections import OrderedDict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOPICS_GOLD = os.path.join(HERE, "gold_sets/topics_gold.csv")
OUT = os.path.join(HERE, "datasets/summarization/clusters.csv")
MIN_CLUSTERS = 5


def main():
    if not os.path.exists(TOPICS_GOLD):
        sys.exit(f"ERROR: {TOPICS_GOLD} missing. Run scripts/make_topics_gold.py first.")

    clusters = OrderedDict()      # story_id -> {"label":..., "posts":[...]}
    with open(TOPICS_GOLD, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            text = (r.get("text") or "").strip()
            sid = (r.get("story_id") or "").strip()
            if not text or not sid:
                continue
            c = clusters.setdefault(sid, {"label": r.get("story_label", ""), "posts": []})
            c["posts"].append(text)

    # keep only clusters with >= 2 posts (a single post is not a "cluster")
    usable = [(sid, c) for sid, c in clusters.items() if len(c["posts"]) >= 2]

    print("=== summarization clusters (from topics_gold.csv) ===")
    for sid, c in clusters.items():
        tag = "" if len(c["posts"]) >= 2 else "   (dropped: <2 posts)"
        print(f"  story {sid}: {len(c['posts'])} posts — {c['label'][:48]}{tag}")

    if len(usable) < MIN_CLUSTERS:
        print(
            f"\nSTOP: only {len(usable)} usable clusters (>= 2 posts each); need "
            f"{MIN_CLUSTERS}.\nExpand gold_sets/topics_gold.csv before running "
            "summarization eval — not summarizing single posts as clusters.",
            file=sys.stderr,
        )
        sys.exit(1)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    try:
        sort_key = sorted(usable, key=lambda kv: int(kv[0]))
    except ValueError:
        sort_key = sorted(usable, key=lambda kv: kv[0])
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["story_id", "story_label", "n_posts", "text"])
        for sid, c in sort_key:
            w.writerow([sid, c["label"], len(c["posts"]), "\n".join(c["posts"])])

    print(f"\n  wrote {len(usable)} clusters -> {os.path.relpath(OUT, HERE)}")


if __name__ == "__main__":
    main()
