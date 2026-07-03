#!/usr/bin/env python3
"""
make_summarization_gold.py — generate gold_sets/summarization_gold.csv (Prompt 6, Task 2).

*** PLACEHOLDER REFERENCE SUMMARIES — REVIEW / REWRITE BEFORE TRUSTING NUMBERS ***

One human-quality reference summary per topic cluster (2-3 sentences, English,
capturing the core story even though the source posts are Roman Urdu / Urdu /
code-switched). story_id matches gold_sets/topics_gold.csv (and therefore
datasets/summarization/clusters.csv). These are MY drafts, not ground truth — the
whole point of a reference is that a human writes/vets it, so replace these with
your own wording (and re-generate the clusters) before reporting ROUGE/BERTScore.

Schema: story_id, story_label, reference_summary, source
"""
import csv
import io
import os

# story_id -> (story_label, reference_summary). Must line up with make_topics_gold.py.
REFERENCES = {
    1: ("Cricket — Pakistan win a big international match",
        "Pakistan won a high-pressure international cricket match, with strong batting "
        "and a decisive bowling performance in the closing overs. Fans across the "
        "country, including in Karachi, celebrated the victory."),
    2: ("Political rally / jalsa in a major city",
        "A large political rally drew thousands of supporters in a major city, with "
        "crowds so big that roads were blocked. A party leader used the stage to make "
        "a significant announcement to an energised audience."),
    3: ("Economic policy — fuel price hike / IMF & the rupee",
        "The government raised fuel prices again and the rupee weakened against the "
        "dollar, stoking fears of higher inflation following a new IMF agreement. The "
        "public reacted angrily as electricity and gas bills also climbed."),
    4: ("Monsoon flooding in Sindh / Punjab",
        "Heavy monsoon rains flooded several districts in Sindh and Punjab, damaging "
        "crops and displacing thousands of people. Rescue teams were deployed and "
        "authorities issued flood warnings as river levels kept rising."),
    5: ("Viral social-media controversy over a public remark",
        "A controversial public remark went viral and sparked heavy backlash across "
        "social media, trending for much of the day. Opinion was sharply divided, with "
        "many users demanding an apology."),
    6: ("New drama / film release trending",
        "A newly released drama and film drew strong attention, filling cinemas and "
        "earning praise for the cast's performances and the soundtrack. The release "
        "trended widely, with viewers recommending it to others."),
}


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(here, "gold_sets", "summarization_gold.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    with io.open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["story_id", "story_label", "reference_summary", "source"])
        for sid in sorted(REFERENCES):
            label, summ = REFERENCES[sid]
            w.writerow([sid, label, summ, "placeholder"])

    print(f"Wrote {len(REFERENCES)} PLACEHOLDER reference summaries -> {out}")
    print("  NOTE: these are drafts — review/rewrite them (and re-run the clusters)")
    print("        before trusting ROUGE-L / BERTScore numbers.")


if __name__ == "__main__":
    main()
