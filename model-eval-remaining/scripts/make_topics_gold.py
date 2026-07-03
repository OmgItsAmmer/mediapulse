#!/usr/bin/env python3
"""
make_topics_gold.py — generate gold_sets/topics_gold.csv (Prompt 5, Task 2).

*** PLACEHOLDER DATA — REVIEW / EXPAND / CORRECT BEFORE TRUSTING ANY NUMBERS ***

Topic modeling has no single "correct" label the way NER/sentiment do, so instead
of ground truth we hand-author 40 posts grouped into 6 realistic Pakistani
news/political "story" clusters, each tagged with a manually assigned `story_id`.
Within each story the posts deliberately MIX english / roman_urdu / urdu /
code-switched text, the way real coverage of one story looks on social media.

This is the evaluation set for ARI / NMI in eval_topics.py: a good clustering
should recover these 6 stories (posts of the same story_id land in the same
predicted topic). The stories are generic templates — replace them with real
scraped posts from actual events, and expand to ~15-25 posts per story, before
reporting real metrics.

Schema: text, story_id, story_label, lang, source
  story_id    : 1..6 (manually assigned cluster id)
  story_label : human-readable description of the story (not used by metrics)
  lang        : english | roman_urdu | urdu | mixed  (rough tag of that post)
"""
import csv
import io
import os

# Each entry: (story_id, story_label, [(text, lang), ...])
STORIES = [
    (1, "Cricket — Pakistan win a big international match", [
        ("Pakistan ne aaj shaandaar match jeet liya, kya batting thi Babar ki!", "roman_urdu"),
        ("What a win for Pakistan! Bowlers absolutely dominated in the final overs.", "english"),
        ("پاکستان کرکٹ ٹیم نے میچ جیت کر پوری قوم کا دل خوش کر دیا۔", "urdu"),
        ("Green shirts ne aaj proper khela, last over thriller tha yaar.", "roman_urdu"),
        ("Congratulations team Pakistan, that chase under pressure was unbelievable.", "english"),
        ("Kya match tha! Karachi mein log sarkon pe nikal aaye celebrate karne.", "mixed"),
        ("شاباش گرین شرٹس، فائنل اوور میں جو کھیل دکھایا وہ یادگار ہے۔", "urdu"),
    ]),
    (2, "Political rally / jalsa in a major city", [
        ("Aaj ka jalsa historic tha, awaam ka hujoom dekh kar sab hairan.", "roman_urdu"),
        ("Huge crowd at the rally today, supporters came from across the province.", "english"),
        ("لاہور میں سیاسی جلسے میں ہزاروں کارکنوں نے شرکت کی۔", "urdu"),
        ("Leader ne stage se bara announcement kiya, crowd went wild.", "mixed"),
        ("Rally mein itni bheer thi ke roads block ho gayin poora sheher.", "roman_urdu"),
        ("The opposition's public gathering drew massive attendance this evening.", "english"),
        ("کارکنوں کے جوش و خروش نے جلسہ گاہ کا ماحول گرما دیا۔", "urdu"),
    ]),
    (3, "Economic policy — fuel price hike / IMF & the rupee", [
        ("Petrol phir se mehnga ho gaya, aam aadmi ki kamar toot gayi.", "roman_urdu"),
        ("Government announced another fuel price increase effective midnight.", "english"),
        ("آئی ایم ایف سے معاہدے کے بعد مہنگائی میں مزید اضافے کا خدشہ ہے۔", "urdu"),
        ("Rupee again gir gaya against the dollar, imports aur mehnge ho jayenge.", "mixed"),
        ("Budget ke baad bijli aur gas ke bills ne logon ko pareshan kar diya.", "roman_urdu"),
        ("New IMF conditions push inflation higher, economists warn of tough months.", "english"),
        ("پیٹرول کی قیمت میں اضافے پر عوام کی جانب سے شدید ردعمل سامنے آیا۔", "urdu"),
    ]),
    (4, "Monsoon flooding in Sindh / Punjab", [
        ("Barish ne tabahi macha di, kai ilaqon mein paani ghuss gaya.", "roman_urdu"),
        ("Heavy monsoon rains flooded several districts, thousands displaced.", "english"),
        ("سندھ کے نچلے علاقوں میں سیلابی پانی سے فصلیں تباہ ہو گئیں۔", "urdu"),
        ("Flood ki wajah se roads band, rescue teams active hain har jagah.", "mixed"),
        ("Log apne gharon se nikalne pe majboor, relief camps full ho gaye.", "roman_urdu"),
        ("Authorities issued flood warnings as river levels keep rising overnight.", "english"),
    ]),
    (5, "Viral social-media controversy over a public remark", [
        ("Uska bayaan viral ho gaya, Twitter pe log bura reaction de rahe hain.", "roman_urdu"),
        ("That statement sparked huge backlash online, trending all day.", "english"),
        ("سوشل میڈیا پر اس متنازع بیان پر بحث چھڑ گئی۔", "urdu"),
        ("Clip social media pe fire ki tarah phail gayi, sab discuss kar rahe.", "mixed"),
        ("Log demand kar rahe hain ke maafi maangi jaye is remark pe.", "roman_urdu"),
        ("The controversial comment divided opinion across every timeline today.", "english"),
        ("ٹرینڈ ٹاپ پر آ گیا اور صارفین نے سخت تنقید کی۔", "urdu"),
    ]),
    (6, "New drama / film release trending", [
        ("Naya drama release ho gaya, first episode ne dil jeet liya.", "roman_urdu"),
        ("The new film opened to packed cinemas over the weekend.", "english"),
        ("نئی فلم کی کہانی اور موسیقی نے شائقین کو متاثر کیا۔", "urdu"),
        ("Drama ka soundtrack already viral, cast ne acting kamaal ki hai.", "mixed"),
        ("Sab log is serial ki baat kar rahe hain, must watch yaar.", "roman_urdu"),
        ("Critics praised the lead performances in this weekend's big release.", "english"),
    ]),
]


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(here, "gold_sets", "topics_gold.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    n = 0
    with io.open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["text", "story_id", "story_label", "lang", "source"])
        for sid, label, posts in STORIES:
            for text, lang in posts:
                w.writerow([text, sid, label, lang, "placeholder"])
                n += 1

    print(f"Wrote {n} PLACEHOLDER rows across {len(STORIES)} stories -> {out}")
    print("  cluster sizes: " + ", ".join(f"story{sid}={len(posts)}" for sid, _, posts in STORIES))
    print("  NOTE: placeholder stories — replace with real scraped posts and expand to")
    print("        ~15-25 posts/story before trusting ARI/NMI numbers.")


if __name__ == "__main__":
    main()
