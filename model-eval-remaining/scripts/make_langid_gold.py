#!/usr/bin/env python3
"""
make_langid_gold.py — generate gold_sets/langid_gold.csv from hand-authored rows.

*** PLACEHOLDER DATA — REVIEW AND EXPAND BEFORE USING FOR REAL EVAL ***
40 rows, 10 per class (english, urdu, roman_urdu, mixed), Pakistani social-media style.
The roman_urdu set deliberately includes HARD cases that look like English words
("ap kaisay ho", "theek hai yaar") — the documented failure mode where language-ID
tools tag Roman Urdu as English. Expand to ~200-300 real rows before trusting numbers.

Schema: text, true_language, source   (true_language in english/urdu/roman_urdu/mixed)
"""
import csv
import io
import os

ENGLISH = [
    "The prime minister addressed the nation on live television last night.",
    "I can't believe how expensive groceries have become these days.",
    "Our team played really well and won the championship final.",
    "Please share your feedback about the new mobile application.",
    "The weather in the northern areas is absolutely beautiful this season.",
    "Traffic on the main highway was terrible during rush hour today.",
    "She just got admission into one of the top universities abroad.",
    "The government announced new economic policies earlier this week.",
    "Everyone is talking about the latest cricket match on social media.",
    "Thank you so much for your help, I really appreciate it.",
]
URDU = [
    "وزیراعظم نے قوم سے خطاب کیا اور اہم اعلانات کیے۔",
    "آج موسم بہت خوشگوار ہے اور ہلکی بارش ہو رہی ہے۔",
    "پاکستان کرکٹ ٹیم نے شاندار کارکردگی دکھائی۔",
    "مہنگائی نے عام آدمی کی کمر توڑ دی ہے۔",
    "کراچی میں ٹریفک جام کی وجہ سے لوگ پریشان ہیں۔",
    "نئی حکومتی پالیسیوں پر عوام کا ردعمل ملا جلا ہے۔",
    "بجلی کی لوڈشیڈنگ نے شہریوں کو مشکل میں ڈال دیا۔",
    "اس نے بیرون ملک اعلیٰ تعلیم کے لیے داخلہ لے لیا۔",
    "سوشل میڈیا پر یہ خبر تیزی سے پھیل رہی ہے۔",
    "آپ کی مدد کا بہت شکریہ، میں دل سے ممنون ہوں۔",
]
# Roman Urdu — several look deceptively like English tokens (the failure mode).
ROMAN_URDU = [
    "Ap kaisay ho, sab theek hai na?",
    "Yaar bahut mehngai ho gayi hai aaj kal.",
    "Kal shaam ko milte hain chai peene.",
    "Pakistan ne match jeet liya, kya baat hai.",
    "Mujhe samajh nahi aa raha kya karun.",
    "Theek hai yaar, koi baat nahi.",
    "Bhai zara idhar aana, kaam hai.",
    "Aj school nahi jana, chutti hai.",
    "Wo bahut acha insaan hai, mujhe pasand hai.",
    "Khana kha liya ya nahi abhi tak?",
]
MIXED = [
    "Yaar the weather is bahut acha today, let's go out.",
    "Meeting kal hai at 3pm, please don't be late.",
    "This movie was total bakwaas, waste of paisa.",
    "I'm so tired yaar, poori raat neend nahi aayi.",
    "The government ne petrol price barha di again, unbelievable.",
    "Bro just chill karo, everything will be theek.",
    "She said mujhe nahi jana but phir chali gayi anyway.",
    "Traffic was crazy today, main office late pahuncha.",
    "Let's order food from foodpanda, ghar pe kuch nahi hai.",
    "Exam kal hai and I haven't studied at all, help.",
]


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(here, "gold_sets", "langid_gold.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    data = [("english", ENGLISH), ("urdu", URDU),
            ("roman_urdu", ROMAN_URDU), ("mixed", MIXED)]
    with io.open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["text", "true_language", "source"])
        n = 0
        for lang, rows in data:
            for t in rows:
                w.writerow([t, lang, "placeholder"])
                n += 1
    print(f"Wrote {n} PLACEHOLDER rows -> {out}")
    print("  classes: " + ", ".join(f"{lang}={len(rows)}" for lang, rows in data))
    print("  NOTE: placeholder data — review and expand to ~200-300 rows before real eval.")


if __name__ == "__main__":
    main()
