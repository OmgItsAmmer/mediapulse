#!/usr/bin/env python3
"""
make_sentiment_gold.py — generate gold_sets/sentiment_gold.csv from hand-authored rows.

*** PLACEHOLDER DATA — REVIEW AND EXPAND BEFORE USING FOR REAL EVAL ***
30 plausible Pakistani social-media posts focused on SARCASM (where the literal words
disagree with the intended sentiment), mixing Roman Urdu and code-switched Urdu/English.
`label` is the INTENDED sentiment. `sarcasm` is True when literal != intent.
Expand to ~150-200 real, hand-labeled rows before trusting numbers.

Schema: text, label, sarcasm, lang, source
label in {positive, negative, neutral}; sarcasm in {True, False};
lang in {roman_urdu, code_switched}.
"""
import csv
import io
import os

# (text, intended_label, sarcasm, lang)
EXAMPLES = [
    # --- sarcastic: positive words, negative intent ---
    ("Wah kya government hai, mehngai ne to record tor diya. Shabash!", "negative", True, "roman_urdu"),
    ("Bohat maza aya aaj traffic mein 3 ghante phans kar, zabardast.", "negative", True, "roman_urdu"),
    ("Oh wow, another load shedding in this heat. Thank you so much WAPDA.", "negative", True, "code_switched"),
    ("Kya shaandar cricket team hai humari, har match haar kar dil khush kar deti hai.", "negative", True, "roman_urdu"),
    ("Great, flight 6 ghante late ho gayi. Best day ever yaar.", "negative", True, "code_switched"),
    ("Petrol phir mehnga, kya baat hai, awaam ki to lottery lag gayi.", "negative", True, "roman_urdu"),
    ("Wah ji wah, bijli ka bill dekh kar to dil bagh bagh ho gaya.", "negative", True, "roman_urdu"),
    ("Nice, exam kal hai aur main ne kuch nahi parha. Perfect timing.", "negative", True, "code_switched"),
    ("Bohat imaandar leaders hain hamare, roz naya scandal aata hai. Proud of them.", "negative", True, "code_switched"),
    ("Kya khoobsurat mausam hai, itni garmi mein to AC bhi ro raha hai.", "negative", True, "roman_urdu"),
    ("Shukriya online class, internet bhi sahi se nahi chalta. Maza aa gaya.", "negative", True, "code_switched"),
    ("Zabardast service thi restaurant ki, sirf 2 ghante wait karaya. Wah.", "negative", True, "roman_urdu"),
    ("Our team lost again, wah kya baat hai, ab to aadat si ho gayi hai.", "negative", True, "code_switched"),
    ("Bara maza aya us se baat kar ke, poori raat insult karta raha. Lovely person.", "negative", True, "code_switched"),
    ("Kya zabardast plan tha, sab kuch ulta ho gaya. Perfectly planned.", "negative", True, "code_switched"),
    # --- sarcastic: negative words, positive intent (harder) ---
    ("Yaar tum to bade 'bekaar' ho, itna acha result la kar sabko peeche chor diya!", "positive", True, "roman_urdu"),
    ("Ye banda to poora pagal hai, itni mehnat kaun karta hai aaj kal, respect!", "positive", True, "code_switched"),
    # --- genuine positive (not sarcastic) ---
    ("Alhamdulillah aaj interview bohat acha hua, umeed hai job mil jayegi.", "positive", False, "roman_urdu"),
    ("Bohat khushi hui aap se mil kar, dil se shukriya.", "positive", False, "roman_urdu"),
    ("Just got my visa approved, super excited for the trip!", "positive", False, "code_switched"),
    ("Match jeet gaye hum, kya raat thi, poora mohalla khush.", "positive", False, "roman_urdu"),
    ("Naya mobile acha hai, camera zabardast, kaafi khush hoon.", "positive", False, "roman_urdu"),
    # --- genuine negative (not sarcastic) ---
    ("Bohat dukh hua ye khabar sun kar, Allah marhoom ko jannat de.", "negative", False, "roman_urdu"),
    ("Mera phone chori ho gaya bazaar mein, bohat pareshan hoon.", "negative", False, "roman_urdu"),
    ("This is really disappointing, service bilkul acha nahi tha.", "negative", False, "code_switched"),
    ("Kya bakwas movie thi, paise waste ho gaye, bilkul mat dekhna.", "negative", False, "roman_urdu"),
    # --- neutral (not sarcastic) ---
    ("Kal meeting 4 baje hai office mein, sab time pe aa jayen.", "neutral", False, "roman_urdu"),
    ("Aaj mausam thora cloudy hai, shayad baarish ho.", "neutral", False, "roman_urdu"),
    ("Event next Monday ko hai main hall mein, details baad mein.", "neutral", False, "code_switched"),
    ("Petrol pump band tha to CNG dalwa li aaj.", "neutral", False, "roman_urdu"),
]


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(here, "gold_sets", "sentiment_gold.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with io.open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["text", "label", "sarcasm", "lang", "source"])
        for text, label, sarc, lang in EXAMPLES:
            w.writerow([text, label, sarc, lang, "placeholder"])

    n_sarc = sum(1 for _, _, s, _ in EXAMPLES if s)
    from collections import Counter
    print(f"Wrote {len(EXAMPLES)} PLACEHOLDER rows -> {out}")
    print(f"  sarcasm=True : {n_sarc}  | sarcasm=False : {len(EXAMPLES)-n_sarc}")
    print(f"  labels       : {dict(Counter(l for _, l, _, _ in EXAMPLES))}")
    print(f"  langs        : {dict(Counter(l for _, _, _, l in EXAMPLES))}")
    print("  NOTE: placeholder data — review labels and expand to ~150-200 rows before real eval.")


if __name__ == "__main__":
    main()
