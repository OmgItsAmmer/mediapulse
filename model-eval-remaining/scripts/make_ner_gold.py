#!/usr/bin/env python3
"""
make_ner_gold.py — generate gold_sets/ner_gold.csv from hand-authored examples.

*** PLACEHOLDER DATA — REVIEW AND EXPAND BEFORE USING FOR REAL EVAL ***
These 30 rows are plausible Pakistani political / social-media style sentences in
Roman Urdu and code-switched Urdu/English, written by the assistant as a TEMPLATE.
You must review/correct the entity labels and expand to ~150 real hand-labeled
rows before trusting the eval numbers.

To add rows: append (text, lang, [(entity_surface, TYPE), ...]) tuples below (list
entities in the order they appear in the sentence) and re-run this script. Offsets
are computed automatically, so you never hand-count characters.

Schema written: text, entities(JSON list of {text,type,start,end}), lang, source
lang in {roman_urdu, code_switched}. Types: PERSON, ORG, LOCATION, BRAND, DATE, MISC.
"""
import csv
import io
import json
import os
import re
import sys

# (text, lang, [(entity_surface_text, TYPE), ...])  entities in surface order
EXAMPLES = [
    ("Imran Khan ne aaj Lahore mein PTI ka jalsa kiya, awaam ka hujoom dekh kar dushman pareshan.",
     "roman_urdu", [("Imran Khan", "PERSON"), ("aaj", "DATE"), ("Lahore", "LOCATION"), ("PTI", "ORG")]),
    ("Bilawal Bhutto aur PPP ke workers Karachi mein kal shaam rally karenge.",
     "roman_urdu", [("Bilawal Bhutto", "PERSON"), ("PPP", "ORG"), ("Karachi", "LOCATION"), ("kal shaam", "DATE")]),
    ("Nawaz Sharif ki taqreer PMLN jalse mein sun kar log bohat khush hue Islamabad ke andar.",
     "roman_urdu", [("Nawaz Sharif", "PERSON"), ("PMLN", "ORG"), ("Islamabad", "LOCATION")]),
    ("Maryam Nawaz ne kaha ke Punjab ki awaam hamare saath hai, GT Road par aao.",
     "roman_urdu", [("Maryam Nawaz", "PERSON"), ("Punjab", "LOCATION"), ("GT Road", "LOCATION")]),
    ("Just attended the PTI power show at Minar-e-Pakistan, bhai crowd was insane! Imran Khan zindabad.",
     "code_switched", [("PTI", "ORG"), ("Minar-e-Pakistan", "LOCATION"), ("Imran Khan", "PERSON")]),
    ("Shehbaz Sharif ne petrol price barha di, awaam ka bura haal hai is mehngai mein.",
     "roman_urdu", [("Shehbaz Sharif", "PERSON")]),
    ("Breaking: Supreme Court ne aaj important faisla sunaya, CJP ke remarks Twitter par trend kar rahe hain.",
     "code_switched", [("Supreme Court", "ORG"), ("aaj", "DATE"), ("CJP", "MISC"), ("Twitter", "BRAND")]),
    ("Babar Azam ne shaandar century banayi, Pakistan ne India ko Dubai mein haraya.",
     "roman_urdu", [("Babar Azam", "PERSON"), ("Pakistan", "LOCATION"), ("India", "LOCATION"), ("Dubai", "LOCATION")]),
    ("PSL final dekhne Gaddafi Stadium ja raha hoon, Lahore Qalandars jeetega inshallah.",
     "roman_urdu", [("PSL", "ORG"), ("Gaddafi Stadium", "LOCATION"), ("Lahore Qalandars", "ORG")]),
    ("Yaar this new Samsung phone is amazing but mehnga bohat hai, Daraz pe order karunga.",
     "code_switched", [("Samsung", "BRAND"), ("Daraz", "BRAND")]),
    ("Aaj Faizabad par dharna hai, TLP ke supporters road block kar rahe hain.",
     "roman_urdu", [("Aaj", "DATE"), ("Faizabad", "LOCATION"), ("TLP", "ORG")]),
    ("Molana Fazlur Rehman ne JUI-F ka azadi march Islamabad ki taraf shuru kiya.",
     "roman_urdu", [("Molana Fazlur Rehman", "PERSON"), ("JUI-F", "ORG"), ("Islamabad", "LOCATION")]),
    ("The IMF deal finally hogaya, Ishaq Dar announced it in a presser today.",
     "code_switched", [("IMF", "ORG"), ("Ishaq Dar", "PERSON"), ("today", "DATE")]),
    ("Karachi mein barish ke baad K-Electric ne load shedding barha di, awaam ghusse mein.",
     "roman_urdu", [("Karachi", "LOCATION"), ("K-Electric", "ORG")]),
    ("Atif Aslam ka naya gaana Coke Studio pe release hua hai, sab dekh rahe hain.",
     "roman_urdu", [("Atif Aslam", "PERSON"), ("Coke Studio", "BRAND")]),
    ("Bhai Careem aur Uber dono Lahore mein protest ki wajah se band ho gaye.",
     "code_switched", [("Careem", "BRAND"), ("Uber", "BRAND"), ("Lahore", "LOCATION")]),
    ("Asif Ali Zardari ne kaha ke sab siyasi jamaatein Sindh ke liye mil kar kaam karein.",
     "roman_urdu", [("Asif Ali Zardari", "PERSON"), ("Sindh", "LOCATION")]),
    ("Aleem Khan ne IPP join kar li, Jahangir Tareen ke saath press conference ki.",
     "roman_urdu", [("Aleem Khan", "PERSON"), ("IPP", "ORG"), ("Jahangir Tareen", "PERSON")]),
    ("Elections on 8 February confirmed, ECP ne notification jari kar diya finally.",
     "code_switched", [("8 February", "DATE"), ("ECP", "ORG")]),
    ("Peshawar mein dhamaka hua, rescue teams mauqe par pohanch gayi hain abhi.",
     "roman_urdu", [("Peshawar", "LOCATION")]),
    ("Quetta se khabar aayi ke according, Balochistan Assembly ka ijlas kal hoga.",
     "code_switched", [("Quetta", "LOCATION"), ("Balochistan Assembly", "ORG"), ("kal", "DATE")]),
    ("Fawad Chaudhry ne tweet kiya ke media par pabandi na lagayi jaye, PEMRA sun raha hai?",
     "roman_urdu", [("Fawad Chaudhry", "PERSON"), ("PEMRA", "ORG")]),
    ("Guys the PSX crashed today, KSE-100 index gir gaya buri tarah, investors tension mein.",
     "code_switched", [("PSX", "ORG"), ("today", "DATE"), ("KSE-100", "MISC")]),
    ("Shaheen Afridi ne Multan Sultans ke against 5 wickets liye, kya bowling thi yaar.",
     "roman_urdu", [("Shaheen Afridi", "PERSON"), ("Multan Sultans", "ORG")]),
    ("Rana Sanaullah ne Punjab Police ko orders diye, Faisalabad mein operation shuru.",
     "roman_urdu", [("Rana Sanaullah", "PERSON"), ("Punjab Police", "ORG"), ("Faisalabad", "LOCATION")]),
    ("Watched PM Shehbaz Sharif's speech on PTV, honestly kuch naya nahi tha.",
     "code_switched", [("PM", "MISC"), ("Shehbaz Sharif", "PERSON"), ("PTV", "ORG")]),
    ("Sialkot ke factory workers ne strike ki, wages barhane ka mutalba kar rahe hain.",
     "roman_urdu", [("Sialkot", "LOCATION")]),
    ("Just booked a Serene Air ticket to Islamabad on foodpanda by mistake, lol.",
     "code_switched", [("Serene Air", "ORG"), ("Islamabad", "LOCATION"), ("foodpanda", "BRAND")]),
    ("Hamza Shehbaz aur Ata Tarar ne Model Town mein PMLN ka ijlas kiya.",
     "roman_urdu", [("Hamza Shehbaz", "PERSON"), ("Ata Tarar", "PERSON"), ("Model Town", "LOCATION"), ("PMLN", "ORG")]),
    ("Ramzan next week se shuru, Ehsaas program ke through ration milega inshallah.",
     "code_switched", [("Ramzan", "DATE"), ("next week", "DATE"), ("Ehsaas program", "ORG")]),
]


def normalize_ws(s):
    return re.sub(r"\s+", " ", s).strip()


def build(text, pairs):
    ents, cursor, problems = [], 0, []
    for etext, etype in pairs:
        etext = normalize_ws(etext)
        idx = text.find(etext, cursor)
        if idx == -1:
            idx = text.find(etext)
        if idx == -1:
            problems.append(etext)
            continue
        ents.append({"text": etext, "type": etype, "start": idx, "end": idx + len(etext)})
        cursor = idx + len(etext)
    return ents, problems


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(here, "gold_sets", "ner_gold.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    all_problems = 0
    with io.open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["text", "entities", "lang", "source"])
        for text, lang, pairs in EXAMPLES:
            text = normalize_ws(text)
            ents, problems = build(text, pairs)
            if problems:
                all_problems += len(problems)
                print(f"[WARN] could not locate {problems} in: {text}", file=sys.stderr)
            w.writerow([text, json.dumps(ents, ensure_ascii=False), lang, "placeholder"])

    n_hard = sum(1 for _, l, _ in EXAMPLES if l in ("roman_urdu", "code_switched"))
    print(f"Wrote {len(EXAMPLES)} PLACEHOLDER rows -> {out}")
    print(f"  hard (roman_urdu/code_switched): {n_hard}")
    print(f"  entity-location problems: {all_problems}")
    print("  NOTE: placeholder data — review labels and expand to ~150 rows before real eval.")


if __name__ == "__main__":
    main()
