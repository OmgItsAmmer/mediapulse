#!/usr/bin/env python3
"""
prepare_ner_data.py — normalize public Urdu NER corpora into a single CSV.

Sources (downloaded from the mirfan899/Urdu GitHub repo, ner/ folder):
  - UNER      (datasets/ner/raw/ner/uner.txt)     UTF-16LE, one sentence/line,
              inline <TYPE>...</TYPE> tags.
  - MK-PUCIT  (datasets/ner/raw/ner/mk-pucit.txt) UTF-8, one "token\\tTag" per line,
              NO sentence boundaries -> segmented into entity-safe token windows
              (an approximation; flagged via source="mk-pucit").

Output: datasets/ner/ner_public.csv with columns:
  text, entities, lang, source
where `entities` is a JSON list of {"text","type","start","end"} and each type is
one of the common set: PERSON, ORG, LOCATION, BRAND, DATE, MISC.
(BRAND does not occur in these Urdu-script corpora; it appears only in the gold set.)
"""
import argparse
import csv
import io
import json
import os
import re
import sys

# ---- entity-type mapping to the common set --------------------------------
# NUMBER/NUMBERS are dropped (not in the target set, high-noise for our tasks).
TYPE_MAP = {
    "PERSON": "PERSON", "PER": "PERSON",
    "ORGANIZATION": "ORG", "ORGANISATION": "ORG", "ORG": "ORG",
    "LOCATION": "LOCATION", "LOC": "LOCATION", "GPE": "LOCATION",
    "DATE": "DATE",
    "TIME": "MISC",
    "DESIGNATION": "MISC",
    "BRAND": "BRAND",
    "MISC": "MISC",
}
DROP_TAGS = {"NUMBER", "NUMBERS", "O", "OTHER"}


def map_type(raw_tag):
    """Return the common-set type for a raw tag, or None to drop it."""
    t = raw_tag.strip().upper()
    if t in DROP_TAGS:
        return None
    return TYPE_MAP.get(t, "MISC")


def normalize_ws(s):
    return re.sub(r"\s+", " ", s).strip()


def build_entities_by_search(text, pairs):
    """Given clean `text` and ordered (entity_text, type) pairs (in surface order),
    compute char offsets by scanning left-to-right. Returns list of entity dicts."""
    ents = []
    cursor = 0
    for etext, etype in pairs:
        etext = normalize_ws(etext)
        if not etext or etype is None:
            continue
        idx = text.find(etext, cursor)
        if idx == -1:                       # fall back to a global search
            idx = text.find(etext)
        if idx == -1:
            continue                        # entity text not locatable -> skip
        ents.append({"text": etext, "type": etype,
                     "start": idx, "end": idx + len(etext)})
        cursor = idx + len(etext)
    return ents


# ---- UNER -----------------------------------------------------------------
TAG_RE = re.compile(r"<([A-Z][A-Z_]*)>(.*?)</\1>", re.DOTALL)


def parse_uner(path, max_rows=None):
    rows = []
    if not os.path.exists(path):
        print(f"[UNER] MISSING: {path}", file=sys.stderr)
        return rows
    with io.open(path, encoding="utf-16") as fh:
        for line in fh:
            raw = line.strip()
            if not raw:
                continue
            # clean text = tags removed (inner kept); collect (inner, type) in order
            pairs = []
            clean_parts, last = [], 0
            for m in TAG_RE.finditer(raw):
                clean_parts.append(raw[last:m.start()])
                inner = m.group(2)
                clean_parts.append(inner)
                pairs.append((inner, map_type(m.group(1))))
                last = m.end()
            clean_parts.append(raw[last:])
            text = normalize_ws("".join(clean_parts))
            if not text:
                continue
            ents = build_entities_by_search(text, pairs)
            rows.append({"text": text, "entities": ents,
                         "lang": "urdu", "source": "uner"})
            if max_rows and len(rows) >= max_rows:
                break
    return rows


# ---- MK-PUCIT (streaming, segment into entity-safe token windows) ---------
def parse_mkpucit(path, max_rows=None, window=25):
    rows = []
    if not os.path.exists(path):
        print(f"[MK-PUCIT] MISSING: {path}", file=sys.stderr)
        return rows
    toks = []  # list of (token, mapped_type_or_None)

    def flush(toks):
        text, ents = "", []
        open_type = open_start = open_end = None
        for i, (tok, mtype) in enumerate(toks):
            if i:
                text += " "
            start = len(text)
            text += tok
            end = len(text)
            if mtype is None:
                if open_type is not None:
                    ents.append({"text": text[open_start:open_end], "type": open_type,
                                 "start": open_start, "end": open_end})
                    open_type = None
            elif open_type == mtype:
                open_end = end
            else:
                if open_type is not None:
                    ents.append({"text": text[open_start:open_end], "type": open_type,
                                 "start": open_start, "end": open_end})
                open_type, open_start, open_end = mtype, start, end
        if open_type is not None:
            ents.append({"text": text[open_start:open_end], "type": open_type,
                         "start": open_start, "end": open_end})
        return text, ents

    with io.open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or "\t" not in line:
                continue
            tok, tag = line.split("\t", 1)
            toks.append((tok, map_type(tag)))
            # cut a window once long enough AND at a non-entity boundary
            if len(toks) >= window and toks[-1][1] is None:
                text, ents = flush(toks)
                if text:
                    rows.append({"text": text, "entities": ents,
                                 "lang": "urdu", "source": "mk-pucit"})
                toks = []
                if max_rows and len(rows) >= max_rows:
                    break
    return rows


def write_csv(rows, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with io.open(out_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["text", "entities", "lang", "source"])
        for r in rows:
            w.writerow([r["text"],
                        json.dumps(r["entities"], ensure_ascii=False),
                        r["lang"], r["source"]])


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--uner", default=os.path.join(here, "datasets/ner/raw/ner/uner.txt"))
    ap.add_argument("--mkpucit", default=os.path.join(here, "datasets/ner/raw/ner/mk-pucit.txt"))
    ap.add_argument("--out", default=os.path.join(here, "datasets/ner/ner_public.csv"))
    ap.add_argument("--max-uner", type=int, default=0, help="0 = all")
    ap.add_argument("--max-mkpucit", type=int, default=500,
                    help="max segmented windows to sample from MK-PUCIT (0 = all)")
    ap.add_argument("--mkpucit-window", type=int, default=25)
    args = ap.parse_args()

    uner = parse_uner(args.uner, args.max_uner or None)
    mk = parse_mkpucit(args.mkpucit, args.max_mkpucit or None, args.mkpucit_window)
    rows = uner + mk
    write_csv(rows, args.out)

    def ent_count(rs):
        return sum(len(r["entities"]) for r in rs)

    print("=== NER public dataset normalization ===")
    print(f"  UNER     : {len(uner):5d} rows, {ent_count(uner):6d} entities")
    print(f"  MK-PUCIT : {len(mk):5d} rows, {ent_count(mk):6d} entities "
          f"(segmented into ~{args.mkpucit_window}-token windows; approximate sentences)")
    print(f"  TOTAL    : {len(rows):5d} rows -> {args.out}")

    # sanity: offsets must reproduce the entity surface text
    bad = 0
    for r in rows:
        for e in r["entities"]:
            if r["text"][e["start"]:e["end"]] != e["text"]:
                bad += 1
    print(f"  offset self-check: {'OK' if bad == 0 else str(bad) + ' MISMATCHES'}")


if __name__ == "__main__":
    main()
