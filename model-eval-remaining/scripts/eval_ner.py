#!/usr/bin/env python3
"""
eval_ner.py — evaluate ONE NER model on one or more test CSVs.

Model types (from config.yaml, models.ner):
  - spacy : loads a spaCy pipeline (e.g. en_core_web_sm) locally. NOTE: en_core_web_sm
            is an ENGLISH-ONLY baseline with no Urdu/Roman-Urdu support — flagged as such.
  - vllm  : sends chat completions to a vLLM OpenAI-compatible server (start it first,
            or use scripts/serve_and_eval.sh). Strict JSON-only prompt + few-shot examples
            covering Roman-Urdu / code-switched names.

Metrics (entity level, micro-averaged): precision / recall / F1 for
  - exact    : span (start,end) AND type must match
  - partial  : same type AND char spans overlap  (looser)
  - untyped  : span matches, type ignored        (diagnoses type confusion)
Reported overall AND on the HARD subset (rows with lang in roman_urdu/code_switched),
since Roman-Urdu / code-switched accuracy is the whole point of this eval.

Outputs (per model x dataset):
  results/ner_<model>_<dataset>_<ts>.csv          per-row gold/pred/latency
  results/ner_<model>_<dataset>_<ts>.summary.json aggregate metrics

Invocation:
  # direct (spaCy needs no server):
  venv/bin/python scripts/eval_ner.py --model spacy_en_core_web_sm
  # via serve_and_eval.sh (sets EVAL_MODEL + VLLM_BASE_URL for the served model):
  ./scripts/serve_and_eval.sh Qwen/Qwen3-8B eval_ner.py
"""
import argparse
import csv
import datetime as dt
import json
import os
import re
import statistics
import sys
import time

import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMON_TYPES = {"PERSON", "ORG", "LOCATION", "BRAND", "DATE", "MISC"}
HARD_LANGS = {"roman_urdu", "code_switched"}

# --- type normalization for model / spaCy labels -> common set --------------
TYPE_SYNONYMS = {
    "PERSON": "PERSON", "PER": "PERSON", "PEOPLE": "PERSON",
    "ORG": "ORG", "ORGANIZATION": "ORG", "ORGANISATION": "ORG", "COMPANY": "ORG",
    "LOCATION": "LOCATION", "LOC": "LOCATION", "GPE": "LOCATION", "PLACE": "LOCATION",
    "FAC": "LOCATION", "FACILITY": "LOCATION",
    "BRAND": "BRAND", "PRODUCT": "BRAND",
    "DATE": "DATE",
    "TIME": "MISC", "NORP": "MISC", "EVENT": "MISC", "WORK_OF_ART": "MISC",
    "LAW": "MISC", "LANGUAGE": "MISC", "MISC": "MISC", "MISCELLANEOUS": "MISC",
}
# spaCy numeric labels we simply drop (not in the target set)
SPACY_DROP = {"MONEY", "PERCENT", "QUANTITY", "ORDINAL", "CARDINAL"}


def norm_type(t):
    if not t:
        return None
    t = str(t).strip().upper()
    if t in SPACY_DROP:
        return None
    return TYPE_SYNONYMS.get(t, "MISC")


def normalize_ws(s):
    return re.sub(r"\s+", " ", str(s)).strip()


# ============================ prompt (vLLM) =================================
SYSTEM_PROMPT = (
    "You are an expert Named Entity Recognition system for Pakistani social-media text, "
    "which may be English, Urdu, Roman Urdu (Urdu in Latin script), or code-switched.\n"
    "Extract all named entities and respond with ONLY a JSON object, no prose, no markdown.\n"
    'Schema: {"entities": [{"text": <exact substring>, "type": <TYPE>, "start": <int>, "end": <int>}]}\n'
    "TYPE must be exactly one of: PERSON, ORG, LOCATION, BRAND, DATE, MISC.\n"
    "Rules:\n"
    "- text must be an exact substring of the input.\n"
    "- Political parties/acronyms (PTI, PMLN, PPP, JUI-F) -> ORG. Companies/apps (Daraz, Careem, Samsung) -> BRAND.\n"
    "- Roman-Urdu person names (Imran Khan, Bilawal Bhutto) -> PERSON. Cities/countries -> LOCATION.\n"
    "- Dates/relative time (aaj, kal, 8 February) -> DATE. Titles/other (CJP, PM) -> MISC.\n"
    '- If there are no entities, return {"entities": []}.'
)
# 4 few-shot examples covering Roman-Urdu + code-switched + brands.
FEWSHOT = [
    ('Text: "Imran Khan ne Lahore mein PTI ka jalsa kiya"',
     '{"entities": [{"text": "Imran Khan", "type": "PERSON", "start": 0, "end": 10}, '
     '{"text": "Lahore", "type": "LOCATION", "start": 14, "end": 20}, '
     '{"text": "PTI", "type": "ORG", "start": 26, "end": 29}]}'),
    ('Text: "Bilawal met the PPP workers in Karachi yesterday"',
     '{"entities": [{"text": "Bilawal", "type": "PERSON", "start": 0, "end": 7}, '
     '{"text": "PPP", "type": "ORG", "start": 16, "end": 19}, '
     '{"text": "Karachi", "type": "LOCATION", "start": 31, "end": 38}, '
     '{"text": "yesterday", "type": "DATE", "start": 39, "end": 48}]}'),
    ('Text: "Maine Daraz se naya Samsung phone order kiya"',
     '{"entities": [{"text": "Daraz", "type": "BRAND", "start": 6, "end": 11}, '
     '{"text": "Samsung", "type": "BRAND", "start": 20, "end": 27}]}'),
    ('Text: "Nawaz Sharif aur Shehbaz Sharif PMLN ke leader hain"',
     '{"entities": [{"text": "Nawaz Sharif", "type": "PERSON", "start": 0, "end": 12}, '
     '{"text": "Shehbaz Sharif", "type": "PERSON", "start": 17, "end": 31}, '
     '{"text": "PMLN", "type": "ORG", "start": 32, "end": 36}]}'),
]


def build_messages(text):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for u, a in FEWSHOT:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": 'Text: "' + text.replace('"', "'") + '"\nReturn ONLY the JSON.'})
    return msgs


def extract_json(raw):
    if not raw:
        return None
    raw = raw.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if m:
        raw = m.group(1).strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    i, j = raw.find("{"), raw.rfind("}")
    if i != -1 and j != -1 and j > i:
        try:
            return json.loads(raw[i:j + 1])
        except Exception:
            return None
    return None


def locate_entities(text, raw_entities):
    """Re-derive char offsets by searching the source text (LLM offsets are unreliable).
    Robust to ordering and duplicate surface forms."""
    out, used = [], set()
    if not isinstance(raw_entities, list):
        return out
    for e in raw_entities:
        if not isinstance(e, dict):
            continue
        etext = normalize_ws(e.get("text", ""))
        etype = norm_type(e.get("type", ""))
        if not etext or etype is None:
            continue
        idx, start = -1, 0
        while True:
            p = text.find(etext, start)
            if p == -1:
                break
            if p not in used:
                idx = p
                break
            start = p + 1
        if idx == -1:
            continue
        used.add(idx)
        out.append({"text": etext, "type": etype, "start": idx, "end": idx + len(etext)})
    return out


# ============================ predictors ===================================
class SpacyPredictor:
    def __init__(self, model_id):
        import spacy
        self.nlp = spacy.load(model_id)

    def predict(self, text):
        t0 = time.perf_counter()
        doc = self.nlp(text)
        ents = []
        for e in doc.ents:
            typ = norm_type(e.label_)
            if typ is None:
                continue
            ents.append({"text": e.text, "type": typ,
                         "start": e.start_char, "end": e.end_char})
        return ents, (time.perf_counter() - t0) * 1000.0, True, ""


class VLLMPredictor:
    def __init__(self, model_id, base_url, api_key, gen):
        from openai import OpenAI
        self.model_id = model_id
        self.gen = gen
        self.client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY",
                             timeout=gen.get("request_timeout_s", 60))

    def predict(self, text):
        t0 = time.perf_counter()
        try:
            resp = self.client.chat.completions.create(
                model=self.model_id,
                messages=build_messages(text),
                temperature=self.gen.get("temperature", 0.0),
                max_tokens=self.gen.get("max_tokens", 512),
            )
            raw = resp.choices[0].message.content or ""
        except Exception as ex:
            return [], (time.perf_counter() - t0) * 1000.0, False, f"API_ERROR: {ex}"
        latency = (time.perf_counter() - t0) * 1000.0
        parsed = extract_json(raw)
        if parsed is None:
            return [], latency, False, raw[:200]
        ents = locate_entities(text, parsed.get("entities", []) if isinstance(parsed, dict) else [])
        return ents, latency, True, ""


# ============================ metrics ======================================
def overlap(a, b):
    return max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


def row_counts(gold, pred):
    from collections import Counter
    gc = Counter((e["start"], e["end"], e["type"]) for e in gold)
    pc = Counter((e["start"], e["end"], e["type"]) for e in pred)
    tp_exact = sum((gc & pc).values())
    gcu = Counter((e["start"], e["end"]) for e in gold)
    pcu = Counter((e["start"], e["end"]) for e in pred)
    tp_untyped = sum((gcu & pcu).values())
    used = [False] * len(pred)
    tp_partial = 0
    for g in gold:
        for i, p in enumerate(pred):
            if not used[i] and p["type"] == g["type"] and overlap(g, p) > 0:
                used[i] = True
                tp_partial += 1
                break
    return {"tp_exact": tp_exact, "tp_untyped": tp_untyped, "tp_partial": tp_partial,
            "n_gold": len(gold), "n_pred": len(pred)}


def prf(tp, n_pred, n_gold):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "tp": tp, "n_pred": n_pred, "n_gold": n_gold}


def aggregate(counts_list):
    def s(k):
        return sum(c[k] for c in counts_list)
    return {
        "exact": prf(s("tp_exact"), s("n_pred"), s("n_gold")),
        "partial": prf(s("tp_partial"), s("n_pred"), s("n_gold")),
        "untyped_exact": prf(s("tp_untyped"), s("n_pred"), s("n_gold")),
    }


# ============================ config / model resolution ====================
def load_config(path):
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_model(cfg, model_arg):
    for spec in cfg.get("models", {}).get("ner", []):
        if model_arg in (spec.get("name"), spec.get("model_id")):
            return dict(spec)
    if model_arg and "/" in model_arg:   # looks like an HF id -> assume vLLM
        return {"name": model_arg.split("/")[-1], "type": "vllm", "model_id": model_arg}
    return None


def sanitize(s):
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)


# ============================ eval one dataset =============================
def eval_dataset(spec, predictor, csv_path, results_dir, limit, is_llm):
    rows = []
    with open(csv_path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
            if limit and len(rows) >= limit:
                break

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    base = f"ner_{sanitize(spec['name'])}_{stem}_{ts}"
    per_row_path = os.path.join(results_dir, base + ".csv")
    summary_path = os.path.join(results_dir, base + ".summary.json")

    all_counts, hard_counts, latencies = [], [], []
    parse_failures = 0

    with open(per_row_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "lang", "source", "text", "n_gold", "n_pred",
                    "tp_exact", "tp_partial", "latency_ms", "parse_ok",
                    "gold_entities", "pred_entities", "error"])
        for i, r in enumerate(rows):
            text = r["text"]
            gold = json.loads(r["entities"]) if r.get("entities") else []
            pred, latency, ok, err = predictor.predict(text)
            if not ok:
                parse_failures += 1
            latencies.append(latency)
            c = row_counts(gold, pred)
            all_counts.append(c)
            if r.get("lang") in HARD_LANGS:
                hard_counts.append(c)
            w.writerow([i, r.get("lang", ""), r.get("source", ""), text,
                        c["n_gold"], c["n_pred"], c["tp_exact"], c["tp_partial"],
                        round(latency, 1), ok,
                        json.dumps(gold, ensure_ascii=False),
                        json.dumps(pred, ensure_ascii=False), err])

    summary = {
        "task": "ner",
        "model": spec["name"],
        "model_type": spec["type"],
        "model_id": spec.get("model_id"),
        "dataset": os.path.relpath(csv_path, HERE),
        "n_rows": len(rows),
        "n_hard_rows": len(hard_counts),
        "parse_failures": parse_failures if is_llm else None,
        "metrics_overall": aggregate(all_counts) if all_counts else None,
        "metrics_hard_subset": aggregate(hard_counts) if hard_counts else None,
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 1) if latencies else None,
            "median": round(statistics.median(latencies), 1) if latencies else None,
            "p95": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 1) if latencies else None,
            "total_s": round(sum(latencies) / 1000.0, 1) if latencies else None,
        },
        "note": spec.get("note", ""),
        "timestamp": ts,
        "per_row_csv": os.path.relpath(per_row_path, HERE),
    }
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    # console report
    m = summary["metrics_overall"]
    hm = summary["metrics_hard_subset"]
    print(f"\n  [{spec['name']}] on {stem}  ({len(rows)} rows)")
    if m:
        print(f"    overall   exact F1={m['exact']['f1']:.3f}  partial F1={m['partial']['f1']:.3f}"
              f"  (P={m['exact']['precision']:.3f} R={m['exact']['recall']:.3f})")
    if hm:
        print(f"    HARD ({len(hard_counts)}) exact F1={hm['exact']['f1']:.3f}  partial F1={hm['partial']['f1']:.3f}")
    if is_llm:
        print(f"    parse failures: {parse_failures}/{len(rows)}")
    print(f"    latency mean={summary['latency_ms']['mean']} ms | -> {os.path.basename(summary_path)}")
    return summary


def preflight_vllm(base_url):
    import requests
    root = base_url.rsplit("/v1", 1)[0]
    try:
        requests.get(root + "/health", timeout=5).raise_for_status()
        return True
    except Exception as ex:
        print(f"ERROR: vLLM not reachable at {base_url} ({ex}).\n"
              f"       Start it first:  vllm serve <model> --port 8000\n"
              f"       or run via:      ./scripts/serve_and_eval.sh <model> eval_ner.py",
              file=sys.stderr)
        return False


def main():
    ap = argparse.ArgumentParser(description="Evaluate one NER model.")
    ap.add_argument("--model", default=os.environ.get("EVAL_MODEL"),
                    help="config model name or HF id (default: $EVAL_MODEL)")
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="CSV paths (default: gold + public if present)")
    ap.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL"))
    ap.add_argument("--limit", type=int, default=int(os.environ.get("EVAL_LIMIT", "0")),
                    help="cap rows per dataset (0=all; or set $EVAL_LIMIT for quick smoke runs)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if not args.model:
        print("ERROR: no model given. Use --model <name|hf_id> or set $EVAL_MODEL.", file=sys.stderr)
        sys.exit(2)
    spec = resolve_model(cfg, args.model)
    if not spec:
        avail = [s["name"] for s in cfg.get("models", {}).get("ner", [])]
        print(f"ERROR: model '{args.model}' not in config.ner and not an HF id. Available: {avail}", file=sys.stderr)
        sys.exit(2)

    # default datasets
    if args.datasets:
        datasets = args.datasets
    else:
        cand = [os.path.join(HERE, "gold_sets/ner_gold.csv"),
                os.path.join(HERE, "datasets/ner/ner_public.csv")]
        datasets = [p for p in cand if os.path.exists(p)]
    if not datasets:
        print("ERROR: no datasets found. Run scripts/prepare_ner_data.py + make_ner_gold.py first.", file=sys.stderr)
        sys.exit(2)

    results_dir = os.path.join(HERE, cfg.get("paths", {}).get("results", "results"))
    os.makedirs(results_dir, exist_ok=True)

    is_llm = spec["type"] == "vllm"
    print(f"Model: {spec['name']} (type={spec['type']}, id={spec.get('model_id')})")
    if spec["type"] == "spacy" and spec.get("model_id") == "en_core_web_sm":
        print("WARNING: en_core_web_sm is an ENGLISH-ONLY baseline — no Urdu/Roman-Urdu support. "
              "Low scores on Urdu-script/Roman-Urdu data are expected.")

    # build predictor
    if spec["type"] == "spacy":
        predictor = SpacyPredictor(spec["model_id"])
    elif spec["type"] == "vllm":
        base_url = args.base_url or cfg.get("vllm_base_url")
        if not preflight_vllm(base_url):
            sys.exit(1)
        predictor = VLLMPredictor(spec["model_id"], base_url,
                                  cfg.get("vllm_api_key", "EMPTY"),
                                  cfg.get("vllm_gen", {}))
    else:
        print(f"ERROR: eval_ner.py does not handle model type '{spec['type']}'.", file=sys.stderr)
        sys.exit(2)

    for csv_path in datasets:
        if not os.path.exists(csv_path):
            print(f"  skip (missing): {csv_path}", file=sys.stderr)
            continue
        eval_dataset(spec, predictor, csv_path, results_dir, args.limit, is_llm)


if __name__ == "__main__":
    main()
