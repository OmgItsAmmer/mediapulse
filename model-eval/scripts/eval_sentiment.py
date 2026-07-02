#!/usr/bin/env python3
"""
eval_sentiment.py — evaluate ONE sentiment model on one or more test CSVs.

Model types (config.yaml, models.sentiment):
  - hf_pipeline : HuggingFace transformers sentiment pipeline (e.g.
                  cardiffnlp/twitter-xlm-roberta-base-sentiment) — the resource-efficient
                  baseline. Needs torch+transformers -> RUN WITH conda-base python:
                  /home/temp/miniconda3/bin/python  (the eval venv has no torch).
  - vllm        : chat completion to a vLLM server; few-shot, sarcasm-aware; output = one
                  label word. (openai client is present in conda-base too.)

Metrics: accuracy + macro-F1 overall, AND a SEPARATE accuracy on sarcasm=True rows
(printed separately, NOT averaged in) — sarcasm is the hard subset for this task.

Outputs: results/sentiment_<model>_<dataset>_<ts>.csv + .summary.json

Invocation (always conda-base python because of torch):
  CONDA=/home/temp/miniconda3/bin/python
  $CONDA scripts/eval_sentiment.py --model xlmr_sentiment_baseline
  VENV_PY=$CONDA ./scripts/serve_and_eval.sh Qwen/Qwen3-4B-Instruct-2507 eval_sentiment.py
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
LABELS = ["positive", "negative", "neutral"]
LABEL_MAP = {
    "positive": "positive", "pos": "positive", "p": "positive", "2": "positive", "label_2": "positive",
    "negative": "negative", "neg": "negative", "n": "negative", "0": "negative", "label_0": "negative",
    "neutral": "neutral", "neu": "neutral", "o": "neutral", "1": "neutral", "label_1": "neutral",
}


def norm_label(x):
    if x is None:
        return None
    return LABEL_MAP.get(str(x).strip().lower())


def is_true(x):
    return str(x).strip().lower() in ("true", "1", "yes")


# ============================ prompt (vLLM) =================================
SYSTEM_PROMPT = (
    "You are a sentiment classifier for Pakistani social-media posts written in English, "
    "Urdu, Roman Urdu, or code-switched text.\n"
    "Classify the OVERALL INTENDED sentiment as exactly one of: positive, negative, neutral.\n"
    "IMPORTANT: watch for SARCASM/irony — when the literal words are positive but the intent "
    "is negative (or vice-versa), label the INTENT, not the literal words.\n"
    "Reply with ONLY one lowercase word: positive, negative, or neutral. No punctuation, no explanation."
)
# 6 few-shot examples: sarcastic (both directions), genuine, neutral, Roman-Urdu + code-switched.
FEWSHOT = [
    ("Wah kya government hai, mehngai ne record tor diya. Shabash!", "negative"),   # sarcasm +words/-intent
    ("Oh great, another load shedding in this heat. Thank you WAPDA.", "negative"), # sarcasm code-switched
    ("Yaar tum to bade 'bekaar' ho, itna acha result la kar sabko peeche chor diya!", "positive"),  # sarcasm -words/+intent
    ("Alhamdulillah aaj interview bohat acha hua, umeed hai job mil jaye.", "positive"),  # genuine +
    ("Mera phone chori ho gaya bazaar mein, bohat pareshan hoon.", "negative"),     # genuine -
    ("Kal meeting 4 baje office mein hai, sab time pe aa jayen.", "neutral"),        # neutral
]


def build_messages(text):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for u, a in FEWSHOT:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": text})
    return msgs


def parse_label(raw):
    if not raw:
        return None
    s = raw.strip().lower()
    if s in LABELS:
        return s
    found = [lab for lab in LABELS if re.search(r"\b" + lab + r"\b", s)]
    return found[0] if found else None


# ============================ predictors ===================================
class HFPipelinePredictor:
    def __init__(self, model_id):
        import torch
        from transformers import pipeline
        device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline("sentiment-analysis", model=model_id, tokenizer=model_id,
                             device=device, truncation=True, max_length=256)
        self.device = device

    def predict(self, text):
        t0 = time.perf_counter()
        out = self.pipe(text)[0]
        latency = (time.perf_counter() - t0) * 1000.0
        return norm_label(out.get("label")), latency, True, ""


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
                model=self.model_id, messages=build_messages(text),
                temperature=self.gen.get("temperature", 0.0), max_tokens=8)
            raw = resp.choices[0].message.content or ""
        except Exception as ex:
            return None, (time.perf_counter() - t0) * 1000.0, False, f"API_ERROR: {ex}"
        latency = (time.perf_counter() - t0) * 1000.0
        label = parse_label(raw)
        return label, latency, (label is not None), (raw[:60] if label is None else "")


# ============================ metrics ======================================
def acc(y_true, y_pred):
    n = len(y_true)
    return (sum(1 for a, b in zip(y_true, y_pred) if a == b) / n) if n else 0.0


def macro_f1(y_true, y_pred, per_class=False):
    from sklearn.metrics import f1_score
    if not y_true:
        return ({} if per_class else 0.0)
    yp = [p if p in LABELS else "__none__" for p in y_pred]
    if per_class:
        scores = f1_score(y_true, yp, labels=LABELS, average=None, zero_division=0)
        return {lab: round(float(s), 4) for lab, s in zip(LABELS, scores)}
    return round(float(f1_score(y_true, yp, labels=LABELS, average="macro", zero_division=0)), 4)


def sanitize(s):
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)


# ============================ config / resolution ==========================
def load_config(path):
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_model(cfg, model_arg):
    for spec in cfg.get("models", {}).get("sentiment", []):
        if model_arg in (spec.get("name"), spec.get("model_id")):
            return dict(spec)
    if model_arg and "/" in model_arg:
        return {"name": model_arg.split("/")[-1], "type": "vllm", "model_id": model_arg}
    return None


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
    base = f"sentiment_{sanitize(spec['name'])}_{stem}_{ts}"
    per_row_path = os.path.join(results_dir, base + ".csv")
    summary_path = os.path.join(results_dir, base + ".summary.json")

    yt, yp, lats = [], [], []
    sarc_t, sarc_p = [], []
    by_lang = {}
    parse_failures = 0

    with open(per_row_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "lang", "source", "sarcasm", "text", "gold", "pred",
                    "correct", "latency_ms", "parse_ok", "error"])
        for i, r in enumerate(rows):
            text = r["text"]
            gold = norm_label(r.get("label"))
            if gold is None:
                continue
            pred, latency, ok, err = predictor.predict(text)
            if not ok:
                parse_failures += 1
            lats.append(latency)
            yt.append(gold); yp.append(pred)
            lang = r.get("lang", "")
            by_lang.setdefault(lang, [0, 0])
            by_lang[lang][1] += 1
            if pred == gold:
                by_lang[lang][0] += 1
            if is_true(r.get("sarcasm", "")):
                sarc_t.append(gold); sarc_p.append(pred)
            w.writerow([i, lang, r.get("source", ""), r.get("sarcasm", ""), text,
                        gold, pred, int(pred == gold), round(latency, 1), ok, err])

    metrics_overall = {"accuracy": round(acc(yt, yp), 4), "macro_f1": macro_f1(yt, yp),
                       "per_class_f1": macro_f1(yt, yp, per_class=True)}
    metrics_hard = None
    if sarc_t:
        metrics_hard = {"accuracy": round(acc(sarc_t, sarc_p), 4),
                        "macro_f1": macro_f1(sarc_t, sarc_p), "n": len(sarc_t)}

    summary = {
        "task": "sentiment",
        "model": spec["name"], "model_type": spec["type"], "model_id": spec.get("model_id"),
        "dataset": os.path.relpath(csv_path, HERE),
        "n_rows": len(yt), "n_hard_rows": len(sarc_t),
        "parse_failures": parse_failures if is_llm else None,
        "metrics_overall": metrics_overall,
        "metrics_hard_subset": metrics_hard,   # hard = sarcasm=True
        "by_lang_accuracy": {k: {"accuracy": round(v[0] / v[1], 4), "n": v[1]}
                             for k, v in by_lang.items() if v[1]},
        "latency_ms": {
            "mean": round(statistics.mean(lats), 1) if lats else None,
            "median": round(statistics.median(lats), 1) if lats else None,
            "p95": round(sorted(lats)[int(0.95 * (len(lats) - 1))], 1) if lats else None,
            "total_s": round(sum(lats) / 1000.0, 1) if lats else None,
        },
        "note": spec.get("note", ""), "timestamp": ts,
        "per_row_csv": os.path.relpath(per_row_path, HERE),
    }
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(f"\n  [{spec['name']}] on {stem}  ({len(yt)} rows)")
    print(f"    overall  accuracy={metrics_overall['accuracy']:.3f}  macro-F1={metrics_overall['macro_f1']:.3f}")
    if metrics_hard:
        print(f"    SARCASM ({metrics_hard['n']})  accuracy={metrics_hard['accuracy']:.3f}  "
              f"macro-F1={metrics_hard['macro_f1']:.3f}   <-- hard subset (reported separately)")
    if is_llm:
        print(f"    parse failures: {parse_failures}/{len(yt)}")
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
              f"       Serve first, or run via serve_and_eval.sh (with VENV_PY=conda python).",
              file=sys.stderr)
        return False


def main():
    ap = argparse.ArgumentParser(description="Evaluate one sentiment model.")
    ap.add_argument("--model", default=os.environ.get("EVAL_MODEL"))
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL"))
    ap.add_argument("--limit", type=int, default=int(os.environ.get("EVAL_LIMIT", "0")),
                    help="cap rows per dataset (0=all; combined.csv is ~21k so set this for LLM runs)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if not args.model:
        print("ERROR: no model given. Use --model <name|hf_id> or set $EVAL_MODEL.", file=sys.stderr)
        sys.exit(2)
    spec = resolve_model(cfg, args.model)
    if not spec:
        avail = [s["name"] for s in cfg.get("models", {}).get("sentiment", [])]
        print(f"ERROR: model '{args.model}' not in config.sentiment and not an HF id. Available: {avail}", file=sys.stderr)
        sys.exit(2)

    if args.datasets:
        datasets = args.datasets
    else:
        cand = [os.path.join(HERE, "gold_sets/sentiment_gold.csv"),
                os.path.join(HERE, "datasets/sentiment/combined.csv")]
        datasets = [p for p in cand if os.path.exists(p)]
    if not datasets:
        print("ERROR: no datasets. Run prepare_sentiment_data.py + make_sentiment_gold.py first.", file=sys.stderr)
        sys.exit(2)

    results_dir = os.path.join(HERE, cfg.get("paths", {}).get("results", "results"))
    os.makedirs(results_dir, exist_ok=True)
    is_llm = spec["type"] == "vllm"
    print(f"Model: {spec['name']} (type={spec['type']}, id={spec.get('model_id')})")

    if spec["type"] == "hf_pipeline":
        predictor = HFPipelinePredictor(spec["model_id"])
        print(f"  (transformers pipeline on device={'GPU' if predictor.device == 0 else 'CPU'})")
    elif spec["type"] == "vllm":
        base_url = args.base_url or cfg.get("vllm_base_url")
        if not preflight_vllm(base_url):
            sys.exit(1)
        predictor = VLLMPredictor(spec["model_id"], base_url,
                                  cfg.get("vllm_api_key", "EMPTY"), cfg.get("vllm_gen", {}))
    else:
        print(f"ERROR: eval_sentiment.py does not handle type '{spec['type']}'.", file=sys.stderr)
        sys.exit(2)

    if args.limit == 0 and any("combined" in d for d in datasets) and is_llm:
        print("NOTE: combined.csv is ~21k rows and no --limit set — this LLM run will be very long. "
              "Consider EVAL_LIMIT=200.", file=sys.stderr)

    for csv_path in datasets:
        if not os.path.exists(csv_path):
            print(f"  skip (missing): {csv_path}", file=sys.stderr)
            continue
        eval_dataset(spec, predictor, csv_path, results_dir, args.limit, is_llm)


if __name__ == "__main__":
    main()
