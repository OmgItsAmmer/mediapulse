#!/usr/bin/env python3
"""
eval_langid.py — evaluate ONE language-ID model on one or more test CSVs.

Model types (config.yaml, models.langid):
  - fasttext   : lid.176.bin via `fasttext` (fasttext-predict wheel). Resource-efficient
                 baseline. Cannot output roman_urdu/mixed -> those get mapped to what it
                 actually predicts (usually english) — that IS the documented failure.
  - langdetect : langdetect (seeded, deterministic). Same limitation as fastText.
  - vllm       : chat completion, 4-class prompt + 4 few-shot. Output = one label.

Reports PER-CLASS accuracy + a CONFUSION MATRIX (rows=true, cols=predicted, incl `other`)
so you can see exactly how often each model tags roman_urdu / mixed as english.

Outputs: results/langid_<model>_<dataset>_<ts>.csv + .summary.json (matrix included).

Runs in the eval venv (fasttext-predict + langdetect + openai — no torch needed):
  venv/bin/python scripts/eval_langid.py --model fasttext_lid176
  ./scripts/serve_and_eval.sh Qwen/Qwen3-4B-Instruct-2507 eval_langid.py
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
CLASSES = ["english", "urdu", "roman_urdu", "mixed"]
PRED_LABELS = CLASSES + ["other"]        # baselines may emit `other`
HARD_LANGS = {"roman_urdu", "mixed"}     # where the failure mode lives
# ISO code -> our class (for fastText/langdetect); anything else -> other
ISO_MAP = {"en": "english", "ur": "urdu"}


def map_iso(code):
    return ISO_MAP.get(code, "other")


# ============================ prompt (vLLM) =================================
SYSTEM_PROMPT = (
    "You are a language identifier for Pakistani social-media text.\n"
    "Classify the text as exactly one of: english, urdu, roman_urdu, mixed.\n"
    "- english    = English only.\n"
    "- urdu       = Urdu written in Urdu/Arabic script.\n"
    "- roman_urdu = Urdu written in Latin/English letters (e.g. 'ap kaisay ho', 'theek hai yaar').\n"
    "- mixed      = code-switched: a genuine mix of English and Urdu/Roman-Urdu.\n"
    "Reply with ONLY the label word."
)
FEWSHOT = [
    ("The weather is lovely today and everyone is out.", "english"),
    ("آج بہت گرمی ہے اور لوگ پریشان ہیں۔", "urdu"),
    ("Yaar aaj bahut garmi hai, ghar pe raho.", "roman_urdu"),
    ("It's so hot today yaar, ghar pe raho please.", "mixed"),
]


def build_messages(text):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for u, a in FEWSHOT:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": text})
    return msgs


def parse_lang(raw):
    if not raw:
        return None
    s = raw.strip().lower().replace("-", "_").replace(" ", "_")
    for lab in CLASSES:
        if s == lab:
            return lab
    for lab in ["roman_urdu", "mixed", "english", "urdu"]:   # roman_urdu before urdu
        if lab in s:
            return lab
    return None


# ============================ predictors ===================================
class FastTextPredictor:
    def __init__(self, model_id):
        import fasttext
        path = model_id if os.path.isabs(model_id) else os.path.join(HERE, model_id)
        self.model = fasttext.load_model(path)

    def predict(self, text):
        t0 = time.perf_counter()
        labels, _ = self.model.predict(text.replace("\n", " "))
        latency = (time.perf_counter() - t0) * 1000.0
        code = labels[0].replace("__label__", "") if labels else ""
        return map_iso(code), latency, True, ""


class LangDetectPredictor:
    def __init__(self, _model_id):
        from langdetect import DetectorFactory
        DetectorFactory.seed = 0

    def predict(self, text):
        from langdetect import detect
        from langdetect.lang_detect_exception import LangDetectException
        t0 = time.perf_counter()
        try:
            code = detect(text)
        except LangDetectException:
            code = ""
        latency = (time.perf_counter() - t0) * 1000.0
        return map_iso(code), latency, True, ""


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
        lab = parse_lang(raw)
        return lab, latency, (lab is not None), (raw[:40] if lab is None else "")


# ============================ metrics ======================================
def macro_f1(y_true, y_pred):
    from sklearn.metrics import f1_score
    yp = [p if p in CLASSES else "__none__" for p in y_pred]
    return round(float(f1_score(y_true, yp, labels=CLASSES, average="macro", zero_division=0)), 4)


def sanitize(s):
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)


def print_matrix(cm):
    hdr = "true\\pred  " + "".join(f"{c[:9]:>11s}" for c in PRED_LABELS)
    print("    " + hdr)
    for t in CLASSES:
        row = f"{t:>10s} " + "".join(f"{cm[t][p]:>11d}" for p in PRED_LABELS)
        print("    " + row)


# ============================ config / resolution ==========================
def load_config(path):
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_model(cfg, model_arg):
    for spec in cfg.get("models", {}).get("langid", []):
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
    base = f"langid_{sanitize(spec['name'])}_{stem}_{ts}"
    per_row_path = os.path.join(results_dir, base + ".csv")
    summary_path = os.path.join(results_dir, base + ".summary.json")

    cm = {t: {p: 0 for p in PRED_LABELS} for t in CLASSES}
    lats, parse_failures = [], 0
    correct = total = 0
    hard_correct = hard_total = 0
    yt_all, yp_all = [], []

    with open(per_row_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "true_language", "source", "text", "pred",
                    "correct", "latency_ms", "parse_ok", "error"])
        for i, r in enumerate(rows):
            true = (r.get("true_language") or "").strip().lower()
            if true not in CLASSES:
                continue
            text = r["text"]
            pred, latency, ok, err = predictor.predict(text)
            if not ok:
                parse_failures += 1
            pred_b = pred if pred in PRED_LABELS else "other"
            cm[true][pred_b] += 1
            lats.append(latency)
            yt_all.append(true); yp_all.append(pred)
            total += 1
            hit = int(pred == true)
            correct += hit
            if true in HARD_LANGS:
                hard_total += 1
                hard_correct += hit
            w.writerow([i, true, r.get("source", ""), text, pred, hit,
                        round(latency, 1), ok, err])

    per_class_acc = {t: (round(cm[t][t] / sum(cm[t].values()), 4) if sum(cm[t].values()) else None)
                     for t in CLASSES}
    # headline failure metric: how often roman_urdu / mixed were called english
    ru_as_en = cm["roman_urdu"]["english"]
    mixed_as_en = cm["mixed"]["english"]

    summary = {
        "task": "langid",
        "model": spec["name"], "model_type": spec["type"], "model_id": spec.get("model_id"),
        "dataset": os.path.relpath(csv_path, HERE),
        "n_rows": total, "n_hard_rows": hard_total,
        "parse_failures": parse_failures if is_llm else None,
        "metrics_overall": {"accuracy": round(correct / total, 4) if total else 0.0,
                            "macro_f1": macro_f1(yt_all, yp_all)},
        "metrics_hard_subset": ({"accuracy": round(hard_correct / hard_total, 4), "n": hard_total}
                                if hard_total else None),
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": cm,
        "roman_urdu_called_english": ru_as_en,
        "mixed_called_english": mixed_as_en,
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

    print(f"\n  [{spec['name']}] on {stem}  ({total} rows)")
    print(f"    accuracy={summary['metrics_overall']['accuracy']:.3f}  "
          f"macro-F1={summary['metrics_overall']['macro_f1']:.3f}")
    print("    per-class accuracy: " +
          ", ".join(f"{t}={per_class_acc[t]}" for t in CLASSES))
    if summary["metrics_hard_subset"]:
        print(f"    HARD (roman_urdu+mixed, {hard_total}) accuracy="
              f"{summary['metrics_hard_subset']['accuracy']:.3f}")
    print(f"    >> roman_urdu tagged as english: {ru_as_en}/{sum(cm['roman_urdu'].values())}"
          f" | mixed tagged as english: {mixed_as_en}/{sum(cm['mixed'].values())}")
    print("    confusion matrix:")
    print_matrix(cm)
    if is_llm:
        print(f"    parse failures: {parse_failures}/{total}")
    print(f"    latency mean={summary['latency_ms']['mean']} ms | -> {os.path.basename(summary_path)}")
    return summary


def preflight_vllm(base_url):
    import requests
    root = base_url.rsplit("/v1", 1)[0]
    try:
        requests.get(root + "/health", timeout=5).raise_for_status()
        return True
    except Exception as ex:
        print(f"ERROR: vLLM not reachable at {base_url} ({ex}). Serve first / use serve_and_eval.sh.",
              file=sys.stderr)
        return False


def main():
    ap = argparse.ArgumentParser(description="Evaluate one language-ID model.")
    ap.add_argument("--model", default=os.environ.get("EVAL_MODEL"))
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL"))
    ap.add_argument("--limit", type=int, default=int(os.environ.get("EVAL_LIMIT", "0")),
                    help="cap rows per dataset (0=all)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if not args.model:
        print("ERROR: no model given. Use --model <name|hf_id> or set $EVAL_MODEL.", file=sys.stderr)
        sys.exit(2)
    spec = resolve_model(cfg, args.model)
    if not spec:
        avail = [s["name"] for s in cfg.get("models", {}).get("langid", [])]
        print(f"ERROR: model '{args.model}' not in config.langid and not an HF id. Available: {avail}", file=sys.stderr)
        sys.exit(2)

    if args.datasets:
        datasets = args.datasets
    else:
        cand = [os.path.join(HERE, "gold_sets/langid_gold.csv"),
                os.path.join(HERE, "datasets/langid/combined.csv")]
        datasets = [p for p in cand if os.path.exists(p)]
    if not datasets:
        print("ERROR: no datasets. Run prepare_langid_data.py + make_langid_gold.py first.", file=sys.stderr)
        sys.exit(2)

    results_dir = os.path.join(HERE, cfg.get("paths", {}).get("results", "results"))
    os.makedirs(results_dir, exist_ok=True)
    is_llm = spec["type"] == "vllm"
    print(f"Model: {spec['name']} (type={spec['type']}, id={spec.get('model_id')})")

    if spec["type"] == "fasttext":
        predictor = FastTextPredictor(spec["model_id"])
    elif spec["type"] == "langdetect":
        predictor = LangDetectPredictor(spec.get("model_id"))
    elif spec["type"] == "vllm":
        base_url = args.base_url or cfg.get("vllm_base_url")
        if not preflight_vllm(base_url):
            sys.exit(1)
        predictor = VLLMPredictor(spec["model_id"], base_url,
                                  cfg.get("vllm_api_key", "EMPTY"), cfg.get("vllm_gen", {}))
    else:
        print(f"ERROR: eval_langid.py does not handle type '{spec['type']}'.", file=sys.stderr)
        sys.exit(2)

    for csv_path in datasets:
        if not os.path.exists(csv_path):
            print(f"  skip (missing): {csv_path}", file=sys.stderr)
            continue
        eval_dataset(spec, predictor, csv_path, results_dir, args.limit, is_llm)


if __name__ == "__main__":
    main()
