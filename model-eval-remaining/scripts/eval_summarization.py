#!/usr/bin/env python3
"""
eval_summarization.py — evaluate ONE summarization model (Prompt 6, Task 3).

Models (config.yaml, models.summarization):
  - textrank : extractive TextRank baseline via `sumy` (CPU). gensim's summarize
               module was removed in gensim 4.0, so sumy is the maintained choice.
  - vllm     : abstractive summary via a vLLM chat completion (Qwen3-4B), using the
               PROMPT below. vLLM must be served first (serve_and_eval.sh).

Input : datasets/summarization/clusters.csv  (one concatenated blob per story cluster)
Refs  : gold_sets/summarization_gold.csv      (one PLACEHOLDER reference per cluster)

Scoring:
  - ROUGE-L (rouge_score, F-measure)      — lexical overlap with the reference
  - BERTScore (multilingual model, F1)    — semantic similarity; multilingual since
      the source posts are code-switched even though the summaries are English
  - Factuality sanity check               — flags named entities that appear in the
      SUMMARY but NOT in the source cluster text (possible hallucination). Uses a
      simple capitalized-token heuristic, augmented with the 4.1 NER-gold entity
      gazetteer (gold_sets/ner_gold.csv) when available; if absent, that augmentation
      is skipped and noted.

Runs in CONDA BASE (BERTScore needs torch + transformers; openai for the vLLM path):
  conda run -n base python scripts/eval_summarization.py --model textrank_sumy
  # vLLM path (serve Qwen3-4B first; conda python so BERTScore has torch):
  VENV_PY=$(command -v python) ./scripts/serve_and_eval.sh \
      Qwen/Qwen3-4B-Instruct-2507 eval_summarization.py
Install (into base):  pip install sumy rouge-score bert-score nltk

Outputs: results/summarization_<model>_<ts>.csv + .summary.json
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
SENTENCES_COUNT = 3        # TextRank: how many sentences to extract

# =============================== PROMPT (vLLM) ================================
# The exact instruction requested in Prompt 6, Task 3.
SYSTEM_PROMPT = (
    "You are a news analyst for a Pakistani social-media monitoring platform. "
    "You write concise, factual briefs from clusters of related social-media posts."
)
USER_TEMPLATE = (
    "Summarize this cluster of social media posts in 2-3 sentences, in English, "
    "capturing the main story even if posts are in Roman Urdu or mixed language. "
    "Only use information present in the posts; do not invent names, numbers, or events.\n\n"
    "Posts:\n{posts}"
)


def build_messages(posts_text):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(posts=posts_text)},
    ]


def sanitize(s):
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)


# =============================== summarizers =================================
class TextRankSummarizer:
    type = "textrank"

    def __init__(self, spec):
        try:
            import nltk  # noqa: F401
            from sumy.summarizers.text_rank import TextRankSummarizer as _TR  # noqa: F401
        except ImportError:
            sys.exit("ERROR: sumy/nltk not installed. Run: pip install sumy nltk")
        self._ensure_nltk()
        self.spec = spec

    @staticmethod
    def _ensure_nltk():
        import nltk
        for pkg in ("punkt", "punkt_tab"):
            try:
                nltk.data.find(f"tokenizers/{pkg}")
            except LookupError:
                try:
                    nltk.download(pkg, quiet=True)
                except Exception:
                    pass

    def summarize(self, posts_text):
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.summarizers.text_rank import TextRankSummarizer as _TR
        t0 = time.perf_counter()
        # give TextRank discrete sentences: one post per line, period-terminated
        sents = [p.strip().rstrip(".") + "." for p in posts_text.split("\n") if p.strip()]
        doc = "\n".join(sents)
        parser = PlaintextParser.from_string(doc, Tokenizer("english"))
        n = min(SENTENCES_COUNT, max(1, len(sents)))
        picked = _TR()(parser.document, n)
        summary = " ".join(str(s) for s in picked).strip()
        if not summary:                       # fallback: first sentence(s)
            summary = " ".join(sents[:n])
        return summary, (time.perf_counter() - t0) * 1000.0, True, ""


class VLLMSummarizer:
    type = "vllm"

    def __init__(self, spec, base_url, api_key, gen, api_model=None):
        from openai import OpenAI
        self.spec = spec
        # api_model overrides the string sent as `model` (for --served-model-name aliases);
        # spec["model_id"] stays as the real model for labeling the results.
        self.model_id = api_model or spec["model_id"]
        self.gen = gen
        self.client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY",
                             timeout=gen.get("request_timeout_s", 60))

    def summarize(self, posts_text):
        t0 = time.perf_counter()
        try:
            resp = self.client.chat.completions.create(
                model=self.model_id, messages=build_messages(posts_text),
                temperature=self.gen.get("temperature", 0.0),
                max_tokens=self.gen.get("max_tokens", 256))
            summary = (resp.choices[0].message.content or "").strip()
        except Exception as ex:
            return None, (time.perf_counter() - t0) * 1000.0, False, f"API_ERROR: {ex}"
        return summary, (time.perf_counter() - t0) * 1000.0, bool(summary), ""


# =============================== scoring =====================================
def rouge_l_scores(cands, refs):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return [round(scorer.score(r, c)["rougeL"].fmeasure, 4) for c, r in zip(cands, refs)]


def bertscore_f1(cands, refs, model_type, device):
    """Returns per-pair F1 (list of floats) + the model actually used."""
    from bert_score import score as bs_score
    kwargs = dict(cands=cands, refs=refs, model_type=model_type, verbose=False,
                  device=device, rescale_with_baseline=False)
    _, _, f1 = bs_score(**kwargs)
    return [round(float(x), 4) for x in f1.tolist()], model_type


# =============================== factuality ==================================
# Words that are Capitalized but not really named entities (reduce false positives).
_STOP_CAPS = {
    "The", "A", "An", "This", "That", "These", "Those", "It", "They", "He", "She",
    "We", "I", "In", "On", "At", "For", "And", "But", "Or", "Pakistani", "English",
    "Urdu", "Roman", "Posts", "Summarize", "Summary",
}
_PROPER = re.compile(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b")


def load_ner_gazetteer(path):
    """Distinct entity surface strings from the 4.1 NER gold set, if present."""
    if not os.path.exists(path):
        return None
    gaz = set()
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                for e in json.loads(r.get("entities") or "[]"):
                    t = (e.get("text") or "").strip()
                    if len(t) >= 3:
                        gaz.add(t)
            except (json.JSONDecodeError, AttributeError):
                continue
    return gaz or None


def extract_summary_entities(summary, gazetteer):
    ents = set()
    for m in _PROPER.finditer(summary):
        cand = m.group(1).strip()
        # drop single tokens that are just a capitalized common/stop word
        if " " not in cand and cand in _STOP_CAPS:
            continue
        ents.add(cand)
    if gazetteer:
        low = summary.lower()
        for g in gazetteer:
            if g.lower() in low:
                ents.add(g)
    return ents


def factuality_flags(summary, source_text, gazetteer):
    """Entities mentioned in the summary but absent from the source cluster."""
    src = source_text.lower()
    ents = extract_summary_entities(summary, gazetteer)
    flagged = sorted(e for e in ents if e.lower() not in src)
    return flagged


# =============================== io / config =================================
def load_config(path):
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_model(cfg, model_arg):
    for spec in cfg.get("models", {}).get("summarization", []):
        if model_arg in (spec.get("name"), spec.get("model_id")):
            return dict(spec)
    if model_arg and "/" in model_arg:       # bare HF id -> treat as vllm
        return {"name": model_arg.split("/")[-1], "type": "vllm", "model_id": model_arg}
    return None


def load_clusters(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    return rows


def load_refs(path):
    refs = {}
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            refs[str(r.get("story_id"))] = r.get("reference_summary", "")
    return refs


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


# =============================== main ========================================
def main():
    ap = argparse.ArgumentParser(description="Evaluate one summarization model.")
    ap.add_argument("--model", default=os.environ.get("EVAL_MODEL"),
                    help="config name: textrank_sumy | qwen3_4b_instruct")
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--clusters", default=os.path.join(HERE, "datasets/summarization/clusters.csv"))
    ap.add_argument("--gold", default=os.path.join(HERE, "gold_sets/summarization_gold.csv"))
    ap.add_argument("--ner-gold", default=os.path.join(HERE, "gold_sets/ner_gold.csv"))
    ap.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL"))
    ap.add_argument("--api-model", default=os.environ.get("VLLM_API_MODEL"),
                    help="override the `model` string sent to vLLM (for --served-model-name aliases); "
                         "the result label stays the real config model_id.")
    ap.add_argument("--bertscore-model", default="xlm-roberta-large",
                    help="multilingual BERTScore model (e.g. bert-base-multilingual-cased for a lighter one).")
    ap.add_argument("--bertscore-device", default=None, help="cuda | cpu (default: auto).")
    ap.add_argument("--no-bertscore", action="store_true", help="skip BERTScore (ROUGE-L only).")
    ap.add_argument("--limit", type=int, default=0, help="cap clusters (0=all).")
    args = ap.parse_args()

    if not args.model:
        sys.exit("ERROR: no model. Use --model textrank_sumy|qwen3_4b_instruct or set $EVAL_MODEL.")
    cfg = load_config(args.config)
    spec = resolve_model(cfg, args.model)
    if not spec:
        avail = [s["name"] for s in cfg.get("models", {}).get("summarization", [])]
        sys.exit(f"ERROR: model '{args.model}' not in config.summarization. Available: {avail}")

    for p, what, fixer in [(args.clusters, "clusters", "prepare_summarization_data.py"),
                           (args.gold, "gold references", "make_summarization_gold.py")]:
        if not os.path.exists(p):
            sys.exit(f"ERROR: {what} missing: {p}. Run scripts/{fixer} first.")

    clusters = load_clusters(args.clusters)
    refs = load_refs(args.gold)
    if args.limit:
        clusters = clusters[:args.limit]
    missing = [c["story_id"] for c in clusters if str(c["story_id"]) not in refs]
    if missing:
        print(f"  WARNING: no reference summary for story_id(s) {missing} — they are skipped.",
              file=sys.stderr)
        clusters = [c for c in clusters if str(c["story_id"]) in refs]
    if not clusters:
        sys.exit("ERROR: no clusters with a matching reference summary.")

    gazetteer = load_ner_gazetteer(args.ner_gold)
    gaz_note = (f"NER-gold gazetteer: {len(gazetteer)} entities"
                if gazetteer else
                "NER-gold gazetteer: NOT FOUND — factuality uses the capitalized-token "
                "heuristic only (no gazetteer augmentation).")

    results_dir = os.path.join(HERE, cfg.get("paths", {}).get("results", "results"))
    os.makedirs(results_dir, exist_ok=True)

    print(f"Model: {spec['name']} (type={spec['type']}, id={spec.get('model_id')})")
    print(f"Clusters: {len(clusters)}  |  {gaz_note}")

    # ---- build the summarizer ------------------------------------------------
    if spec["type"] == "textrank":
        summarizer = TextRankSummarizer(spec)
    elif spec["type"] == "vllm":
        base_url = args.base_url or cfg.get("vllm_base_url")
        if not preflight_vllm(base_url):
            sys.exit(1)
        summarizer = VLLMSummarizer(spec, base_url, cfg.get("vllm_api_key", "EMPTY"),
                                    cfg.get("vllm_gen", {}), api_model=args.api_model)
        if args.api_model:
            print(f"  (calling served alias '{args.api_model}' at {base_url}; "
                  f"labeling results as {spec.get('model_id')})")
    else:
        sys.exit(f"ERROR: eval_summarization.py does not handle type '{spec['type']}'.")

    # ---- generate summaries --------------------------------------------------
    per_cluster = []
    for c in clusters:
        summ, latency, ok, err = summarizer.summarize(c["text"])
        if not ok:
            print(f"  [story {c['story_id']}] generation failed: {err}", file=sys.stderr)
            summ = summ or ""
        flagged = factuality_flags(summ, c["text"], gazetteer)
        per_cluster.append({
            "story_id": c["story_id"], "story_label": c.get("story_label", ""),
            "reference": refs[str(c["story_id"])], "summary": summ,
            "latency_ms": round(latency, 1), "ok": ok, "error": err,
            "flagged_entities": flagged,
        })

    # ---- score ---------------------------------------------------------------
    cands = [p["summary"] for p in per_cluster]
    golds = [p["reference"] for p in per_cluster]
    rouge = rouge_l_scores(cands, golds)
    for p, rl in zip(per_cluster, rouge):
        p["rougeL"] = rl

    bert_model_used = None
    if args.no_bertscore:
        for p in per_cluster:
            p["bertscore_f1"] = None
    else:
        device = args.bertscore_device
        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        print(f"  scoring BERTScore ({args.bertscore_model}, device={device}) ...")
        bs, bert_model_used = bertscore_f1(cands, golds, args.bertscore_model, device)
        for p, f in zip(per_cluster, bs):
            p["bertscore_f1"] = f

    # ---- write outputs -------------------------------------------------------
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"summarization_{sanitize(spec['name'])}_{ts}"
    per_row_path = os.path.join(results_dir, base + ".csv")
    summary_path = os.path.join(results_dir, base + ".summary.json")

    with open(per_row_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["story_id", "story_label", "rougeL", "bertscore_f1",
                    "n_flagged_entities", "flagged_entities", "latency_ms",
                    "summary", "reference"])
        for p in per_cluster:
            w.writerow([p["story_id"], p["story_label"], p["rougeL"], p.get("bertscore_f1"),
                        len(p["flagged_entities"]), "; ".join(p["flagged_entities"]),
                        p["latency_ms"], p["summary"], p["reference"]])

    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return round(statistics.mean(vals), 4) if vals else None

    lats = [p["latency_ms"] for p in per_cluster]
    total_flagged = sum(len(p["flagged_entities"]) for p in per_cluster)
    summary = {
        "task": "summarization",
        "model": spec["name"], "model_type": spec["type"], "model_id": spec.get("model_id"),
        "api_model_served": args.api_model if spec["type"] == "vllm" else None,
        "clusters": os.path.relpath(args.clusters, HERE),
        "gold": os.path.relpath(args.gold, HERE),
        "n_clusters": len(per_cluster),
        "prompt_template": USER_TEMPLATE if spec["type"] == "vllm" else None,
        "metrics": {
            "rougeL_mean": _mean([p["rougeL"] for p in per_cluster]),
            "bertscore_f1_mean": _mean([p.get("bertscore_f1") for p in per_cluster]),
            "bertscore_model": bert_model_used,
        },
        "factuality": {
            "gazetteer": gaz_note,
            "total_flagged_entities": total_flagged,
            "clusters_with_flags": sum(1 for p in per_cluster if p["flagged_entities"]),
            "per_cluster": {p["story_id"]: p["flagged_entities"] for p in per_cluster},
            "note": "Flagged = named entity in the summary but not in the source cluster "
                    "(possible hallucination). Heuristic — verify flags by hand.",
        },
        "latency_ms": {
            "mean": round(statistics.mean(lats), 1) if lats else None,
            "median": round(statistics.median(lats), 1) if lats else None,
            "total_s": round(sum(lats) / 1000.0, 1) if lats else None,
        },
        "note": spec.get("note", ""),
        "placeholder_gold": "summarization_gold.csv references are PLACEHOLDER drafts — "
                            "review/rewrite before trusting ROUGE/BERTScore.",
        "timestamp": ts,
        "per_row_csv": os.path.relpath(per_row_path, HERE),
    }
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    # ---- console summary -----------------------------------------------------
    m = summary["metrics"]
    print(f"\n  [{spec['name']}]  ROUGE-L={m['rougeL_mean']}  "
          f"BERTScore-F1={m['bertscore_f1_mean']}  ({len(per_cluster)} clusters)")
    print(f"    factuality: {total_flagged} flagged entities across "
          f"{summary['factuality']['clusters_with_flags']} clusters")
    for p in per_cluster:
        flag = f"  ⚠ {p['flagged_entities']}" if p["flagged_entities"] else ""
        print(f"      story {p['story_id']}: ROUGE-L={p['rougeL']} "
              f"BERTScore={p.get('bertscore_f1')}{flag}")
    print(f"    latency mean={summary['latency_ms']['mean']} ms/cluster")
    print(f"    -> {os.path.basename(summary_path)}")


if __name__ == "__main__":
    main()
