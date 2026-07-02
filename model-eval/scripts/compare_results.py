#!/usr/bin/env python3
"""
compare_results.py — aggregate results/*.summary.json into a comparison report.

- Loads every *.summary.json in results/ (keeps the LATEST run per model+dataset).
- One table per task (ner / sentiment / langid): model | headline | hard-subset |
  avg latency | notes.
- For NER, headline = micro F1 (exact) re-aggregated across a model's datasets from
  tp/n_pred/n_gold counts; hard-subset = same restricted to Roman-Urdu/code-switched
  rows. For sentiment/langid (once those summaries exist) it falls back to
  accuracy / macro-F1, weighted by row count.
- Recommendation section: winner on the HARD subset per task, and flags any
  escalation-path (LLM) model whose hard-subset gain over the best fast/baseline
  model does not justify its latency cost.
- Writes a single markdown file (default results/final_comparison.md).

Usage:
  venv/bin/python scripts/compare_results.py                       # -> results/final_comparison.md
  venv/bin/python scripts/compare_results.py --out results/ner_comparison.md
"""
import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAST_TYPES = {"spacy", "fasttext", "langdetect", "hf_pipeline"}   # resource-efficient path
ESCALATION_TYPES = {"vllm", "cloud"}                              # heavier / slower path
MIN_MEANINGFUL_GAIN = 0.05     # min hard-subset headline gain to justify escalation
TASK_ORDER = ["ner", "sentiment", "langid"]
TASK_TITLES = {"ner": "4.1 Named Entity Recognition (NER)",
               "sentiment": "4.2 Sentiment Analysis",
               "langid": "4.4 Language Identification"}


def f1(tp, n_pred, n_gold):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def load_summaries(results_dir):
    """Return {task: {model: [summary, ...]}}, keeping latest run per model+dataset."""
    latest = {}   # (task, model, dataset) -> summary (highest timestamp)
    for path in glob.glob(os.path.join(results_dir, "*.summary.json")):
        try:
            with open(path, encoding="utf-8") as fh:
                s = json.load(fh)
        except Exception as e:
            print(f"  [skip] {os.path.basename(path)}: {e}")
            continue
        key = (s.get("task"), s.get("model"), s.get("dataset"))
        if key not in latest or s.get("timestamp", "") > latest[key].get("timestamp", ""):
            latest[key] = s
    out = {}
    for s in latest.values():
        out.setdefault(s.get("task"), {}).setdefault(s.get("model"), []).append(s)
    return out


def headline_from_metrics(metrics):
    """Return (value, label, counts_or_None). Handles NER (nested 'exact') and
    flat accuracy/macro_f1 for other tasks."""
    if not metrics:
        return None, None, None
    if isinstance(metrics.get("exact"), dict):        # NER style
        ex = metrics["exact"]
        return ex.get("f1"), "F1 (exact)", (ex.get("tp"), ex.get("n_pred"), ex.get("n_gold"))
    for k, label in (("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"), ("f1", "F1")):
        if k in metrics:
            return metrics[k], label, None
    return None, None, None


def combine(summaries, metrics_key):
    """Combine a metric across a model's dataset-summaries. Micro-average via counts
    when available (NER), else row-weighted mean. Returns (value, label) or (None,None)."""
    tp = npred = ngold = 0
    have_counts = False
    wsum = wtot = 0.0
    label = None
    for s in summaries:
        val, lbl, counts = headline_from_metrics(s.get(metrics_key))
        if val is None:
            continue
        label = label or lbl
        if counts and None not in counts:
            tp += counts[0]; npred += counts[1]; ngold += counts[2]
            have_counts = True
        else:
            n = s.get("n_rows", 0) or 0
            wsum += val * n; wtot += n
    if have_counts:
        return f1(tp, npred, ngold), label
    if wtot:
        return wsum / wtot, label
    return None, None


def combine_partial(summaries, metrics_key):
    """NER-only: micro partial-F1 across datasets (returns None if not present)."""
    tp = npred = ngold = 0
    have = False
    for s in summaries:
        m = s.get(metrics_key) or {}
        part = m.get("partial") if isinstance(m, dict) else None
        if isinstance(part, dict) and part.get("tp") is not None:
            tp += part["tp"]; npred += part["n_pred"]; ngold += part["n_gold"]; have = True
    return f1(tp, npred, ngold) if have else None


def avg_latency(summaries):
    wsum = wtot = 0.0
    for s in summaries:
        m = (s.get("latency_ms") or {}).get("mean")
        n = s.get("n_rows", 0) or 0
        if m is not None and n:
            wsum += m * n; wtot += n
    return (wsum / wtot) if wtot else None


def aggregate_model(model, summaries):
    o_val, o_lbl = combine(summaries, "metrics_overall")
    h_val, h_lbl = combine(summaries, "metrics_hard_subset")
    return {
        "model": model,
        "type": summaries[0].get("model_type", "?"),
        "model_id": summaries[0].get("model_id", ""),
        "overall": o_val, "overall_label": o_lbl or "score",
        "hard": h_val,
        "overall_partial": combine_partial(summaries, "metrics_overall"),
        "hard_partial": combine_partial(summaries, "metrics_hard_subset"),
        "latency_ms": avg_latency(summaries),
        "parse_failures": sum((s.get("parse_failures") or 0) for s in summaries),
        "note": next((s.get("note") for s in summaries if s.get("note")), ""),
        "datasets": sorted({s.get("dataset") for s in summaries}),
        "n_rows": sum(s.get("n_rows", 0) for s in summaries),
    }


def fmt(x, pct=False):
    if x is None:
        return "—"
    return f"{x*100:.1f}%" if pct else f"{x:.3f}"


def fmt_ms(x):
    return "—" if x is None else f"{x:.1f}"


# ------------------------------- rendering ---------------------------------
def render_task(task, models_map):
    rows = [aggregate_model(m, s) for m, s in models_map.items()]
    # winner on hard subset (fall back to overall if no hard metric)
    rows.sort(key=lambda r: (r["hard"] if r["hard"] is not None else -1,
                             r["overall"] if r["overall"] is not None else -1),
              reverse=True)

    md = [f"### {TASK_TITLES.get(task, task)}", ""]
    hl = rows[0]["overall_label"] if rows else "score"
    md.append(f"| Model | Type | {hl} (overall) | Hard-subset | Partial (hard) | Avg latency (ms) | Notes |")
    md.append("|---|---|---|---|---|---|---|")
    for r in rows:
        notes = []
        if r["model_id"]:
            notes.append(f"`{r['model_id']}`")
        if r["parse_failures"]:
            notes.append(f"{r['parse_failures']} parse-fail")
        if r["note"]:
            notes.append(r["note"])
        md.append("| {m} | {t} | {o} | {h} | {hp} | {lat} | {n} |".format(
            m=r["model"], t=r["type"], o=fmt(r["overall"]), h=fmt(r["hard"]),
            hp=fmt(r["hard_partial"]), lat=fmt_ms(r["latency_ms"]),
            n="; ".join(notes) or ""))
    md.append("")
    all_ds = sorted({d for r in rows for d in r["datasets"]})
    hard_desc = {"ner": "Roman-Urdu / code-switched rows",
                 "sentiment": "sarcasm=True rows",
                 "langid": "Roman-Urdu / code-switched rows"}.get(task, "hard rows")
    md.append(f"_Overall = micro-average across {len(all_ds)} dataset(s): "
              + ", ".join(f"`{d}`" for d in all_ds) +
              f". Hard-subset = {hard_desc}._")
    md.append("")
    md.append(recommend(task, rows))
    md.append("")
    return "\n".join(md)


def recommend(task, rows):
    hard_rows = [r for r in rows if r["hard"] is not None]
    if not hard_rows:
        return "_No hard-subset metric available for this task._"
    winner = max(hard_rows, key=lambda r: r["hard"])
    lines = [f"**Recommendation — {task}:** winner on the hard subset is "
             f"**`{winner['model']}`** (hard-subset {fmt(winner['hard'])}, "
             f"{fmt_ms(winner['latency_ms'])} ms/call)."]

    fast = [r for r in hard_rows if r["type"] in FAST_TYPES]
    esc = [r for r in hard_rows if r["type"] in ESCALATION_TYPES]
    if fast and esc:
        bf = max(fast, key=lambda r: r["hard"])
        be = max(esc, key=lambda r: r["hard"])
        gain = be["hard"] - bf["hard"]
        ratio = (be["latency_ms"] / bf["latency_ms"]) if (bf["latency_ms"] and be["latency_ms"]) else None
        rtxt = f"{ratio:.0f}×" if ratio else "much"
        if bf["hard"] >= be["hard"]:
            lines.append(f"⚠️ The fast baseline `{bf['model']}` already matches/beats the "
                         f"escalation model `{be['model']}` on the hard subset "
                         f"({fmt(bf['hard'])} vs {fmt(be['hard'])}) — **escalation not worth it here.**")
        elif gain < MIN_MEANINGFUL_GAIN:
            lines.append(f"⚠️ `{be['model']}` beats `{bf['model']}` by only **+{gain:.3f}** hard-subset "
                         f"for ~**{rtxt}** the latency ({fmt_ms(be['latency_ms'])} vs {fmt_ms(bf['latency_ms'])} ms) "
                         f"— **gain likely doesn't justify the escalation cost;** prefer the fast path unless "
                         f"that +{gain:.3f} matters.")
        else:
            lines.append(f"✅ `{be['model']}` beats the fast baseline `{bf['model']}` by **+{gain:.3f}** "
                         f"hard-subset for ~**{rtxt}** the latency — escalation is justified for the hard subset; "
                         f"use the fast path for easy rows and escalate the rest (see `escalation_thresholds`).")
    return "\n\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "final_comparison.md"))
    args = ap.parse_args()

    data = load_summaries(args.results_dir)
    if not data:
        print(f"No *.summary.json found in {args.results_dir}. Run an eval first.")
        return

    doc = ["# Model Evaluation — Comparison Report",
           "",
           "_Social media monitoring (Pakistani digital media). Hard subset = "
           "Roman Urdu / sarcasm / code-switched rows — the whole point of this eval._",
           ""]
    present = [t for t in TASK_ORDER if t in data] + [t for t in data if t not in TASK_ORDER]
    for task in present:
        doc.append(render_task(task, data[task]))
    missing = [t for t in TASK_ORDER if t not in data]
    if missing:
        doc.append("### Not yet evaluated")
        doc.append("")
        doc.append("No results found for: **" + ", ".join(missing) +
                   "** — run the corresponding eval scripts (Prompts 2 / 3), then re-run this.")
        doc.append("")

    report = "\n".join(doc)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(report)
    print(f"\n[written] {os.path.relpath(args.out, HERE)}")


if __name__ == "__main__":
    main()
