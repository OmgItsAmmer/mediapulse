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

# --- Part 2 (media / clustering / summarization) -----------------------------
# Everything from PART2_MARKER down is (re)generated; content ABOVE it — the
# existing 4.1/4.2/4.4 tables — is preserved (never overwritten).
PART2_MARKER = "## Part 2 —"
PART2_TITLES = {
    "topics": "4.3 Topic Modeling / Clustering",
    "summarization": "4.5 Summarization",
    "stt": "4.6 Speech-to-Text",
    "object_detection": "4.7 Object Detection",
    "face_detection": "4.8 Face Detection & Recognition",
    "scene_detection": "4.9 Scene Detection",
}


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


# =========================== Part 2 rendering ==============================
def load_all_summaries(results_dir):
    """{task: [summary, ...]} — ALL runs (Part-2 tasks pick per-task, since some
    have several runs per model with different configs)."""
    out = {}
    for path in glob.glob(os.path.join(results_dir, "*.summary.json")):
        try:
            with open(path, encoding="utf-8") as fh:
                s = json.load(fh)
        except Exception as e:
            print(f"  [skip] {os.path.basename(path)}: {e}")
            continue
        out.setdefault(s.get("task"), []).append(s)
    return out


def latest_by_model(summaries):
    best = {}
    for s in summaries:
        m = s.get("model")
        if m not in best or s.get("timestamp", "") > best[m].get("timestamp", ""):
            best[m] = s
    return best


def _v(x):
    return "—" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def render_topics(summaries):
    md = [f"### {PART2_TITLES['topics']}", ""]
    if not summaries:
        return "\n".join(md + ["_No topic-modeling results found._", ""])
    rows = sorted(latest_by_model(summaries).values(),
                  key=lambda s: (s.get("metrics", {}).get("ARI") if s.get("metrics") else -1),
                  reverse=True)
    md.append("| Model | ARI | NMI | Fragmentation | Latency (gold fit / pool) |")
    md.append("|---|---|---|---|---|")
    for s in rows:
        me, fr, lat = s.get("metrics", {}), s.get("fragmentation", {}), s.get("latency", {})
        frag = (f"{_v(fr.get('n_fragmented_stories'))}/{_v(fr.get('n_gold_stories'))} stories split, "
                f"{_v(fr.get('n_merged_topics'))} merged; purity {_v(fr.get('cluster_purity'))}")
        latency = (f"{_v(lat.get('gold_fit_transform_s'))}s / {_v(lat.get('pool_fit_transform_s'))}s "
                   f"({_v(lat.get('pool_n_docs'))} docs)")
        md.append(f"| {s['model']} | {_v(me.get('ARI'))} | {_v(me.get('NMI'))} | {frag} | {latency} |")
    md.append("")
    if any(s.get("placeholder_gold") for s in rows):
        md.append("_Gold = PLACEHOLDER (40 hand-authored posts / 6 stories). Ranking is robust; "
                  "absolute ARI/NMI are provisional until a real gold set is built._")
        md.append("")
    best = rows[0]
    md.append(f"**Recommendation — topic modeling:** default **`{best['model']}`** "
              f"(ARI {_v(best.get('metrics', {}).get('ARI'))}). BERTopic (multilingual embeddings) "
              "decisively beats the bag-of-words LDA baseline on short, code-switched text; LDA is "
              "near-random (ARI ~0.08), so the CPU-cheap baseline is **not** a usable fast path here. "
              "BERTopic needs a GPU for embeddings, but topic modeling is a **batch** job off the "
              "live-stream path, so it does not bind the architecture.md §9.3 concurrency ceiling — "
              "schedule it so it doesn't overlap the real-time detectors.")
    return "\n".join(md + [""])


def render_summarization(summaries):
    md = [f"### {PART2_TITLES['summarization']}", ""]
    if not summaries:
        return "\n".join(md + ["_No summarization results found._", ""])
    rows = sorted(latest_by_model(summaries).values(),
                  key=lambda s: (s.get("metrics", {}).get("rougeL_mean") or -1), reverse=True)
    md.append("| Model | Type | ROUGE-L | BERTScore-F1 | Latency (ms/cluster) | Factuality flags |")
    md.append("|---|---|---|---|---|---|")
    for s in rows:
        me, lat = s.get("metrics", {}), s.get("latency_ms", {})
        fl = (s.get("factuality") or {}).get("total_flagged_entities")
        md.append(f"| {s['model']} | {s.get('model_type','?')} | {_v(me.get('rougeL_mean'))} | "
                  f"{_v(me.get('bertscore_f1_mean'))} | {_v(lat.get('mean'))} | {_v(fl)} |")
    md.append("")
    if any(s.get("placeholder_gold") for s in rows):
        md.append("_References = PLACEHOLDER drafts. BERTScore uses a multilingual model. "
                  "Factuality flags on the Qwen run were all false positives on inspection "
                  "(transliteration + common nouns) — no real hallucinations._")
        md.append("")
    best = rows[0]
    md.append(f"**Recommendation — summarization:** default **`{best['model']}`** for quality "
              f"(ROUGE-L {_v(best.get('metrics', {}).get('rougeL_mean'))}, "
              f"BERTScore {_v(best.get('metrics', {}).get('bertscore_f1_mean'))}). The extractive "
              "TextRank baseline (CPU, ~15 ms) is ~2.5× worse on ROUGE-L and returns raw code-switched "
              "posts, so it's only a degraded offline fallback. Qwen3-4B needs vLLM/GPU but is a **batch** "
              "job off the live path (§9.3 ceiling doesn't bind); run it after clustering.")
    return "\n".join(md + [""])


def render_stt(summaries, results_dir):
    md = [f"### {PART2_TITLES['stt']}", ""]
    if not summaries:
        return "\n".join(md + ["_No STT results found._", ""])
    by_model = {}
    for s in summaries:
        by_model.setdefault(s["model"], []).append(s)
    scored, spot = {}, []
    for m, runs in by_model.items():
        wer_runs = [r for r in runs if r.get("wer_overall") is not None]
        if not wer_runs:
            spot += [r for r in runs]
            continue
        best = min(wer_runs, key=lambda r: r["wer_overall"])
        gpu = next((r for r in runs if r.get("device") == "cuda"
                    and (r.get("real_time_factor") or {}).get("mean") is not None), None)
        vram = max([(r.get("vram_mb") or {}).get("model_delta_est") or 0 for r in runs] or [0])
        scored[m] = {"best": best, "gpu": gpu, "vram": vram or None}
    spot += [s for m, runs in by_model.items() for s in runs
             if m not in scored and s.get("wer_overall") is None]

    md.append("| Model | WER | CER | WER by lang (ur/en/mixed) | RTF (GPU) | VRAM (MiB) |")
    md.append("|---|---|---|---|---|---|")
    for m, p in sorted(scored.items(), key=lambda kv: kv[1]["best"]["wer_overall"]):
        b = p["best"]
        bl = b.get("by_language") or {}
        def _wl(t):
            return _v((bl.get(t) or {}).get("wer")) if bl.get(t) else "—"
        rtf = ((p["gpu"] or {}).get("real_time_factor") or {}).get("mean") if p["gpu"] \
            else (b.get("real_time_factor") or {}).get("mean")
        md.append(f"| {m} | {_v(b.get('wer_overall'))} | {_v(b.get('cer_overall'))} | "
                  f"{_wl('urdu')} / {_wl('english')} / {_wl('mixed')} | {_v(rtf)} | {_v(p['vram'])} |")
    md.append("")
    md.append("_WER/CER from the **forced `--language ur`** runs (best config); RTF/VRAM from the "
              "GPU float16 runs. Metrics are on the **FLEURS ur_pk read-speech** baseline — clean, "
              "monolingual Urdu, NOT the broadcast target._")
    md.append("")
    md.append("**BLOCKED — needs:** real Pakistani **broadcast** clips (spontaneous, code-switched, "
              "noisy) with hand transcripts in `gold_sets/stt_gold.csv` for a domain-valid WER. FLEURS "
              "read speech is a floor; broadcast will be harder.")
    md.append("")
    if spot:
        parts = []
        for s in spot:
            rtf = (s.get("real_time_factor") or {}).get("mean")
            parts.append(f"{s['model']} (RTF {_v(rtf)})")
        md.append("_Broadcast spot-check (Geo News 90 s, no reference → no WER): transcribed by "
                  + ", ".join(sorted(set(parts))) + ". large-v3 read it most coherently; auto-detect "
                  "correctly chose Urdu on this clip (the Hindi mis-ID was a read-speech artifact)._")
        md.append("")
    if scored:
        best_m = min(scored, key=lambda k: scored[k]["best"]["wer_overall"])
        fast_m = max(scored, key=lambda k: scored[k]["best"]["wer_overall"])  # cheapest usable
        md.append(
            f"**Recommendation — STT:** **force / route the language** (auto-detect flips Urdu→Hindi, "
            f"which wrecked WER >100%). Resource-efficient default **`fw_small`/`fw_medium`** (forced ur, "
            f"tiny VRAM, RTF ≪1); escalate to **`{best_m}`** (best WER "
            f"{_v(scored[best_m]['best'].get('wer_overall'))}) for accuracy-critical streams. STT is on "
            f"the **live-stream path**, so it competes for the single 24 GB GPU with the object/face "
            f"detectors and vLLM — cross-check architecture.md §9.3: `fw_large_v3` (~4 GB, GPU RTF ~0.13) "
            f"leaves headroom for only a few concurrent streams once detectors are co-resident; "
            f"`fw_small`/`fw_medium` (~1–2 GB) allow more. For genuinely `mixed` clips a single forced "
            f"language may hurt English segments — still open.")
        md.append("")
    return "\n".join(md)


def render_blocked(task, needs):
    return "\n".join([f"### {PART2_TITLES[task]}", "",
                      f"**BLOCKED — needs:** {needs}", "",
                      "_(Owned by a team member; not evaluated here. No fake result row.)_", "",
                      f"**Recommendation — {task.replace('_',' ')}:** pending data. Note this is on the "
                      "**live-stream path** — architecture.md §9.3's GPU concurrency ceiling will gate the "
                      "choice: heavier variants (e.g. YOLOv8x / InsightFace `buffalo_l`) cut the max "
                      "concurrent streams per GPU, so the winning model must be checked against the "
                      "real-time budget, not just accuracy.", ""])


def render_scenes(summaries, results_dir):
    md = [f"### {PART2_TITLES['scene_detection']}", "",
          "_Single candidate (PySceneDetect) — a sanity check, not a comparison; no accuracy metric "
          "without hand-labelled cuts._", ""]
    scene_files = sorted(glob.glob(os.path.join(results_dir, "scenes_*.json")))
    if scene_files:
        try:
            d = json.load(open(scene_files[-1], encoding="utf-8"))
            md.append(f"- Last run: `{d.get('clip')}` → **{d.get('n_cuts')} cuts** / "
                      f"{d.get('n_scenes')} scenes, RTF **{d.get('real_time_factor')}** "
                      f"(detector={d.get('detector')}, threshold={d.get('threshold')}).")
        except Exception:
            pass
    md.append("- Harness verified on a synthetic clip (hard cuts at 3/6/9/12 s → detected exactly). "
              "Needs a real 1–3 min broadcast clip in `datasets/scenes/` for the actual spot-check.")
    md.append("")
    md.append("**Recommendation — scene detection:** keep **PySceneDetect** (content detector). It's "
              "CPU-only with a negligible RTF (~0.002), so it runs comfortably alongside the GPU "
              "detectors and does **not** touch the architecture.md §9.3 GPU ceiling. Tune `--threshold` "
              "once real footage exists.")
    return "\n".join(md + [""])


def build_part2(all_data, results_dir):
    md = ["## Part 2 — Media, Clustering & Summarization (4.3, 4.5–4.9)", "",
          "_Appended after the 4.1/4.2/4.4 tables. Topic-modeling and summarization gold sets are "
          "PLACEHOLDER (rankings robust, absolute numbers provisional); STT is on a read-speech "
          "baseline; object/face detection are blocked on data._", ""]
    md.append(render_topics(all_data.get("topics", [])))
    md.append(render_summarization(all_data.get("summarization", [])))
    md.append(render_stt(all_data.get("stt", []), results_dir))
    md.append(render_blocked("object_detection",
              "labelled Pakistani broadcast frames (`gold_sets/objects_gold.csv`: frame_path + boxes) "
              "and the YOLO/Detectron2 runs. COCO smoke-test is a generic check, not broadcast/logos."))
    md.append(render_blocked("face_detection",
              "a WIDER FACE detection subset AND an enrollment gallery + gold test frames "
              "(`gold_sets/faces_gold.csv`, `datasets/faces/gallery/<id>/`). Biometric — legal review "
              "per architecture.md §9.1 before any production use."))
    md.append(render_scenes(all_data.get("scene_detection", []), results_dir))
    md.append("### Live-stream GPU budget (architecture.md §9.3)")
    md.append("")
    md.append("Only **STT, object detection, and face detection** run on the live path and share the "
              "single 24 GB RTX 3090. Measured footprints so far: `fw_large_v3` ~4 GB / GPU-RTF ~0.13, "
              "`fw_medium` ~2.2 GB, `fw_small` ~1 GB. Topic modeling, summarization and scene detection "
              "are batch/CPU and don't bind the ceiling. **Flag:** stacking `fw_large_v3` STT with heavy "
              "object + face detectors (and any vLLM) on one 3090 will blow the concurrency ceiling — "
              "prefer smaller STT, or dedicate/scale GPUs, before committing the live path. Re-check "
              "against §9.3's exact per-GPU stream limit once object/face numbers exist.")
    md.append("")
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "final_comparison.md"))
    args = ap.parse_args()

    text_data = load_summaries(args.results_dir)
    all_data = load_all_summaries(args.results_dir)
    if not all_data:
        print(f"No *.summary.json found in {args.results_dir}. Run an eval first.")
        return

    part2 = build_part2(all_data, args.results_dir)

    if os.path.exists(args.out) and os.path.getsize(args.out) > 0:
        # PRESERVE existing 4.1/4.2/4.4 content; (re)generate only Part 2 (idempotent).
        existing = open(args.out, encoding="utf-8").read()
        idx = existing.find(PART2_MARKER)
        head = (existing[:idx] if idx != -1 else existing).rstrip()
        report = head + "\n\n" + part2
    else:
        # No prior report — build the text-task sections too, then Part 2.
        doc = ["# Model Evaluation — Comparison Report", "",
               "_Social media monitoring (Pakistani digital media). Hard subset = "
               "Roman Urdu / sarcasm / code-switched rows — the whole point of this eval._", ""]
        present = ([t for t in TASK_ORDER if t in text_data]
                   + [t for t in text_data if t not in TASK_ORDER])
        for task in present:
            doc.append(render_task(task, text_data[task]))
        report = "\n".join(doc) + "\n" + part2

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(part2)
    print(f"\n[written/appended Part 2 -> {os.path.relpath(args.out, HERE)}]")


if __name__ == "__main__":
    main()
