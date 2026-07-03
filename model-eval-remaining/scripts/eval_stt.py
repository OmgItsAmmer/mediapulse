#!/usr/bin/env python3
"""
eval_stt.py — evaluate ONE speech-to-text model on the gold audio (Prompt 7, Task 3).

*** BLOCKED until gold audio exists. *** This script + the gold template are PREP,
not a finished eval. Populate gold_sets/stt_gold.csv + datasets/stt/audio/ first
(see datasets/stt/README.md). With an empty gold set the script prints a clear
BLOCKED message and exits without inventing anything.

Models (config.yaml, models.stt):
  - faster_whisper : Faster-Whisper small / medium / large-v3 (CTranslate2). GPU.
  - nemo           : NVIDIA NeMo Parakeet/Canary (OPTIONAL). Imported lazily; if
                     nemo_toolkit isn't installed / fails to import, the model is
                     skipped with a clear flag (single-GPU boxes often can't build it).

For each gold clip it records:
  - WER (jiwer) and CER, on normalised text (lowercased, punctuation incl. Urdu stripped)
  - real-time factor = processing_time / audio_duration  (the live-stream latency signal)
  - the model's detected language (Whisper) for a code-switch sanity read
WER is aggregated OVERALL and split by language_tag {urdu, english, mixed}
(corpus WER = total errors / total words, not a mean of per-clip WERs).
Peak VRAM for THIS process is snapshotted from nvidia-smi (per-PID, so it's valid
even while the GPU is shared).

Runtime: conda env STTLiveTransciptionVoxtralEnv (faster-whisper + soundfile present).
  conda run -n STTLiveTransciptionVoxtralEnv pip install jiwer      # one-time
  conda run -n STTLiveTransciptionVoxtralEnv python scripts/eval_stt.py --model fw_small

Outputs: results/stt_<model>_<ts>.csv + .summary.json
"""
import argparse
import csv
import datetime as dt
import json
import os
import re
import statistics
import subprocess
import sys
import time

import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANG_TAGS = ["urdu", "english", "mixed"]

# Normalisation for WER/CER: lowercase (Latin), strip Urdu diacritics + fold
# Arabic-form codepoints to their Urdu equivalents, drop punctuation incl. Urdu
# marks, collapse whitespace. \w with re.UNICODE keeps Urdu letters + digits.
_NONWORD = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")
# Arabic/Urdu diacritics (harakat) + tatweel/kashida. FLEURS refs are diacritised,
# Whisper output is not — without stripping these, every mark counts as an error.
_URDU_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭـ]")
_ZERO_WIDTH = re.compile(r"[​-\u200F\u202A-\u202E﻿]")
# Fold Arabic-form letters to Urdu forms (a common Whisper-vs-FLEURS codepoint mismatch).
_URDU_FOLD = str.maketrans({
    "ي": "ی",  # Arabic yeh   ي -> Urdu yeh   ی
    "ى": "ی",  # alef maksura ى -> Urdu yeh   ی
    "ك": "ک",  # Arabic kaf   ك -> Urdu keheh ک
    "ه": "ہ",  # Arabic heh   ه -> Urdu heh   ہ
    "ة": "ہ",  # teh marbuta  ة -> Urdu heh   ہ
    "أ": "ا",  # alef+hamza above أ -> alef ا
    "إ": "ا",  # alef+hamza below إ -> alef ا
    "ٱ": "ا",  # alef wasla   ٱ -> alef ا
})


def normalize(s):
    """Lowercase, strip Urdu diacritics/zero-width marks, fold Arabic->Urdu forms,
    drop punctuation (incl. Urdu ۔،؟), collapse whitespace. Harmless for English."""
    s = (s or "").strip().lower()
    s = _URDU_DIACRITICS.sub("", s)
    s = _ZERO_WIDTH.sub("", s)
    s = s.translate(_URDU_FOLD)
    s = _NONWORD.sub(" ", s)
    return _WS.sub(" ", s).strip()


def sanitize(s):
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)


# ============================ VRAM (per-PID) ===============================
def gpu_used_mb_for_pid(pid):
    """MiB attributed to this PID by nvidia-smi (valid even on a shared GPU)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"], text=True, timeout=5)
    except Exception:
        return None
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0] == str(pid):
            try:
                return int(float(parts[1]))
            except ValueError:
                return None
    return None


# ============================ transcribers =================================
class FasterWhisper:
    type = "faster_whisper"

    def __init__(self, spec, device, compute_type, beam_size, language):
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            sys.exit("ERROR: faster-whisper not installed. Use the "
                     "STTLiveTransciptionVoxtralEnv env (has it), or: pip install faster-whisper")
        self.beam_size = beam_size
        self.language = language        # None = auto-detect (best for mixed)
        print(f"  loading Faster-Whisper '{spec['model_id']}' (device={device}, "
              f"compute_type={compute_type}) ...")
        self.model = WhisperModel(spec["model_id"], device=device, compute_type=compute_type)

    def transcribe(self, audio_path):
        t0 = time.perf_counter()
        segments, info = self.model.transcribe(
            audio_path, beam_size=self.beam_size, language=self.language)
        text = " ".join(seg.text for seg in segments).strip()   # generator -> compute here
        proc_s = time.perf_counter() - t0
        return text, info.duration, getattr(info, "language", None), proc_s, ""


class NeMoASR:
    """OPTIONAL NVIDIA NeMo (Parakeet/Canary). Skipped cleanly if nemo won't import."""
    type = "nemo"

    def __init__(self, spec, device, *_):
        try:
            import nemo.collections.asr as nemo_asr
        except Exception as ex:      # ImportError or a build/CUDA error
            raise RuntimeError(f"NeMo unavailable ({type(ex).__name__}: {ex}). "
                               "Install nemo_toolkit[asr] or leave this model commented out.")
        print(f"  loading NeMo '{spec['model_id']}' ...")
        self.model = nemo_asr.models.ASRModel.from_pretrained(spec["model_id"])
        try:
            import soundfile  # noqa: F401
        except ImportError:
            sys.exit("ERROR: soundfile needed for NeMo duration; pip install soundfile")

    def transcribe(self, audio_path):
        import soundfile as sf
        info = sf.info(audio_path)
        duration = info.frames / float(info.samplerate)
        t0 = time.perf_counter()
        out = self.model.transcribe([audio_path])
        proc_s = time.perf_counter() - t0
        text = out[0] if out else ""
        if not isinstance(text, str):                 # some NeMo versions return objects
            text = getattr(text, "text", str(text))
        return text.strip(), duration, None, proc_s, ""


# ============================ scoring ======================================
def corpus_wer_cer(refs, hyps):
    import jiwer
    refs_n = [normalize(r) for r in refs]
    hyps_n = [normalize(h) for h in hyps]
    # guard against all-empty refs (jiwer divides by ref word count)
    if not any(refs_n):
        return None, None
    wer = round(float(jiwer.wer(refs_n, hyps_n)), 4)
    cer = round(float(jiwer.cer(refs_n, hyps_n)), 4)
    return wer, cer


# ============================ config / io ==================================
def load_config(path):
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_model(cfg, model_arg):
    for spec in cfg.get("models", {}).get("stt", []):
        if model_arg in (spec.get("name"), spec.get("model_id")):
            return dict(spec)
    return None


def load_gold(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            ap = (r.get("audio_path") or "").strip()
            ref = (r.get("reference_transcript") or "").strip()
            tag = (r.get("language_tag") or "").strip().lower()
            if ap and ref:
                rows.append({"audio_path": ap, "reference": ref,
                             "language_tag": tag if tag in LANG_TAGS else "unknown"})
    return rows


# ============================ main =========================================
def main():
    ap = argparse.ArgumentParser(description="Evaluate one speech-to-text model.")
    ap.add_argument("--model", default=os.environ.get("EVAL_MODEL"),
                    help="config name: fw_small | fw_medium | fw_large_v3 | ...")
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--gold", default=os.path.join(HERE, "gold_sets/stt_gold.csv"))
    ap.add_argument("--audio-root", default=HERE,
                    help="base dir for relative audio_path values (default: project root).")
    ap.add_argument("--device", default="cuda", help="cuda | cpu")
    ap.add_argument("--compute-type", default=None,
                    help="faster-whisper compute type (default float16 on cuda, int8 on cpu).")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--language", default=None,
                    help="force a language code (e.g. ur, en). Default: auto-detect (best for mixed).")
    ap.add_argument("--limit", type=int, default=0, help="cap clips (0=all).")
    ap.add_argument("--audio", nargs="*", default=None,
                    help="transcribe these audio files/dirs with NO reference (qualitative "
                         "spot-check): outputs hypotheses + RTF, WER/CER = N/A. Bypasses the gold set.")
    args = ap.parse_args()

    if not args.model:
        sys.exit("ERROR: no model. Use --model fw_small|fw_medium|fw_large_v3 or set $EVAL_MODEL.")
    cfg = load_config(args.config)
    spec = resolve_model(cfg, args.model)
    if not spec:
        avail = [s["name"] for s in cfg.get("models", {}).get("stt", [])]
        sys.exit(f"ERROR: model '{args.model}' not in config.stt. Available: {avail}")

    # ---- clips: either --audio spot-check (no reference) or the gold set ------
    _AUDIO_EXT = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4", ".aac")
    no_reference = bool(args.audio)
    if no_reference:
        clips = []
        for p in args.audio:
            ap = p if os.path.isabs(p) else os.path.join(args.audio_root, p)
            if os.path.isdir(ap):
                clips += [os.path.join(ap, fn) for fn in sorted(os.listdir(ap))
                          if fn.lower().endswith(_AUDIO_EXT)]
            elif os.path.exists(ap):
                clips.append(ap)
            else:
                print(f"  WARNING: audio not found: {p}", file=sys.stderr)
        if not clips:
            sys.exit("ERROR: --audio given but no audio files found.")
        gold = [{"audio_path": os.path.relpath(c, HERE), "reference": "",
                 "language_tag": "unknown", "abs_path": c} for c in clips]
        print("  MODE: transcription spot-check (no reference) — WER/CER = N/A, "
              "eyeball the hypotheses.")
    else:
        # ---- BLOCKED guard: no gold audio yet --------------------------------
        if not os.path.exists(args.gold):
            print("BLOCKED — needs: gold audio. gold_sets/stt_gold.csv does not exist.\n"
                  "  Run scripts/make_stt_gold.py, then follow datasets/stt/README.md.",
                  file=sys.stderr)
            sys.exit(3)
        gold = load_gold(args.gold)
        if args.limit:
            gold = gold[:args.limit]
        if not gold:
            print("BLOCKED — needs: gold audio + transcripts.\n"
                  "  gold_sets/stt_gold.csv has no usable rows (audio_path + reference_transcript).\n"
                  "  Drop 10-20 clips in datasets/stt/audio/ and fill the CSV "
                  "(see datasets/stt/README.md). Or use --audio <clip> for a no-reference spot-check.",
                  file=sys.stderr)
            sys.exit(3)
        for g in gold:
            g["abs_path"] = g["audio_path"] if os.path.isabs(g["audio_path"]) \
                else os.path.join(args.audio_root, g["audio_path"])
        missing = [g["audio_path"] for g in gold if not os.path.exists(g["abs_path"])]
        if missing:
            print(f"  WARNING: {len(missing)} audio file(s) listed in gold but not found; skipping: "
                  f"{missing[:5]}{' ...' if len(missing) > 5 else ''}", file=sys.stderr)
            gold = [g for g in gold if os.path.exists(g["abs_path"])]
        if not gold:
            sys.exit("BLOCKED — needs: gold audio. Rows exist but no audio files were found on disk.")
    if args.limit:
        gold = gold[:args.limit]

    compute_type = args.compute_type or ("float16" if args.device == "cuda" else "int8")
    results_dir = os.path.join(HERE, cfg.get("paths", {}).get("results", "results"))
    os.makedirs(results_dir, exist_ok=True)
    pid = os.getpid()
    vram_before = gpu_used_mb_for_pid(pid) or 0

    print(f"Model: {spec['name']} (type={spec['type']}, id={spec.get('model_id')})")
    print(f"Gold clips: {len(gold)}  |  by tag: "
          + ", ".join(f"{t}={sum(1 for g in gold if g['language_tag']==t)}" for t in LANG_TAGS))

    # ---- build the transcriber -----------------------------------------------
    if spec["type"] == "faster_whisper":
        tx = FasterWhisper(spec, args.device, compute_type, args.beam_size, args.language)
    elif spec["type"] == "nemo":
        try:
            tx = NeMoASR(spec, args.device)
        except RuntimeError as ex:
            print(f"SKIPPED (optional): {spec['name']} — {ex}", file=sys.stderr)
            sys.exit(4)
    else:
        sys.exit(f"ERROR: eval_stt.py does not handle type '{spec['type']}'.")

    # ---- transcribe every clip -----------------------------------------------
    per_clip = []
    vram_peak = vram_before
    for g in gold:
        try:
            hyp, duration, det_lang, proc_s, err = tx.transcribe(g["abs_path"])
        except Exception as ex:
            print(f"  [{g['audio_path']}] transcription failed: {ex}", file=sys.stderr)
            hyp, duration, det_lang, proc_s, err = "", None, None, None, str(ex)
        rtf = round(proc_s / duration, 3) if (proc_s and duration) else None
        vram_peak = max(vram_peak, gpu_used_mb_for_pid(pid) or vram_peak)
        per_clip.append({**g, "hyp": hyp, "duration_s": round(duration, 2) if duration else None,
                         "detected_language": det_lang, "proc_s": round(proc_s, 3) if proc_s else None,
                         "rtf": rtf, "error": err})

    # ---- WER/CER overall + by language ---------------------------------------
    def bucket(tag):
        rows = [c for c in per_clip if c["language_tag"] == tag and c["hyp"] is not None]
        if not rows:
            return None
        wer, cer = corpus_wer_cer([c["reference"] for c in rows], [c["hyp"] for c in rows])
        return {"n": len(rows), "wer": wer, "cer": cer}

    overall_wer, overall_cer = corpus_wer_cer([c["reference"] for c in per_clip],
                                              [c["hyp"] for c in per_clip])
    per_clip_scored = []
    for c in per_clip:
        w, ce = corpus_wer_cer([c["reference"]], [c["hyp"]])
        per_clip_scored.append({**c, "wer": w, "cer": ce})

    rtfs = [c["rtf"] for c in per_clip if c["rtf"] is not None]
    durations = [c["duration_s"] for c in per_clip if c["duration_s"]]
    procs = [c["proc_s"] for c in per_clip if c["proc_s"]]

    # ---- write outputs -------------------------------------------------------
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"stt_{sanitize(spec['name'])}_{'spotcheck_' if no_reference else ''}{ts}"
    per_row_path = os.path.join(results_dir, base + ".csv")
    summary_path = os.path.join(results_dir, base + ".summary.json")

    with open(per_row_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["audio_path", "language_tag", "detected_language", "duration_s",
                    "proc_s", "rtf", "wer", "cer", "reference", "hypothesis", "error"])
        for c in per_clip_scored:
            w.writerow([c["audio_path"], c["language_tag"], c["detected_language"],
                        c["duration_s"], c["proc_s"], c["rtf"], c["wer"], c["cer"],
                        c["reference"], c["hyp"], c["error"]])

    summary = {
        "task": "stt",
        "model": spec["name"], "model_type": spec["type"], "model_id": spec.get("model_id"),
        "device": args.device, "compute_type": compute_type,
        "beam_size": args.beam_size, "forced_language": args.language,
        "gold": os.path.relpath(args.gold, HERE), "n_clips": len(per_clip),
        "wer_overall": overall_wer, "cer_overall": overall_cer,
        "by_language": {t: bucket(t) for t in LANG_TAGS},
        "real_time_factor": {
            "mean": round(statistics.mean(rtfs), 3) if rtfs else None,
            "median": round(statistics.median(rtfs), 3) if rtfs else None,
            "max": max(rtfs) if rtfs else None,
            "note": "processing_time / audio_duration; <1.0 = faster than real time.",
        },
        "audio_seconds_total": round(sum(durations), 1) if durations else None,
        "processing_seconds_total": round(sum(procs), 1) if procs else None,
        "vram_mb": {"before_load": vram_before, "peak": vram_peak,
                    "model_delta_est": (vram_peak - vram_before) if vram_peak else None,
                    "note": "per-PID nvidia-smi; delta ~= this model's footprint even on a shared GPU."},
        "note": spec.get("note", ""),
        "timestamp": ts, "per_row_csv": os.path.relpath(per_row_path, HERE),
    }
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    # ---- console summary -----------------------------------------------------
    if no_reference:
        print(f"\n  [{spec['name']}]  transcription spot-check ({len(per_clip)} clip(s), no reference)")
        for c in per_clip_scored:
            print(f"    {os.path.basename(c['audio_path'])}: detected={c['detected_language']} "
                  f"dur={c['duration_s']}s rtf={c['rtf']}")
            print(f"      hyp: {c['hyp'][:200]}{'…' if len(c['hyp']) > 200 else ''}")
    else:
        print(f"\n  [{spec['name']}]  WER={overall_wer}  CER={overall_cer}  ({len(per_clip)} clips)")
        for t in LANG_TAGS:
            b = summary["by_language"][t]
            if b:
                print(f"    {t:8s}: WER={b['wer']} CER={b['cer']}  (n={b['n']})")
    rtf = summary["real_time_factor"]
    print(f"    real-time factor: mean={rtf['mean']} median={rtf['median']} max={rtf['max']}")
    print(f"    VRAM: ~{summary['vram_mb']['model_delta_est']} MiB (peak {vram_peak} MiB, per-PID)")
    print(f"    -> {os.path.basename(summary_path)}")


if __name__ == "__main__":
    main()
