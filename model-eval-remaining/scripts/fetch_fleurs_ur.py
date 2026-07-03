#!/usr/bin/env python3
"""
fetch_fleurs_ur.py — download a small FLEURS ur_PK (Urdu-Pakistan) subset as a
READ-SPEECH SANITY BASELINE for the 4.6 STT eval.

*** This is NOT the target domain. *** FLEURS ur_PK is clean, monolingual, READ
speech (people reading FLoRes sentences). It is real Pakistani Urdu with ground-truth
transcripts, so the WER is genuine — but it will FLATTER the models vs spontaneous,
code-switched broadcast audio. Kept in a SEPARATE gold file so it never mixes with
the real broadcast clips you will supply (gold_sets/stt_gold.csv).

Writes:
  datasets/stt/audio/fleurs_ur_PK/<id>.wav          (16 kHz mono)
  gold_sets/stt_gold_fleurs_ur_pk.csv               [audio_path, reference_transcript, language_tag=urdu]

Run from the STT env (has datasets + soundfile):
  conda run -n STTLiveTransciptionVoxtralEnv python scripts/fetch_fleurs_ur.py --n 20
"""
import argparse
import csv
import io
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Guard: the project has a datasets/ dir that would shadow the HF `datasets` lib
# if the project root is on sys.path. Drop it before importing datasets.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != HERE]

AUDIO_DIR = os.path.join(HERE, "datasets", "stt", "audio", "fleurs_ur_PK")
GOLD = os.path.join(HERE, "gold_sets", "stt_gold_fleurs_ur_pk.csv")


def load_stream(n):
    # FLEURS is parquet on the Hub now (config is lowercase 'ur_pk'); no loading
    # script, so no trust_remote_code. Stream to avoid downloading the whole split.
    # decode=False -> we get raw audio bytes and decode with soundfile ourselves,
    # avoiding the newer datasets requirement on torchcodec.
    from datasets import Audio, load_dataset
    try:
        ds = load_dataset("google/fleurs", "ur_pk", split="test", streaming=True)
        return ds.cast_column("audio", Audio(decode=False))
    except Exception as ex:
        raise SystemExit(f"ERROR: could not load google/fleurs ur_pk ({ex}).\n"
                         "  Check network / `pip install -U datasets`.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="how many clips to fetch")
    args = ap.parse_args()

    import itertools
    import soundfile as sf

    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(GOLD), exist_ok=True)

    print(f"Streaming google/fleurs ur_PK (test) — taking {args.n} clips ...")
    ds = load_stream(args.n)

    rows = []
    for i, ex in enumerate(itertools.islice(ds, args.n)):
        audio = ex["audio"]                       # {'bytes': ..., 'path': ...} (decode=False)
        raw = audio.get("bytes")
        if not raw:
            print(f"  [{i+1}] no embedded audio bytes — skipping", file=sys.stderr)
            continue
        arr, sr = sf.read(io.BytesIO(raw))         # decode wav/flac bytes ourselves
        ref = (ex.get("raw_transcription") or ex.get("transcription") or "").strip()
        if not ref:
            continue
        clip_id = f"fleurs_ur_{ex.get('id', i):0>5}" if str(ex.get("id", "")).isdigit() \
            else f"fleurs_ur_{i:03d}"
        wav_path = os.path.join(AUDIO_DIR, clip_id + ".wav")
        sf.write(wav_path, arr, sr, subtype="PCM_16")
        rel = os.path.relpath(wav_path, HERE)
        rows.append((rel, ref, "urdu"))
        dur = len(arr) / float(sr)
        print(f"  [{i+1:>2}/{args.n}] {clip_id}.wav  {dur:5.1f}s  ref: {ref[:60]}")

    if not rows:
        raise SystemExit("ERROR: fetched 0 usable clips (no transcripts?).")

    with io.open(GOLD, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["audio_path", "reference_transcript", "language_tag"])
        w.writerows(rows)

    print(f"\nWrote {len(rows)} clips -> {os.path.relpath(GOLD, HERE)}")
    print(f"  audio in: {os.path.relpath(AUDIO_DIR, HERE)}/")
    print("  NOTE: READ-SPEECH sanity baseline (all language_tag=urdu), NOT broadcast.")


if __name__ == "__main__":
    main()
