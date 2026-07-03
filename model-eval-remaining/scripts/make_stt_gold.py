#!/usr/bin/env python3
"""
make_stt_gold.py — scaffold the speech-to-text gold set (Prompt 7, Task 2).

We do NOT have labelled Urdu broadcast audio, and we must not fabricate transcripts.
So this only creates the EMPTY template + folders you fill in yourself:

  gold_sets/stt_gold.csv          header only: audio_path, reference_transcript, language_tag
  datasets/stt/audio/             drop your .wav/.mp3 clips here (kept via .gitkeep)
  datasets/stt/README.md          how to record/transcribe (written separately)

language_tag must be one of {urdu, english, mixed}. audio_path is relative to the
project root (e.g. datasets/stt/audio/clip01.wav). eval_stt.py stays BLOCKED until
this CSV has real rows.

Runs in the eval venv (stdlib only):
  venv/bin/python scripts/make_stt_gold.py
"""
import csv
import io
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD = os.path.join(HERE, "gold_sets", "stt_gold.csv")
AUDIO_DIR = os.path.join(HERE, "datasets", "stt", "audio")
HEADER = ["audio_path", "reference_transcript", "language_tag"]


def main():
    os.makedirs(os.path.dirname(GOLD), exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)

    gitkeep = os.path.join(AUDIO_DIR, ".gitkeep")
    if not os.path.exists(gitkeep):
        open(gitkeep, "w").close()

    if os.path.exists(GOLD):
        with open(GOLD, encoding="utf-8") as fh:
            n_rows = sum(1 for _ in csv.reader(fh)) - 1
        print(f"EXISTS (kept, not overwritten): {GOLD}  ({max(n_rows,0)} data rows)")
    else:
        with io.open(GOLD, "w", encoding="utf-8", newline="") as fh:
            csv.writer(fh).writerow(HEADER)
        print(f"Wrote EMPTY template (header only) -> {GOLD}")

    print(f"Audio drop folder: {os.path.relpath(AUDIO_DIR, HERE)}/  (put .wav/.mp3 clips here)")
    print("\nTo populate (see datasets/stt/README.md for full guidance):")
    print(f"  1. Drop 10-20 short (30-90s) broadcast clips into {os.path.relpath(AUDIO_DIR, HERE)}/")
    print("  2. Add one row per clip to gold_sets/stt_gold.csv:")
    print("       audio_path            = datasets/stt/audio/<clip>.wav")
    print("       reference_transcript  = your verbatim hand transcription")
    print("       language_tag          = urdu | english | mixed")
    print("  3. Then run eval_stt.py (it stays BLOCKED until rows exist).")


if __name__ == "__main__":
    main()
