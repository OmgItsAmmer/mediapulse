#!/usr/bin/env python3
"""
eval_scenes.py — PySceneDetect scene/shot-change sanity check (Prompt 10, Module 4.9).

SINGLE candidate, so this is NOT a comparison — it's a spot-check that PySceneDetect
finds real cuts before we rely on it to timestamp live-stream events. There is no
automated accuracy metric without hand-labelled cut points, so it PRINTS the detected
timestamps for you to eyeball against the actual video and confirm/correct.

What it does:
  - runs the content-aware detector (or --detector adaptive|threshold) on the clip
  - writes detected scene-change timestamps + per-scene spans to
    results/scenes_<clip_name>.json
  - prints a human-readable list of cut timestamps (mm:ss.mmm)
  - logs processing_time vs clip_duration as a REAL-TIME FACTOR — this runs CPU-only
    alongside the GPU object/face detectors and must stay well under 1.0

Runtime: conda base (has cv2).  Install:  pip install scenedetect
  conda run -n base python scripts/eval_scenes.py --clip datasets/scenes/<clip>.mp4
"""
import argparse
import datetime as dt
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sanitize(s):
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)


def _secs(tc):
    # PySceneDetect >=0.6.4 exposes .seconds; older uses get_seconds().
    return tc.seconds if hasattr(tc, "seconds") else tc.get_seconds()


def fmt_tc(seconds):
    m, s = divmod(float(seconds), 60)
    return f"{int(m):02d}:{s:06.3f}"


def build_detector(name, threshold, min_len_frames):
    from scenedetect.detectors import (AdaptiveDetector, ContentDetector,
                                        ThresholdDetector)
    if name == "content":
        return ContentDetector(threshold=threshold, min_scene_len=min_len_frames)
    if name == "adaptive":
        return AdaptiveDetector(min_scene_len=min_len_frames)
    if name == "threshold":
        return ThresholdDetector(threshold=threshold, min_scene_len=min_len_frames)
    raise ValueError(f"unknown detector '{name}'")


def main():
    ap = argparse.ArgumentParser(description="PySceneDetect scene-change sanity check.")
    ap.add_argument("--clip", required=True, help="path to a short video clip (.mp4/.mkv/.mov)")
    ap.add_argument("--detector", default="content",
                    choices=["content", "adaptive", "threshold"])
    ap.add_argument("--threshold", type=float, default=27.0,
                    help="content/threshold sensitivity; lower = more cuts (default 27.0).")
    ap.add_argument("--min-scene-len", type=float, default=0.6,
                    help="minimum scene length in SECONDS (suppresses ultra-short false scenes).")
    ap.add_argument("--output-dir", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    clip = args.clip if os.path.isabs(args.clip) else os.path.join(HERE, args.clip)
    if not os.path.exists(clip):
        sys.exit(f"ERROR: clip not found: {clip}\n  Drop a 1-3 min clip in "
                 "datasets/scenes/ (see datasets/scenes/README.md).")

    try:
        from scenedetect import SceneManager, open_video
    except ImportError:
        sys.exit("ERROR: scenedetect not installed. From conda base (has cv2): "
                 "pip install scenedetect")

    video = open_video(clip)
    fps = float(video.frame_rate)
    min_len_frames = max(1, int(round(args.min_scene_len * fps)))

    sm = SceneManager()
    sm.add_detector(build_detector(args.detector, args.threshold, min_len_frames))

    print(f"Clip: {os.path.relpath(clip, HERE)}  (fps={fps:.3f})")
    print(f"Detector: {args.detector} (threshold={args.threshold}, "
          f"min_scene_len={args.min_scene_len}s={min_len_frames}f)")

    t0 = time.perf_counter()
    sm.detect_scenes(video, show_progress=False)
    proc_s = time.perf_counter() - t0
    scenes = sm.get_scene_list()          # list of (start, end) FrameTimecode pairs

    if scenes:
        duration_s = _secs(scenes[-1][1])
    else:
        try:
            duration_s = _secs(video.duration)
        except Exception:
            duration_s = None

    # cut points = start of every scene after the first (the shot boundaries)
    cut_points = [round(_secs(s[0]), 3) for s in scenes[1:]]
    scene_rows = [{
        "scene": i + 1,
        "start_s": round(_secs(s), 3), "end_s": round(_secs(e), 3),
        "start_tc": s.get_timecode(), "end_tc": e.get_timecode(),
        "length_s": round(_secs(e) - _secs(s), 3),
    } for i, (s, e) in enumerate(scenes)]

    rtf = round(proc_s / duration_s, 3) if duration_s else None
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    clip_name = sanitize(os.path.splitext(os.path.basename(clip))[0])
    out_path = os.path.join(args.output_dir, f"scenes_{clip_name}.json")
    os.makedirs(args.output_dir, exist_ok=True)

    summary = {
        "task": "scene_detection",
        "clip": os.path.relpath(clip, HERE), "clip_name": clip_name,
        "detector": args.detector, "threshold": args.threshold,
        "min_scene_len_s": args.min_scene_len, "fps": round(fps, 3),
        "clip_duration_s": round(duration_s, 3) if duration_s else None,
        "n_scenes": len(scenes), "n_cuts": len(cut_points),
        "cut_points_s": cut_points, "scenes": scene_rows,
        "processing_s": round(proc_s, 3), "real_time_factor": rtf,
        "note": "Single candidate (PySceneDetect) — spot-check, not an automated metric. "
                "Eyeball cut_points_s against the video and confirm/correct by hand.",
        "timestamp": ts,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    # ---- human-readable spot-check ------------------------------------------
    print(f"\n  detected {len(scenes)} scenes / {len(cut_points)} cuts  "
          f"(clip {fmt_tc(duration_s) if duration_s else '?'})")
    if cut_points:
        print("  cut timestamps (eyeball these against the video):")
        for i, c in enumerate(cut_points, 1):
            print(f"    cut {i:>2}: {fmt_tc(c)}  ({c:.3f}s)")
    else:
        print("  no cuts detected — try a lower --threshold if the clip clearly has cuts.")
    print(f"  processing={proc_s:.2f}s  clip={duration_s:.2f}s  "
          f"real-time factor={rtf} (CPU; must stay < 1.0 next to GPU detectors)"
          if duration_s else f"  processing={proc_s:.2f}s")
    print(f"  -> {os.path.relpath(out_path, HERE)}")


if __name__ == "__main__":
    main()
