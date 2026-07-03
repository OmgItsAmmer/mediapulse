#!/usr/bin/env bash
# =============================================================================
# run_stt_eval.sh — run eval_stt.py for every STT model in config.yaml.
#
# *** BLOCKED until gold audio exists. *** Each model prints a clear BLOCKED
# message and exits until gold_sets/stt_gold.csv has real rows + the audio is in
# datasets/stt/audio/ (see datasets/stt/README.md).
#
# Runtime: conda env STTLiveTransciptionVoxtralEnv (has faster-whisper + audio libs).
#   One-time:  conda run -n STTLiveTransciptionVoxtralEnv pip install jiwer
#
# NOTE: faster-whisper large-v3 (float16) needs ~3 GB VRAM; if the 3090 is busy with
# another workload, free it or run with --device cpu / --compute-type int8.
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="${CONFIG:-config.yaml}"
STT_ENV="${STT_ENV:-STTLiveTransciptionVoxtralEnv}"
STT_PY="${STT_PY:-$(conda run -n "$STT_ENV" bash -c 'command -v python')}"

[[ -n "$STT_PY" ]] || { echo "ERROR: could not resolve conda '$STT_ENV' python" >&2; exit 1; }

mapfile -t MODELS < <("$STT_PY" - "$CONFIG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
for s in cfg.get("models", {}).get("stt", []):
    print(f"{s.get('name')}\t{s.get('type')}\t{s.get('model_id','')}")
PY
)

echo "STT models: ${#MODELS[@]}  (env=$STT_ENV)"
for line in "${MODELS[@]}"; do
  IFS=$'\t' read -r NAME TYPE MODEL_ID <<< "$line"
  echo; echo "==================================================================="
  echo ">> MODEL: $NAME  (type=$TYPE, id=$MODEL_ID)"
  echo "==================================================================="
  # eval_stt.py exits 3 when BLOCKED and 4 when an optional model (NeMo) is skipped;
  # keep going to the next model instead of aborting the whole loop.
  "$STT_PY" scripts/eval_stt.py --model "$NAME" || echo "  (exit $? — see message above)"
done
echo; echo ">> Done. Results in results/  (stt_*.summary.json + per-row CSVs) once unblocked."
