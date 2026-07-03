#!/usr/bin/env bash
# =============================================================================
# run_langid_eval.sh — run eval_langid.py for every langid model in config.yaml,
# against BOTH gold_sets/langid_gold.csv (balanced 4-class) and
# datasets/langid/combined.csv (roman_urdu + urdu at scale).
#
# Everything runs in the eval VENV (fasttext-predict + langdetect + openai; no torch).
#   - fasttext / langdetect : run eval_langid.py directly.
#   - vllm                  : via serve_and_eval.sh (serve -> health -> eval -> teardown).
#
# LLM output is one word, so runs are quick; set EVAL_LIMIT to cap if desired.
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="${CONFIG:-config.yaml}"
VENV_PY="${VENV_PY:-$PROJECT_ROOT/venv/bin/python}"
export EVAL_LIMIT="${EVAL_LIMIT:-0}"

[[ -x "$VENV_PY" ]] || { echo "ERROR: venv python not found at $VENV_PY" >&2; exit 1; }

mapfile -t MODELS < <("$VENV_PY" - "$CONFIG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
for s in cfg.get("models", {}).get("langid", []):
    print(f"{s.get('name')}\t{s.get('type')}\t{s.get('model_id','')}")
PY
)

echo "Langid models: ${#MODELS[@]}  (EVAL_LIMIT=$EVAL_LIMIT)"
for line in "${MODELS[@]}"; do
  IFS=$'\t' read -r NAME TYPE MODEL_ID <<< "$line"
  echo; echo "==================================================================="
  echo ">> MODEL: $NAME  (type=$TYPE, id=$MODEL_ID)"
  echo "==================================================================="
  case "$TYPE" in
    fasttext|langdetect)
      "$VENV_PY" scripts/eval_langid.py --model "$NAME"
      ;;
    vllm)
      if [[ -x scripts/serve_and_eval.sh ]]; then
        ./scripts/serve_and_eval.sh "$MODEL_ID" eval_langid.py
      else
        echo "serve_and_eval.sh missing; serve manually then:"
        echo "    EVAL_MODEL='$MODEL_ID' $VENV_PY scripts/eval_langid.py"
      fi
      ;;
    *) echo "  (no handler for type '$TYPE' — skipping $NAME)";;
  esac
done
echo; echo ">> Done. Results in results/  (langid_*.summary.json + per-row CSVs)."
