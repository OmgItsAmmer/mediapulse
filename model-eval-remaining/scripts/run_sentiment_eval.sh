#!/usr/bin/env bash
# =============================================================================
# run_sentiment_eval.sh — run eval_sentiment.py for every sentiment model in
# config.yaml, against BOTH gold_sets/sentiment_gold.csv and
# datasets/sentiment/combined.csv.
#
# IMPORTANT: sentiment uses the CONDA-BASE python (torch + transformers live there,
# not in the eval venv). The XLM-R baseline runs on GPU via conda-base torch; vLLM
# models are served via serve_and_eval.sh (also driven by conda-base python).
#
# combined.csv is ~21k rows -> LLM runs are capped by EVAL_LIMIT (default 200).
# Full baseline sweep:  EVAL_LIMIT=0 ./scripts/run_sentiment_eval.sh
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="${CONFIG:-config.yaml}"
CONDA_PY="${CONDA_PY:-/home/temp/miniconda3/bin/python}"   # has torch/transformers/openai
export EVAL_LIMIT="${EVAL_LIMIT:-200}"

[[ -x "$CONDA_PY" ]] || { echo "ERROR: conda-base python not found at $CONDA_PY" >&2; exit 1; }

mapfile -t MODELS < <("$CONDA_PY" - "$CONFIG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
for s in cfg.get("models", {}).get("sentiment", []):
    print(f"{s.get('name')}\t{s.get('type')}\t{s.get('model_id','')}")
PY
)

echo "Sentiment models: ${#MODELS[@]}  (EVAL_LIMIT=$EVAL_LIMIT, python=$CONDA_PY)"
for line in "${MODELS[@]}"; do
  IFS=$'\t' read -r NAME TYPE MODEL_ID <<< "$line"
  echo; echo "==================================================================="
  echo ">> MODEL: $NAME  (type=$TYPE, id=$MODEL_ID)"
  echo "==================================================================="
  case "$TYPE" in
    hf_pipeline)
      "$CONDA_PY" scripts/eval_sentiment.py --model "$NAME"
      ;;
    vllm)
      if [[ -x scripts/serve_and_eval.sh ]]; then
        VENV_PY="$CONDA_PY" ./scripts/serve_and_eval.sh "$MODEL_ID" eval_sentiment.py
      else
        echo "serve_and_eval.sh missing; serve manually then:"
        echo "    EVAL_MODEL='$MODEL_ID' $CONDA_PY scripts/eval_sentiment.py"
      fi
      ;;
    *) echo "  (no handler for type '$TYPE' — skipping $NAME)";;
  esac
done
echo; echo ">> Done. Results in results/  (sentiment_*.summary.json + per-row CSVs)."
