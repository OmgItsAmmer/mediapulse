#!/usr/bin/env bash
# =============================================================================
# run_ner_eval.sh — run eval_ner.py for every NER model in config.yaml, against
# BOTH the public set (datasets/ner/ner_public.csv) and the gold set
# (gold_sets/ner_gold.csv). eval_ner.py evaluates both datasets per invocation.
#
#   - spacy / local models : run eval_ner.py directly (no server needed).
#   - vllm  models         : use scripts/serve_and_eval.sh if present (serves the
#                            model, waits for health, evals, tears down). Otherwise
#                            prints the manual `vllm serve` command to run yourself.
#
# vLLM models are done one at a time (single GPU). Quick smoke run:
#   EVAL_LIMIT=15 ./scripts/run_ner_eval.sh
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="${CONFIG:-config.yaml}"
VENV_PY="${VENV_PY:-$PROJECT_ROOT/venv/bin/python}"
export EVAL_LIMIT="${EVAL_LIMIT:-0}"

[[ -x "$VENV_PY" ]] || { echo "ERROR: venv python not found at $VENV_PY" >&2; exit 1; }

# pull NER model rows as: name<TAB>type<TAB>model_id
mapfile -t MODELS < <("$VENV_PY" - "$CONFIG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
for s in cfg.get("models", {}).get("ner", []):
    print(f"{s.get('name')}\t{s.get('type')}\t{s.get('model_id','')}")
PY
)

echo "NER models to evaluate: ${#MODELS[@]}  (EVAL_LIMIT=$EVAL_LIMIT)"
for line in "${MODELS[@]}"; do
  IFS=$'\t' read -r NAME TYPE MODEL_ID <<< "$line"
  echo; echo "==================================================================="
  echo ">> MODEL: $NAME  (type=$TYPE, id=$MODEL_ID)"
  echo "==================================================================="
  case "$TYPE" in
    spacy|hf_pipeline|langdetect|fasttext)
      "$VENV_PY" scripts/eval_ner.py --model "$NAME"
      ;;
    vllm)
      if [[ -x scripts/serve_and_eval.sh ]]; then
        ./scripts/serve_and_eval.sh "$MODEL_ID" eval_ner.py
      else
        echo "serve_and_eval.sh not found. Serve manually, then run eval:"
        echo "    vllm serve $MODEL_ID --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.90"
        echo "    EVAL_MODEL='$MODEL_ID' $VENV_PY scripts/eval_ner.py"
        echo "  (skipping automated run for $NAME)"
      fi
      ;;
    *)
      echo "  (no handler for type '$TYPE' — skipping $NAME)"
      ;;
  esac
done

echo; echo ">> Done. Results in results/  (ner_*.summary.json + per-row CSVs)."
