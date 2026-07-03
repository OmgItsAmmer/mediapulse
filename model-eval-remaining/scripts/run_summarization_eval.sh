#!/usr/bin/env bash
# =============================================================================
# run_summarization_eval.sh — run eval_summarization.py for every summarization
# model in config.yaml. Everything runs in CONDA BASE (BERTScore needs torch +
# transformers; openai for the vLLM path). Install first:
#   conda run -n base pip install sumy rouge-score bert-score nltk
#
#   - textrank : conda-base python directly (CPU, no vLLM).
#   - vllm     : serve_and_eval.sh with VENV_PY pointed at the conda-base python,
#                so the eval side has torch for BERTScore. Serve with a lower
#                GPU_MEM_UTIL so BERTScore's model also fits on the 24 GB card.
#
# Scored vs gold_sets/summarization_gold.csv (PLACEHOLDER refs — rewrite first).
# Prereqs: scripts/prepare_summarization_data.py + scripts/make_summarization_gold.py.
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="${CONFIG:-config.yaml}"
CONDA_ENV="${CONDA_ENV:-base}"
# conda-base python (has torch); used both directly and as serve_and_eval's VENV_PY.
CONDA_PY="${CONDA_PY:-$(conda run -n "$CONDA_ENV" bash -c 'command -v python')}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.6}"        # leave VRAM for the BERTScore model

[[ -n "$CONDA_PY" ]] || { echo "ERROR: could not resolve conda '$CONDA_ENV' python" >&2; exit 1; }

mapfile -t MODELS < <("$CONDA_PY" - "$CONFIG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
for s in cfg.get("models", {}).get("summarization", []):
    print(f"{s.get('name')}\t{s.get('type')}\t{s.get('model_id','')}")
PY
)

echo "Summarization models: ${#MODELS[@]}  (conda env=$CONDA_ENV, GPU_MEM_UTIL=$GPU_MEM_UTIL)"
for line in "${MODELS[@]}"; do
  IFS=$'\t' read -r NAME TYPE MODEL_ID <<< "$line"
  echo; echo "==================================================================="
  echo ">> MODEL: $NAME  (type=$TYPE, id=$MODEL_ID)"
  echo "==================================================================="
  case "$TYPE" in
    textrank)
      "$CONDA_PY" scripts/eval_summarization.py --model "$NAME"
      ;;
    vllm)
      if [[ -x scripts/serve_and_eval.sh ]]; then
        VENV_PY="$CONDA_PY" ./scripts/serve_and_eval.sh "$MODEL_ID" eval_summarization.py
      else
        echo "serve_and_eval.sh missing; serve $MODEL_ID manually then:"
        echo "    EVAL_MODEL='$MODEL_ID' $CONDA_PY scripts/eval_summarization.py"
      fi
      ;;
    *) echo "  (no handler for type '$TYPE' — skipping $NAME)";;
  esac
done
echo; echo ">> Done. Results in results/  (summarization_*.summary.json + per-row CSVs)."
