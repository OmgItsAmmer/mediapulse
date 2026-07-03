#!/usr/bin/env bash
# =============================================================================
# run_topics_eval.sh — run eval_topics.py for every topic-modeling model in
# config.yaml. Two runtimes, because the deps differ:
#   - gensim_lda : eval VENV (CPU).            venv/bin/pip install gensim
#   - bertopic   : conda base (torch + sentence-transformers).  pip install bertopic
#
# ARI/NMI are measured on gold_sets/topics_gold.csv; fit+transform latency on the
# full datasets/topics/pool.csv (cap with POOL_LIMIT=N for a quick timing).
#
# Prereqs: scripts/prepare_topics_data.py (pool) + scripts/make_topics_gold.py (gold).
# =============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="${CONFIG:-config.yaml}"
VENV_PY="${VENV_PY:-$PROJECT_ROOT/venv/bin/python}"
CONDA_ENV="${CONDA_ENV:-base}"
export POOL_LIMIT="${POOL_LIMIT:-0}"

[[ -x "$VENV_PY" ]] || { echo "ERROR: venv python not found at $VENV_PY" >&2; exit 1; }

mapfile -t MODELS < <("$VENV_PY" - "$CONFIG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
for s in cfg.get("models", {}).get("topics", []):
    print(f"{s.get('name')}\t{s.get('type')}\t{s.get('model_id','')}")
PY
)

echo "Topic models: ${#MODELS[@]}  (POOL_LIMIT=$POOL_LIMIT)"
for line in "${MODELS[@]}"; do
  IFS=$'\t' read -r NAME TYPE MODEL_ID <<< "$line"
  echo; echo "==================================================================="
  echo ">> MODEL: $NAME  (type=$TYPE, id=$MODEL_ID)"
  echo "==================================================================="
  case "$TYPE" in
    gensim_lda)
      "$VENV_PY" scripts/eval_topics.py --model "$NAME"
      ;;
    bertopic)
      if command -v conda >/dev/null 2>&1; then
        conda run -n "$CONDA_ENV" python scripts/eval_topics.py --model "$NAME"
      else
        echo "conda not on PATH; from an env with torch+sentence-transformers+bertopic:"
        echo "    python scripts/eval_topics.py --model $NAME"
      fi
      ;;
    *) echo "  (no handler for type '$TYPE' — skipping $NAME)";;
  esac
done
echo; echo ">> Done. Results in results/  (topics_*.summary.json + per-row CSVs)."
