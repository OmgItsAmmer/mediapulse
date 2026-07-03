#!/usr/bin/env bash
# =============================================================================
# serve_and_eval.sh
#   Serve ONE model on vLLM, wait until it's healthy, run an eval script against
#   it, then gracefully tear the server down. vLLM logs are kept for debugging.
#
# Usage:
#   ./scripts/serve_and_eval.sh <model_name_or_path> <eval_script_name> [extra vllm args...]
#
# Examples:
#   ./scripts/serve_and_eval.sh Qwen/Qwen3-8B eval_ner.py
#   ./scripts/serve_and_eval.sh Qwen/Qwen3-8B eval_sentiment.py --quantization awq
#
# Env overrides (all optional):
#   PORT=8000  MAX_MODEL_LEN=8192  GPU_MEM_UTIL=0.90  HEALTH_TIMEOUT=300
#   VENV_PY=<path to venv python>   (defaults to <project>/venv/bin/python)
#
# The served model id is passed to the eval script via the env var EVAL_MODEL
# (and VLLM_BASE_URL is exported too), so eval scripts stay decoupled from CLI args.
# =============================================================================
set -euo pipefail

# ---- resolve project root (this script lives in <root>/scripts/) ------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ---- args -------------------------------------------------------------------
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <model_name_or_path> <eval_script_name> [extra vllm args...]" >&2
  echo "Example: $0 Qwen/Qwen3-8B eval_ner.py" >&2
  exit 2
fi
MODEL="$1"; shift
EVAL_SCRIPT="$1"; shift
EXTRA_VLLM_ARGS=("$@")

# ---- config -----------------------------------------------------------------
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-300}"          # seconds to wait for readiness
VENV_PY="${VENV_PY:-$PROJECT_ROOT/venv/bin/python}"
BASE_URL="http://localhost:${PORT}/v1"
HEALTH_URL="http://localhost:${PORT}/health"

# ---- resolve eval script path ----------------------------------------------
if [[ -f "$EVAL_SCRIPT" ]]; then
  EVAL_PATH="$EVAL_SCRIPT"
elif [[ -f "scripts/$EVAL_SCRIPT" ]]; then
  EVAL_PATH="scripts/$EVAL_SCRIPT"
else
  echo "ERROR: eval script not found: '$EVAL_SCRIPT' (looked in ./ and ./scripts/)." >&2
  exit 2
fi

# ---- sanity checks ----------------------------------------------------------
command -v vllm >/dev/null 2>&1 || { echo "ERROR: 'vllm' not found on PATH." >&2; exit 1; }
[[ -x "$VENV_PY" ]] || { echo "ERROR: venv python not found/executable at '$VENV_PY'." >&2; exit 1; }

mkdir -p logs results
SAFE_MODEL="$(printf '%s' "$MODEL" | tr '/ :' '___')"
TS="$(date +%Y%m%d_%H%M%S)"
VLLM_LOG="logs/vllm_${SAFE_MODEL}_${TS}.log"

# ---- health probe (curl if available, else venv python/urllib) --------------
health_ok() {
  if command -v curl >/dev/null 2>&1; then
    curl -sf "$HEALTH_URL" >/dev/null 2>&1
  else
    "$VENV_PY" - "$HEALTH_URL" <<'PY'
import sys, urllib.request
try:
    with urllib.request.urlopen(sys.argv[1], timeout=3) as r:
        sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
PY
  fi
}

# ---- refuse to stomp on an already-running server ---------------------------
if health_ok; then
  echo "ERROR: something is already serving on port $PORT (health responds OK)." >&2
  echo "       Stop it first, or set PORT=<other> to use a different port." >&2
  exit 1
fi

# ---- teardown: kill vLLM + all descendants (procps pgrep) -------------------
VLLM_PID=""
kill_tree() {                     # kill_tree <pid> <signal>
  local pid="$1" sig="$2" child
  for child in $(pgrep -P "$pid" 2>/dev/null); do kill_tree "$child" "$sig"; done
  kill -"$sig" "$pid" 2>/dev/null || true
}
cleanup() {
  [[ -z "$VLLM_PID" ]] && return
  if kill -0 "$VLLM_PID" 2>/dev/null; then
    echo ">> Stopping vLLM (pid $VLLM_PID) ..."
    kill_tree "$VLLM_PID" TERM
    for _ in $(seq 1 30); do kill -0 "$VLLM_PID" 2>/dev/null || break; sleep 1; done
    if kill -0 "$VLLM_PID" 2>/dev/null; then
      echo ">> vLLM did not exit in 30s; sending SIGKILL."
      kill_tree "$VLLM_PID" KILL
    fi
  fi
}
trap cleanup EXIT INT TERM

# ---- start vLLM -------------------------------------------------------------
echo ">> Model:       $MODEL"
echo ">> Eval script: $EVAL_PATH"
echo ">> vLLM log:    $VLLM_LOG"
echo ">> Base URL:    $BASE_URL   (health: $HEALTH_URL, timeout ${HEALTH_TIMEOUT}s)"
echo ">> Starting vLLM ..."
vllm serve "$MODEL" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  "${EXTRA_VLLM_ARGS[@]}" \
  > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
echo ">> vLLM PID: $VLLM_PID"

# ---- wait for health --------------------------------------------------------
echo -n ">> Waiting for vLLM to become healthy"
DEADLINE=$(( SECONDS + HEALTH_TIMEOUT ))
until health_ok; do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo ""; echo "ERROR: vLLM exited before becoming healthy. Last 40 log lines:" >&2
    tail -n 40 "$VLLM_LOG" >&2; exit 1
  fi
  if (( SECONDS >= DEADLINE )); then
    echo ""; echo "ERROR: vLLM not healthy within ${HEALTH_TIMEOUT}s. Last 40 log lines:" >&2
    tail -n 40 "$VLLM_LOG" >&2; exit 1
  fi
  echo -n "."; sleep 3
done
echo ""; echo ">> vLLM is healthy."

# ---- run eval ---------------------------------------------------------------
echo ">> Running: EVAL_MODEL='$MODEL' $VENV_PY $EVAL_PATH"
set +e
EVAL_MODEL="$MODEL" VLLM_BASE_URL="$BASE_URL" "$VENV_PY" "$EVAL_PATH"
EVAL_RC=$?
set -e
echo ">> Eval finished with exit code $EVAL_RC"

# teardown happens via the EXIT trap
exit "$EVAL_RC"
