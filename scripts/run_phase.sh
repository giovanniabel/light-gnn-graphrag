#!/usr/bin/env bash
# caffeinate wrapper + env + ollama load/unload (agent.md Sections 2, 6.1).
#
#   scripts/run_phase.sh <phase> [extra args for main.py ...]
#
# Phases that train the GNN get Ollama unloaded first; LLM phases get it kept warm.
# Long phases run under `caffeinate -i` with the laptop on power.

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

PHASE="${1:?usage: run_phase.sh <phase> [args...]}"
shift || true

set -a
# shellcheck disable=SC1091
[ -f .env ] && . ./.env
set +a

PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || { echo "no venv at $PY -- run: uv venv --python 3.12"; exit 1; }

# Section 6.1: which phases may have Ollama loaded.
case "$PHASE" in
  m0|pilot|m4|m7|m7b|m8) OLLAMA_PHASE=1 ;;
  *)                     OLLAMA_PHASE=0 ;;
esac

unload_ollama() {
  echo "[run_phase] unloading ollama models (Section 6.1)"
  "$PY" - <<'PYEOF'
import sys; sys.path.insert(0, ".")
from src.ollama_client import unload_all
print("[run_phase] unloaded:", unload_all() or "nothing loaded")
PYEOF
}

if [ "$OLLAMA_PHASE" = "0" ]; then
  unload_ollama
  export OLLAMA_KEEP_ALIVE=0
else
  pgrep -x ollama >/dev/null 2>&1 || { echo "[run_phase] starting ollama serve"; nohup ollama serve >/tmp/ollama_serve.log 2>&1 & sleep 3; }
  echo "[run_phase] ollama phase: NUM_PARALLEL=${OLLAMA_NUM_PARALLEL:-1} MAX_LOADED=${OLLAMA_MAX_LOADED_MODELS:-1} KEEP_ALIVE=${OLLAMA_KEEP_ALIVE:-30m}"
fi

if [ "$(pmset -g batt | head -1 | grep -c 'AC Power')" -eq 0 ]; then
  echo "[run_phase] WARNING: on battery. agent.md Section 2 says run long phases on power."
fi

echo "[run_phase] caffeinate -i $PY main.py $PHASE $*"
exec caffeinate -i "$PY" main.py "$PHASE" "$@"
