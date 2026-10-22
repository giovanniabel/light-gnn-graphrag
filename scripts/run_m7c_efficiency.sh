#!/usr/bin/env bash
# M7c: re-measure the efficiency stages that Section 7.3 invalidated (D21).
#
#   scripts/run_m7c_efficiency.sh [test|full]
#
# test = 10 questions per (stage, dataset) and an extrapolated total; full = all.
# One dataset per process; the encoder and embedding cache are released before
# Ollama loads. Aborts if swap exceeds the gate, and logs swap growth per stage.

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
MODE="${1:-test}"
PY="$ROOT/.venv/bin/python"
REASON="M7c efficiency re-measurement (${MODE})"
SWAP_GATE_MB=300

set -a; [ -f .env ] && . ./.env; set +a

echo "=== M7c: efficiency re-measurement (${MODE}) ==="
swap=$("$PY" -c 'import psutil;print(int(psutil.swap_memory().used/1048576))')
avail=$("$PY" -c 'import psutil;print(int(psutil.virtual_memory().available/1048576))')
echo "  swap ${swap} MB | RAM available ${avail} MB | gate: swap <= ${SWAP_GATE_MB} MB"
if [ "$swap" -gt "$SWAP_GATE_MB" ]; then
  echo "  ABORT: swap ${swap} MB exceeds the ${SWAP_GATE_MB} MB gate."
  echo "  Section 7.3 invalidates any stage whose swap grows past 500 MB, and starting"
  echo "  above the gate makes that near-certain. Close memory-heavy apps (or reboot)"
  echo "  and re-run. Nothing was measured."
  exit 2
fi

"$PY" - <<'PYEOF'
import sys; sys.path.insert(0, ".")
from src.ollama_client import unload_all
print(f"  ollama unloaded: {unload_all() or 'nothing was loaded'}")
PYEOF

for ds in musique hotpotqa; do
  echo
  echo "--- ${ds}: LLM-Trace (own process, Ollama loaded last) ---"
  ALLOW_VALIDATION=1 ALLOW_VALIDATION_REASON="$REASON" \
    caffeinate -i "$PY" scripts/m7c_stage.py llm_trace "$ds" "$MODE"
  echo "--- ${ds}: GNN + Vector routing (own process, Ollama OFF) ---"
  ALLOW_VALIDATION=1 ALLOW_VALIDATION_REASON="$REASON" \
    caffeinate -i "$PY" scripts/m7c_stage.py routing "$ds" "$MODE"
done

echo
echo "  outputs: results/m7c_<stage>_<dataset>_${MODE}.json"
[ "$MODE" = "test" ] && echo "  this was the 10-question test; re-run with 'full' to measure everything."
