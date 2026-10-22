#!/usr/bin/env bash
# M8: reader EM/F1 over the routed contexts (agent.md Section 11, Section 5).
#
#   scripts/run_m8_reader.sh [test|full]
#
# test = 10 questions per (method, dataset) plus an estimated total; full = all.
# Each (method, dataset) runs in its own process. Resumable: completed questions
# are skipped on re-run.

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
MODE="${1:-test}"
PY="$ROOT/.venv/bin/python"
REASON="M8 reader run (${MODE})"

set -a; [ -f .env ] && . ./.env; set +a

# Section 5 reader conditions. `oracle` feeds the gold paragraphs (the ceiling).
METHODS="vector ppr llm_trace gnn gnn_rrf stacked oracle"

echo "=== M8: reader (${MODE}) ==="
"$PY" -c 'import psutil;print(f"  swap {int(psutil.swap_memory().used/1048576)} MB | RAM available {int(psutil.virtual_memory().available/1048576)} MB")'
echo "  reader: ${SLM_TAG}, greedy, max 24 tokens, K=4 paragraphs, one fixed template"
echo "  conditions: ${METHODS}"

for ds in musique hotpotqa; do
  for m in $METHODS; do
    echo
    echo "--- ${m} / ${ds} ---"
    ALLOW_VALIDATION=1 ALLOW_VALIDATION_REASON="$REASON" \
      caffeinate -i "$PY" scripts/m8_reader.py "$m" "$ds" "$MODE"
  done
done

"$PY" - <<'PYEOF'
import sys; sys.path.insert(0, ".")
from src.ollama_client import unload_all
print(f"\n  ollama unloaded: {unload_all() or 'nothing was loaded'}")
PYEOF

if [ "$MODE" = "test" ]; then
  "$PY" scripts/m8_reader.py --summarize-estimate
else
  "$PY" scripts/m8_reader.py --report
fi
