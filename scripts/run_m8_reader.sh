#!/usr/bin/env bash
# M8: reader EM/F1 over the routed contexts (agent.md Section 11, Section 5).
#
#   ./scripts/run_m8_reader.sh [test|full]
#
# Standalone: resolves the repo from this file, uses .venv directly, loads .env,
# and starts the Ollama server if it is not running.
#
# test = 10 questions per (condition, dataset) + an estimated total; full = all.
# Each (condition, dataset) runs in its own process. RESUMABLE: on re-run,
# questions already in results/read_<method>_<dataset>.jsonl are skipped.

set -euo pipefail
. "$(cd "$(dirname "$0")" && pwd)/_preflight.sh"
preflight yes || exit 1

MODE="${1:-test}"
REASON="M8 reader run (${MODE})"
METHODS="vector ppr llm_trace gnn gnn_rrf stacked oracle"

echo "=== M8: reader (${MODE}) ==="
echo "  reader: ${SLM_TAG}, greedy, max 24 tokens, K=4 paragraphs, one fixed template"
echo "  conditions: ${METHODS}"

# the reader consumes the SAVED M7 routings; it never re-runs a router
need=""
for ds in musique hotpotqa; do
  for m in vector ppr llm_trace gnn gnn_rrf stacked; do
    need="$need results/val_${m}_${ds}.jsonl"
  done
done
require_files $need || exit 1

if ! ollama list | tail -n +2 | awk '{print $1}' | grep -qx "$SLM_TAG"; then
  echo "ERROR: reader model ${SLM_TAG} is not installed. Run: ollama pull ${SLM_TAG}"
  exit 1
fi

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

echo
echo "=== M8 DONE (${MODE}) ==="
if [ "$MODE" = "test" ]; then
  "$PY" scripts/m8_reader.py --summarize-estimate
  echo
  echo "  wrote: results/read_<method>_<dataset>_test.json (14 files, timings only)"
  echo "  re-run the real thing with: ./scripts/run_m8_reader.sh full"
else
  "$PY" scripts/m8_reader.py --report
  echo
  echo "  wrote: results/read_<method>_<dataset>.jsonl  (per-question answers, EM/F1)"
  echo "         results/read_<method>_<dataset>.json   (per-condition summary)"
  echo "  a successful finish prints the EM/F1 table plus paired-bootstrap CIs"
  echo "  against Vector RAG and against the gold oracle."
fi
exit 0
