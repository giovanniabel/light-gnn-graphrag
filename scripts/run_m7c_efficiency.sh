#!/usr/bin/env bash
# M7c: re-measure the efficiency stages that Section 7.3 invalidated (D21).
#
#   ./scripts/run_m7c_efficiency.sh [test|full]
#
# Standalone: resolves the repo from this file, uses .venv directly, loads .env,
# and starts the Ollama server if it is not running. Nothing else is required.
#
# test = 10 questions per (stage, dataset) + an extrapolation; full = all 150.
# One dataset per process; the encoder and embedding cache are released before
# Ollama loads. Aborts if swap exceeds the gate; logs swap growth per stage.

set -euo pipefail
. "$(cd "$(dirname "$0")" && pwd)/_preflight.sh"
preflight yes || exit 1

MODE="${1:-test}"
REASON="M7c efficiency re-measurement (${MODE})"
SWAP_GATE_MB=300

echo "=== M7c: efficiency re-measurement (${MODE}) ==="
require_files results/checkpoints_final data/cache/graphs data/cache/triples.jsonl || exit 1

swap=$("$PY" -c 'import psutil;print(int(psutil.swap_memory().used/1048576))')
echo "  swap gate: ${swap} MB used, limit ${SWAP_GATE_MB} MB"
if [ "$swap" -gt "$SWAP_GATE_MB" ]; then
  cat <<EOF
  ABORT: swap ${swap} MB exceeds the ${SWAP_GATE_MB} MB gate. Nothing was measured.

  Why: Section 7.3 invalidates any stage whose swap grows past 500 MB. Starting
  above the gate on an 8 GB machine makes that near-certain, and it is exactly
  what voided the first LLM-Trace measurement (D21).

  What to do:
    1. Quit memory-heavy apps (browsers, Slack, the Claude app) and re-run.
    2. If swap stays high, reboot -- macOS does not reclaim swap eagerly -- then
       re-run before opening anything else.
    3. Check with:  .venv/bin/python -c 'import psutil;print(psutil.swap_memory())'
    4. Nothing is lost by aborting: no file is written and no earlier result is
       touched. Re-run the same command when swap is low.
EOF
  exit 2
fi

"$PY" - <<'PYEOF'
import sys; sys.path.insert(0, ".")
from src.ollama_client import unload_all
print(f"  ollama unloaded: {unload_all() or 'nothing was loaded'}")
PYEOF

for ds in musique hotpotqa; do
  echo
  echo "--- ${ds}: LLM-Trace (own process; Ollama loaded last) ---"
  ALLOW_VALIDATION=1 ALLOW_VALIDATION_REASON="$REASON" \
    caffeinate -i "$PY" scripts/m7c_stage.py llm_trace "$ds" "$MODE"
  echo "--- ${ds}: GNN + Vector routing (own process; Ollama OFF) ---"
  ALLOW_VALIDATION=1 ALLOW_VALIDATION_REASON="$REASON" \
    caffeinate -i "$PY" scripts/m7c_stage.py routing "$ds" "$MODE"
done

echo
echo "=== M7c DONE (${MODE}) ==="
echo "  wrote: results/m7c_<stage>_<dataset>_${MODE}.json  (4 files)"
ls -1 "$ROOT"/results/m7c_*_"${MODE}".json 2>/dev/null | sed 's|.*/|    |' || true
echo "  check each file's \"invalidated_by_swap\": false before quoting its numbers."
[ "$MODE" = "test" ] && echo "  this was the 10-question test; re-run with: ./scripts/run_m7c_efficiency.sh full"
exit 0
