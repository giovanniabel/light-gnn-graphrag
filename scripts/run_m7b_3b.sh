#!/usr/bin/env bash
# M7b: 3B strength check for LLM-Trace (agent.md Section 5, "Strength check").
#
#   ./scripts/run_m7b_3b.sh [test|full] [model_tag]
#
# Standalone: resolves the repo from this file, uses .venv directly, loads .env,
# and starts the Ollama server if it is not running.
#
# Bounds how much the 1.5B tracer handicaps the LLM-Trace baseline. Fixed, seeded,
# stratified 60-question subset (30 per dataset), shared with the 1.5B run so the
# comparison is paired. NEVER pulls a model: if the tag is missing it prints the
# pull command and exits 3.

set -euo pipefail
. "$(cd "$(dirname "$0")" && pwd)/_preflight.sh"
preflight yes || exit 1

MODE="${1:-test}"
TAG="${2:-${STRENGTH_TAG}}"

echo "=== M7b: 3B strength check (${MODE}) ==="
echo "  tag: ${TAG}"
require_files data/cache/graphs data/cache/triples.jsonl || exit 1
echo "  installed models:"
ollama list | tail -n +2 | awk '{printf "    %s\n", $1}'

if ! ollama list | tail -n +2 | awk '{print $1}' | grep -qx "$TAG"; then
  cat <<EOF

  ${TAG} is NOT installed, so this check cannot run.
  This script never pulls anything. To enable it:

      ollama pull ${TAG}        # about 1.9 GB

  Then re-run:  ./scripts/run_m7b_3b.sh ${MODE}

  Note: llama3.2:3b IS installed, but it is a different model family, so using it
  would confound family with size and would not isolate how much of LLM-Trace's
  weakness is the 1.5B tracer. agent.md Section 2 specifies the qwen2.5 3B tag.
EOF
  exit 3
fi

ALLOW_VALIDATION=1 ALLOW_VALIDATION_REASON="M7b 3B strength check (${MODE})" \
  caffeinate -i "$PY" scripts/m7b_3b_subset.py "$TAG" "$MODE"

echo
echo "=== M7b 3B DONE (${MODE}) ==="
echo "  subset (fixed, reused by both runs): results/m7b_3b_subset.json"
if [ "$MODE" = "full" ]; then
  echo "  wrote: results/val_llm_trace_3b_<dataset>.jsonl  (per-question)"
  echo "         results/m7b_3b_<tag>_full.json            (summary + paired CIs vs 1.5B)"
  echo "  a successful finish prints '3B - 1.5B' CIs for AllGold@4 and SP-Recall@4."
else
  echo "  wrote: results/m7b_3b_<tag>_test.json (timings + projection only)"
  echo "  re-run the real thing with: ./scripts/run_m7b_3b.sh full"
fi
exit 0
