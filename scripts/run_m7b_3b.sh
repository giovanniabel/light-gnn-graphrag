#!/usr/bin/env bash
# M7b: 3B strength check for LLM-Trace (agent.md Section 5, "Strength check").
#
#   scripts/run_m7b_3b.sh [test|full] [model_tag]
#
# Bounds how much the 1.5B tracer handicaps the LLM-Trace baseline. Fixed, seeded,
# stratified 60-question subset (30 per dataset). NEVER pulls a model: if the tag
# is missing it prints the pull command and exits 3.

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
MODE="${1:-test}"
set -a; [ -f .env ] && . ./.env; set +a
TAG="${2:-${STRENGTH_TAG}}"
PY="$ROOT/.venv/bin/python"

echo "=== M7b: 3B strength check (${MODE}) ==="
echo "  tag: ${TAG}"
echo "  installed models:"
ollama list | tail -n +2 | awk '{printf "    %s\n", $1}'
if ! ollama list | tail -n +2 | awk '{print $1}' | grep -qx "$TAG"; then
  echo "  -> ${TAG} is NOT installed. Required: ollama pull ${TAG}"
  echo "     Nothing will be pulled by this script."
fi
"$PY" -c 'import psutil;print(f"  swap {int(psutil.swap_memory().used/1048576)} MB | RAM available {int(psutil.virtual_memory().available/1048576)} MB")'

ALLOW_VALIDATION=1 ALLOW_VALIDATION_REASON="M7b 3B strength check (${MODE})" \
  caffeinate -i "$PY" scripts/m7b_3b_subset.py "$TAG" "$MODE"
