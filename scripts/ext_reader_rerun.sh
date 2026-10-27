#!/bin/bash
# M11 D30: re-read only the methods whose top-4 changed when the relation
# features were fixed. vector / ppr / llm_trace / oracle never go through
# to_pyg, so their contexts are byte-identical and their EM/F1 stand.
set -u
set -o pipefail
cd "$(dirname "$0")/.."
for ds in musique hotpotqa; do
  for m in gnn gnn_rrf stacked; do
    ALLOW_VALIDATION=1 \
    ALLOW_VALIDATION_REASON="M11 D30: re-read GNN-family contexts after the relation-feature fix" \
      .venv/bin/python scripts/m8_reader.py "$m" "$ds" full validation_ext ext_read ext_val 2>&1 \
      || { echo "FAILED: $m/$ds"; exit 1; }
  done
done
ollama stop qwen2.5:1.5b-instruct-q4_K_M 2>/dev/null
echo "model unloaded"
