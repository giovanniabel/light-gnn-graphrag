#!/bin/bash
# M11 step 4c: reader EM/F1 on the new 350, one process per (method, dataset).
# Frozen reader, frozen contexts read from results/ext_val_*.jsonl. Ollama ON.
set -u
cd "$(dirname "$0")/.."
export ALLOW_VALIDATION=1
export ALLOW_VALIDATION_REASON="M11 extension: reader on the new 350 (D27)"
export TOKENIZERS_PARALLELISM=false
for ds in musique hotpotqa; do
  for m in vector ppr llm_trace gnn gnn_rrf stacked oracle; do
    echo "--- $m / $ds ---"
    .venv/bin/python scripts/m8_reader.py "$m" "$ds" full validation_ext ext_read ext_val \
      2>&1 | grep -vE "it/s\]|^Loading weights|Redirects are currently|HF_TOKEN|FutureWarning|warnings.warn"
  done
done
.venv/bin/python -c "import sys; sys.path.insert(0,'.'); from src.ollama_client import unload_all; unload_all(); print('model unloaded')"
