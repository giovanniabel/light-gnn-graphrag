"""M7a orchestrator: run every LLM-free method on validation, each in a fresh process.

Unlocks validation once (logged), spawns one subprocess per (method, dataset) so
each gets its own interpreter and idle RSS baseline, then aggregates.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT, load_config  # noqa: E402
from src.ollama_client import unload_all  # noqa: E402
from src.validation_guard import allow  # noqa: E402

METHODS = ["vector", "ppr", "edge_sim", "gnn", "gnn_rrf", "ppr_rrf",
           "gnn_hybrid_tail", "ppr_hybrid_tail", "stacked"]
DATASETS = ["musique", "hotpotqa"]
REASON = "M7 one-time validation run"


def main(split: str = "validation", prefix: str = "val",
         reason: str | None = None) -> int:
    reason = reason or REASON
    unload_all()
    ps = subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout
    rows = [l for l in ps.strip().splitlines()[1:] if l.strip()]
    print(f"--- M7a: validation retrieval, Ollama OFF ---")
    print(f"  `ollama ps` -> {len(rows)} loaded {'(EMPTY)' if not rows else rows}")
    if rows:
        print("  refusing to run with a model loaded (§6.1)")
        return 1
    allow(reason)          # logged to results/validation_access.log
    print(f"  validation UNLOCKED: {reason!r}  split={split}\n")

    env = {**os.environ, "ALLOW_VALIDATION": "1", "ALLOW_VALIDATION_REASON": reason,
           "TOKENIZERS_PARALLELISM": "false"}
    t0 = time.perf_counter()
    failures = []
    for ds in DATASETS:
        for m in METHODS:
            r = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "m7a_method.py"),
                 m, ds, split, prefix],
                env=env, capture_output=True, text=True, cwd=str(ROOT),
            )
            out = [l for l in r.stdout.splitlines()
                   if l.strip() and "Warning" not in l and "warn" not in l]
            for l in out:
                print(l, flush=True)
            if r.returncode != 0:
                failures.append((m, ds, r.stderr.strip().splitlines()[-3:]))
                print(f"  !! {m}/{ds} FAILED: {r.stderr.strip().splitlines()[-1:]}", flush=True)
    print(f"\n  {len(METHODS)*len(DATASETS)} runs in {time.perf_counter()-t0:.0f}s")
    if failures:
        print(f"  FAILURES: {[(m,d) for m,d,_ in failures]}")
        for m, d, err in failures:
            print(f"    {m}/{d}: {err}")
        return 1
    print(f"\n  per-question outputs: results/{prefix}_<method>_<dataset>.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:4]))
