# CLAUDE.md

**Read [STATUS.md](STATUS.md) first.** It holds the current state of this project:
milestones completed, the settings that were chosen and why, open issues, and the
exact next step.

**[agent.md](agent.md) is the specification and the single source of truth.**
Read it in full before writing code. Build order is Section 11; the rules for
agents are Section 12; every departure from the spec is recorded in the
Section 14 deviation log (D1-D20) with its reason and the measurement behind it.

## Non-negotiables (agent.md Sections 2, 6.1, 12)

* **No cloud APIs.** Everything runs locally: Ollama for the SLM, bge-small for
  embeddings.
* **Never run Ollama generation and GNN training at the same time.** Unload the
  model (`ollama stop <tag>`) before any training phase. `scripts/run_phase.sh`
  does this per phase.
* **`configs/default.yaml` is the only place hyperparameters and seeds live.**
  Do not hardcode them. Use `src.config.set_config_value` to change one key --
  a full `save_config` rewrite strips every comment from the file.
* **Never tune on the eval split.** Model selection and ablations run on 5-fold CV
  over the training questions (D17); the 10% dev carve is only an inner early-stopping
  holdout. Reading the validation split requires `--allow-validation REASON` and is
  appended to `results/validation_access.log` — see `src/validation_guard.py`.
* **Never pool MuSiQue and HotpotQA into one headline number.**
* **Never measure memory from the Python process alone** (`src.profiling.tree_rss_mb`
  includes the Ollama processes; `own_tree_rss_mb` deliberately excludes them and is
  only for leak detection, never for reported metrics).
* **Keep the Section 10 module signatures stable.** If one needs to change, say so
  rather than changing it silently.
* Every phase must stay resumable, cached and time-capped. Re-running a finished
  phase is a no-op that takes seconds.

## Running things

```bash
scripts/run_phase.sh <phase> [--resume] [--max-hours H]   # caffeinate + ollama load/unload
.venv/bin/python -m pytest tests/ -q                      # the test suite
```

Phases: `m0 m1 pilot m3 m4 m5 m6 m7 m7b m8 m10` (see agent.md Section 11).
M6 model selection runs through `scripts/m6_cv.py <stage>`; M7a through
`scripts/m7a_run.py` (one subprocess per method, for clean per-process RSS).
Long runs belong in the background under `caffeinate -i`, on AC power.
