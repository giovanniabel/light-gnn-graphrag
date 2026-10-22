# RUNBOOK — running the remaining phases from a plain Terminal

Everything below runs **from the project root** in an ordinary Terminal window.
No Claude session, no pre-activated virtualenv, no exported variables. Each script
resolves the repo from its own location, uses `.venv/bin/python` directly, loads
`.env` itself, and starts the Ollama server if it is not already running.

```bash
cd <project-root>
```

Verified by running all three with a **completely emptied environment**
(`env -i`) from a different directory.

Each script takes `test` or `full`:

* `test` — 10 questions per unit, prints a projected total, writes only timing
  files. Safe to repeat.
* `full` — the real run.

**Run them on AC power.** Every script warns if you are on battery (`agent.md` §2).

---

## 1. `run_m7c_efficiency.sh` — valid LLM-Trace / routing efficiency

Re-measures the two efficiency stages that §7.3 invalidated on swap (deviation
**D21**). This is the only outstanding *measurement* gap in M7.

```bash
./scripts/run_m7c_efficiency.sh test     # ~2 min, sanity check
./scripts/run_m7c_efficiency.sh full     # ~10 min
```

**Expected duration.** `test` ≈ 2 min. `full` ≈ 10 min (LLM-Trace ≈ 4 min on
MuSiQue + 5 min on HotpotQA; the GNN/Vector routing stages are seconds).

**What it writes.** Four files per mode:

```
results/m7c_llm_trace_musique_<mode>.json
results/m7c_llm_trace_hotpotqa_<mode>.json
results/m7c_routing_musique_<mode>.json
results/m7c_routing_hotpotqa_<mode>.json
```

**A successful finish looks like** `=== M7c DONE (full) ===`, the four filenames
listed, and for each stage a line ending `swap growth +N MB -> valid`.

> **Check `"invalidated_by_swap": false` in every file before quoting its
> numbers.** If any says `true`, that stage is void under §7.3 — exactly what
> happened the first time — and the fix is to free memory and re-run, not to use
> the number.

**If the swap gate aborts** (`ABORT: swap NNN MB exceeds the 300 MB gate`):

1. Nothing was measured and nothing was written. No earlier result is touched.
2. Quit memory-heavy apps — browsers, Slack, the Claude app — and re-run.
3. If swap stays high, **reboot**. macOS does not reclaim swap eagerly, so a
   machine that has been under pressure keeps a large swap file even when idle.
   Re-run before opening anything else.
4. Check swap with:
   ```bash
   .venv/bin/python -c 'import psutil;print(psutil.swap_memory())'
   ```
5. The gate is deliberately stricter (300 MB) than the §7.3 invalidation rule
   (500 MB growth), because starting near the limit all but guarantees tripping it.

---

## 2. `run_m8_reader.sh` — M8, reader EM/F1

Runs the reader over the **saved** M7 routings, so no router is re-run. Seven
conditions × two datasets: `vector ppr llm_trace gnn gnn_rrf stacked oracle`.

```bash
./scripts/run_m8_reader.sh test     # ~3 min, prints the estimated total
./scripts/run_m8_reader.sh full     # ~17 min
```

**Expected duration.** `test` ≈ 3 min. `full` ≈ **17 min** (measured estimate from
the 10-question test: 0.24–0.71 s per question × 150 × 14 condition-dataset pairs).
Well inside a 2-hour budget, so no reduction is needed.

**What it writes.**

```
results/read_<method>_<dataset>.jsonl   # per question: prediction, gold, EM, F1, latency, tokens
results/read_<method>_<dataset>.json    # per condition: EM, F1, latency, tokens, peak RSS
```

**A successful finish looks like** `=== M8 DONE (full) ===` followed by the EM/F1
table and, under each condition, paired-bootstrap CIs against **Vector RAG** and
against the **gold oracle**.

**Resumable.** Re-running `full` skips any question already in the `.jsonl`. If it
is interrupted, just run the same command again.

**Reader prerequisite.** `qwen2.5:1.5b-instruct-q4_K_M` must be installed (it is).
The script checks and exits 1 with the `ollama pull` command if not.

**Swap.** M8 has no hard gate — the reader is one short call per question and the
10-question test showed `+0 MB` growth throughout. Each condition still records
`invalidated_by_swap`; check it before quoting the latency column.

---

## 3. `run_m7b_3b.sh` — 3B strength check (needs a download first)

Bounds how much the 1.5B tracer handicaps LLM-Trace (`agent.md` §5). Uses a
**fixed, seeded, stratified 60-question subset** (30 per dataset; MuSiQue 10/10/10
by hop) saved at `results/m7b_3b_subset.json` and shared with the 1.5B run, so the
comparison is paired.

> **PREREQUISITE — not installed yet.** `ollama list` currently shows
> `qwen2.5:1.5b-instruct-q4_K_M`, `dolphin-phi:2.7b`, `llama3.2:3b` and
> `mxbai-embed-large`. The tag this check needs is **absent**:
>
> ```bash
> ollama pull qwen2.5:3b-instruct-q4_K_M      # about 1.9 GB
> ```
>
> The script **never pulls anything**. Until you pull it, it exits **3** and prints
> that command. `llama3.2:3b` *is* installed but is a different model family, so
> using it would confound family with size and would not isolate the tracer's
> capability — `agent.md` §2 specifies the qwen2.5 3B tag for this reason.

```bash
./scripts/run_m7b_3b.sh test     # ~1 min once the model is pulled
./scripts/run_m7b_3b.sh full     # ~4 min
```

**Expected duration.** `test` ≈ 1 min. `full` ≈ 4 min (60 questions; the 3B model
is roughly 2× slower per call than the 1.5B).

**What it writes.**

```
results/m7b_3b_subset.json                        # the fixed subset (already written)
results/val_llm_trace_3b_<dataset>.jsonl          # per question (full only)
results/m7b_3b_qwen2.5_3b-instruct-q4_K_M_full.json
```

**A successful finish looks like** `=== M7b 3B DONE (full) ===` and, per dataset, a
`3B - 1.5B` paired CI for AllGold@4 and SP-Recall@4. A CI excluding zero means the
1.5B tracer was materially handicapping the LLM-Trace baseline, which must then be
said plainly next to the C3 verdict.

**If it exits 3**, the model is missing — pull it and re-run. Nothing is written
and nothing is changed.

---

## Order I would run them in

1. **`run_m7c_efficiency.sh full`** — first, and on a freshly rebooted machine.
   It is the one thing that closes a known gap (D21), and it is the most sensitive
   to memory state.
2. **`run_m8_reader.sh full`** — independent of M7c and swap-tolerant.
3. **`run_m7b_3b.sh full`** — only after pulling the 3B model, and only if you
   want the C3 caveat quantified.

## Afterwards

```bash
.venv/bin/python -m pytest tests/ -q        # 174 tests
cat results/validation_access.log           # every validation read, with its reason
git status --short                          # results/*.json are tracked; .pt files are not
```

`validation_access.log` should contain only `UNLOCK` and `ALLOWED` lines. **Any
`DENIED` line means something tried to read the validation split without being
authorised** — worth investigating before trusting that run.
