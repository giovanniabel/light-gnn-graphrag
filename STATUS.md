# STATUS

**Updated:** 2026-10-22 · **Spec:** [agent.md](agent.md) (deviations in its §14 log)

M0–M8 are complete, plus the 3B strength check and the Pareto analysis. **The one-time validation run is done**; `results/validation_access.log`
records every read. **Final verdicts: C1 NOT SUPPORTED, C2 INCONCLUSIVE, C3 cost-half SUPPORTED /
quality-half SUPPORTED on HotpotQA and INCONCLUSIVE on MuSiQue, C4 SUPPORTED with
a stated RSS limitation.** Post-hoc analyses (edge-level C2 evidence, answer-level
EM/F1, the 3B check) are reported separately and labelled as such.

M0–M6 detail follows. The connectivity config is **frozen** (D12–D14), validation
ceilings were measured **once**, and all M6 model selection ran on **5-fold CV over
the training questions** (D17) — validation has still never been scored by a model.
165 tests pass.

**The central development-set finding (D18): the router learns edge scoring well
but it does not beat vector retrieval at picking paragraphs.** Graph-only GNN is
*significantly worse* than Vector RAG on AllGold@4 on both datasets, while being
*significantly better* than PPR and Edge-Sim on edge AP. C1–C4 verdicts remain
M10's job on validation; this is development evidence, and it is pointing the way
D15 predicted.

**Read open issue #1 before M7.** The CV results above are the measured version of
the ceiling problem recorded there.

---

## Milestones

| # | Milestone | State | Evidence |
| :-- | :-- | :-- | :-- |
| M0 | Env, pinned deps, device benchmark | **PASS** | `results/m0_report.json`, `results/device_benchmark.json` |
| M1 | `data.py`, `embedder.py`, `build_pool` | **PASS** | `results/m1_report.json` |
| M2 | Pilot + budget governor | **PASS** | `results/budget.json` |
| M3 | Embeddings cached for all pools | **PASS** | `results/m3_report.json` |
| M4 | Full extraction | **PASS** | `results/m4_report.json`, `results/m4_extract.log` |
| M5 | Graphs, labels, ceilings | **PASS** | `results/m5_report.json`, `results/m5_allgold_ceiling.json` |
| M5b | Connectivity fix (D12–D14), frozen | **PASS** (target missed, accepted) | `results/m5_connectivity_sweep.json`, `results/m5_before_after_train.json`, `results/m5_cap_sweep_train.json` |
| M5c | Validation ceilings, measured once | **PASS** (7/7 checks) | `results/m5_report.json` |
| M6 A | EdgeRouter training, main model, 3 seeds | **PASS** (6/6 checks) | `results/m6_stageA.json`, `results/checkpoints/` |
| M6 CV | 5-fold CV: references, grid, ablations, curve, final models | **done** | `results/m6_refs.json`, `m6_grid.json`, `m6_compare.json`, `m6_ablations.json`, `m6_curve.json`, `m6_final.json` |
| M7–M10 | Eval, reader, analysis | not started | |

115 tests pass (`.venv/bin/python -m pytest tests/ -q`): 25 data, 21 extractor, 13 profiling, 42 graph, 14 routing.

## Settings chosen

| Setting | Value | Why |
| :-- | :-- | :-- |
| `DEVICE` | **cpu** | benchmarked twice: at 200/400 caps cpu 0.382 vs mps 0.534 s/epoch; re-run at the D14 sizes (70 nodes/415 edges, 778/390 features) cpu **0.144** vs mps **0.154** (mps 0.94×). cpu both times; 431,745 params, inside §4.3's 1M budget |
| Connectivity | **(a)+(b)+(c) frozen** | title hubs + fuzzy (Jaccard 0.5), `same_as` at cosine ≥ 0.85, paragraph nodes. Chosen on train/dev only (D12) |
| Caps | **300 nodes / 800 edges** | D14, from 200/400: cap-binding 0.0% vs 67.4% |
| `H` | **3** (not raised) | D13: raising it would change the §4.3 feature width |
| Feature dims | **x 778, edge_attr 390** | adds a 3-dim multi-hot node-kind block and a 4-dim edge-kind one-hot |
| SLM | **`qwen2.5:1.5b-instruct-q4_K_M`** | §2; pulled, verified against `ollama list` |
| `max_triples` | **6** | all of {6,5,4} fit the time cap, so the densest option is free (5.5 vs 3.8 triples/paragraph) |
| `max_words_per_paragraph` | **120** | median paragraph is 99 words; keeps 98.8% of words, stops `num_predict` truncation (D2) |
| Pool sizes | **MuSiQue 14, HotpotQA 10** | MuSiQue needs 14 to clear §3.1's 85% recall gate (0.873/0.898); HotpotQA ships only 10 (D1) |
| `N` | **150** per (dataset, split) | governor allowed 157 at t_p=1.905 s, so §3's default target of 150 binds, not the budget |
| `prompt_version` | **`v2-mt6`** | JSON-schema grammar + 120 words + truncation salvage (D2); the `-mt` suffix keeps cache keys distinct (D8) |
| HotpotQA train | **`level=hard` only** | official validation is 100% hard; train is easy/medium/hard (D5) |
| `pos_weight` | **1.78–2.35** | measured positive-edge rate 29.9–36.0% per split (see `results/m5_report.json`) |

## What the data looks like — VALIDATION, frozen config, measured once

| MuSiQue / validation | 2-hop | 3-hop | 4-hop | **ALL** |
| :-- | --: | --: | --: | --: |
| Components per question | 2.1 | 5.3 | 3.9 | **3.8** |
| BFS share of full graph | 81.7% | 66.1% | 77.1% | **75.0%** |
| Pool coverage (evidence edges) | 84.3% | 71.6% | 81.6% | **79.1%** |
| **AllGold@4 ceiling (unconditional)** | 72.0% | 20.0% | 22.0% | **38.0%** |
| AllGold@4 ceiling (conditional on pooling) | 84.0% | 28.0% | 40.0% | 50.7% |
| All-gold-in-pool (Vector RAG's cap) | 88.0% | 80.0% | 54.0% | **74.0%** |
| Nodes / edges (mean) | 71 / 419 | 58 / 303 | 67 / 367 | 65 / 363 |
| Cap-binding | 0.0% | 0.0% | 0.0% | **0.0%** |
| Positive-edge rate / `pos_weight` | 4.4% / 21.9 | 5.2% / 18.2 | 5.8% / 16.1 | 5.1% / 18.6 |

| HotpotQA / validation | **ALL (2-hop)** |
| :-- | --: |
| Components per question | **1.1** |
| BFS share / pool coverage | 99.4% / **99.4%** |
| **AllGold@4 ceiling (uncond / cond)** | **99.3% / 99.3%** |
| All-gold-in-pool | 100.0% |
| Nodes / edges / cap-binding | 58 / 377 / **0.0%** |
| Positive rate / `pos_weight` | 5.6% / 16.9 |

Train, for comparison: MuSiQue coverage 85.0%, unconditional ceiling 48.7%;
HotpotQA 99.3% / 99.3%. On the 135-question train subset actually used for the
sweep (after the dev carve), **evidence-only** coverage is **84.8%** (MuSiQue) and
**99.4%** (HotpotQA). Every coverage figure in this file is evidence-only — it
counts a paragraph only if it has an extracted (`kind == triple`) edge in the
ego-graph. The earlier 88.3% counted scaffolding edges too and **is superseded**:
a paragraph reached only through its own hub is not rankable by the graph. Dropped train questions fell from 18/150 to **4/150**
(MuSiQue) and 7/150 to **0/150** (HotpotQA), because connectivity pulls gold
edges into the ego-graph that BFS previously could not reach.

**Graph ceiling conditional on all-gold-in-pool** (D4's conditioning: restrict the
question set), derived arithmetically from the figures above since
`covered ⊆ pool` makes it an exact ratio — read `n_cond` before the rate:

| | n | n_cond | uncond | in-pool | **conditional** |
| :-- | --: | --: | --: | --: | --: |
| MuSiQue val 2-hop | 50 | 44 | 72.0% | 88.0% | **81.8%** |
| MuSiQue val 3-hop | 50 | 40 | 20.0% | 80.0% | **25.0%** |
| MuSiQue val 4-hop | 50 | 27 | 22.0% | 54.0% | **40.7%** |
| **MuSiQue val ALL** | 150 | **111** | 38.0% | 74.0% | **51.4%** |
| HotpotQA val ALL | 150 | 150 | 99.3% | 100.0% | **99.3%** |

`pos_weight` is **18.6 over all edges** but **4.47 masked to evidence edges**
(measured on train), because §4.1.1 scaffolding is 71% of the graph (29% `triple`,
33% `title`, 5% `same_as`, 33% `para`) and is all-negative by construction.
**Which of these M6 uses is an open choice** — see §4.2.

## M6 stage A results (superseded by the CV results above; kept for the overfit test)

Overfit test: loss **0.798 → 0.000003** on 8 graphs / 978 evidence edges, edge
accuracy **1.0000**.

| dataset | seed | epochs | best ep | wall | dev AP | dev AllGold@4 | AllGold@4 cond (D4) |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| musique | 1234 | 8 | 3 | 2.9s | 0.5647 | 0.200 | 0.273 (11/15) |
| musique | 1235 | 7 | 2 | 2.5s | 0.5407 | 0.200 | 0.273 (11/15) |
| musique | 1236 | 6 | 1 | 2.3s | 0.5589 | 0.200 | 0.273 (11/15) |
| **musique mean** | | | | 2.6s | **0.5548** ±0.0102 | **0.200** ±0.000 | |
| hotpotqa | 1234 | 7 | 2 | 2.3s | 0.5213 | 0.333 | 0.333 (15/15) |
| hotpotqa | 1235 | 21 | 16 | 6.9s | 0.5757 | 0.533 | 0.533 (15/15) |
| hotpotqa | 1236 | 27 | 22 | 9.1s | 0.6203 | 0.400 | 0.400 (15/15) |
| **hotpotqa mean** | | | | 6.1s | **0.5724** ±0.0405 | **0.422** ±0.083 | |

Loss is BCE over **extracted (`kind == triple`) edges only**, `pos_weight` **4.447**
(MuSiQue) / **4.081** (HotpotQA) from training triple edges — all-edge values would
have been 17.66 / 16.94. Early stopping on dev edge-level AP, patience 5; dev
AllGold@4 logged per epoch for information only. MuSiQue dropped 3 of 135 training
questions having no positive evidence edge (§4.2); HotpotQA dropped 0.

**Training does real work**: dev AP rises from **0.22 untrained** to **0.55**,
against a random-scorer baseline of 0.187 (the positive-edge prevalence).

## M6 under cross-validation (D17) — all on TRAIN, validation never scored

Folds: 5, stratified (MuSiQue 10/10/10 per hop bucket; HotpotQA 24–25 bridge + 5–6
comparison), fixed by seed and identical for every config. Each config is 3 seeds ×
5 folds; metrics come from seed-averaged OOF scores. Inner 10% holdout for early
stopping; the fold's test questions never touch stopping. **Noise floor:
fold-to-fold edge-AP sd 0.031 (MuSiQue) / 0.038 (HotpotQA)** — read every delta
below against that.

### Method comparison (OOF, 150 train questions each)

**MuSiQue**

| method | edge AP | SP-Recall@4 | AllGold@4 | conditional (n) |
| :-- | --: | --: | --: | --: |
| gnn_graph_only | 0.5864 | 0.5739 | 0.153 | 0.225 (102/150) |
| gnn_plus_vector | — | 0.6633 | 0.220 | 0.324 (102/150) |
| vector_rag | — | 0.6922 | 0.227 | 0.333 (102/150) |
| ppr | 0.5338 | 0.5617 | 0.140 | 0.206 (102/150) |
| ppr_plus_vector | — | 0.6389 | 0.213 | 0.314 (102/150) |
| edge_sim | 0.4948 | 0.6122 | 0.200 | 0.294 (102/150) |

**HotpotQA**

| method | edge AP | SP-Recall@4 | AllGold@4 | conditional (n) |
| :-- | --: | --: | --: | --: |
| gnn_graph_only | 0.6872 | 0.7667 | 0.567 | 0.567 (150/150) |
| gnn_plus_vector | — | 0.8533 | 0.727 | 0.727 (150/150) |
| vector_rag | — | 0.8633 | 0.733 | 0.733 (150/150) |
| ppr | 0.5548 | 0.7000 | 0.513 | 0.513 (150/150) |
| ppr_plus_vector | — | 0.8233 | 0.660 | 0.660 (150/150) |
| edge_sim | 0.5915 | 0.7767 | 0.593 | 0.593 (150/150) |

### Paired bootstrap (10,000 resamples over questions)

| comparison | MuSiQue | HotpotQA |
| :-- | :-- | :-- |
| GNN − Vector RAG (AllGold@4) | -0.1067 [-0.1867, -0.0267] — **excludes 0** | -0.1667 [-0.2533, -0.0800] — **excludes 0** |
| GNN+Vec − PPR+Vec (AllGold@4) | +0.0200 [-0.0333, +0.0733] — includes 0 (within noise) | +0.0667 [+0.0000, +0.1333] — includes 0 (within noise) |
| GNN+Vec − Vector (AllGold@4) | -0.0067 [-0.0667, +0.0533] — includes 0 (within noise) | -0.0067 [-0.0667, +0.0533] — includes 0 (within noise) |
| **GNN − PPR (edge AP)** | +0.0526 [+0.0193, +0.0872] — **excludes 0** | +0.1324 [+0.0981, +0.1677] — **excludes 0** |
| **GNN − Edge-Sim (edge AP)** | +0.0916 [+0.0506, +0.1341] — **excludes 0** | +0.0957 [+0.0587, +0.1330] — **excludes 0** |

Two findings, pulling in opposite directions:

* **The router scores edges genuinely well.** It beats the non-learned graph
  baseline (PPR) and the no-graph baseline (Edge-Sim) on edge AP with CIs that
  exclude 0 on both datasets, and the `L=0` ablation costs −0.071/−0.098 AP. That
  is direct support for C2's "learning matters" half.
* **That edge quality does not convert into paragraph selection.** Graph-only GNN
  is significantly *worse* than Vector RAG on AllGold@4 (−0.107 and −0.167), and
  the matched RRF pair (GNN+Vector vs PPR+Vector) is positive but within noise
  (+0.020, +0.067 with a lower bound of exactly 0.000). Fusing with vector neither
  helps nor hurts versus vector alone (−0.007 both). This is the D15 cap showing
  up in measurements rather than in projections.

### Hyperparameter grid — no signal

8 configs spanned **0.0093** mean OOF edge AP against a fold sd of ~0.031, i.e.
**0.33–0.36× the noise**, and the per-dataset winners disagreed. Frozen config
**lr 1e-3, dropout 0.3, weight decay 1e-1** beats the default by **+0.0026**. This
is an arbitrary pick inside the noise and **must not be written up as tuned**
(D18a). The useful result is that this model is insensitive to these knobs.

### Ablations — three design elements are inert

Significant on edge AP, both datasets: **L=0** (−0.071/−0.098), **SAGE vs GATv2**
(−0.016/−0.031), **no_title** (−0.021/−0.031), **triples_only** (−0.017/−0.034).

Within noise on both: **L=3**, **no query conditioning** (−0.001/−0.001),
**H=2 vs H=3** (+0.016/−0.001), **no_para** (+0.000/+0.005), **no_same_as**.

**Every ablation is within noise on AllGold@4** — at n=150 that metric cannot
resolve any of them.

So: query conditioning, the H=3 radius, and the `para` links did not earn their
place. `para` is the striking one — it is 33% of all edges and the mechanism that
lifted coverage from 50% to 88%, yet removing it changes edge AP by nothing and
*improves* AllGold@4 point estimates. It bought ceiling, not quality, exactly as
the 60%-distractor-to-distractor link diagnostic suggested. Title hubs are the
scaffolding that does work.

### Learning curve — MuSiQue is saturated

| fraction of each fold's training set | MuSiQue AP | MuSiQue AllGold@4 | HotpotQA AP | HotpotQA AllGold@4 |
| :-- | --: | --: | --: | --: |
| 25% | 0.5660 | 0.153 | 0.6391 | 0.520 |
| 50% | 0.5834 | 0.173 | 0.6624 | 0.553 |
| 75% | 0.5823 | 0.167 | 0.6683 | 0.560 |
| 100% | 0.5864 | 0.153 | 0.6872 | 0.567 |

MuSiQue gains **+0.003** AP from 50% to 100% — flat. HotpotQA still rises (+0.048
across the whole range, +0.019 in the last quartile), but that is under one fold
sd. **More extraction is not worth it**: the MuSiQue bottleneck is the
entity-linking ceiling, not training volume (D18d).

### Final models

Frozen config, trained on the **full** training set (the former dev questions
included), 3 seeds per dataset, 6 checkpoints in `results/checkpoints_final/`.
MuSiQue 131 train + 15 stop-holdout; HotpotQA 135 + 15. **Validation was not
evaluated — that is M7.**

## M8 reader results, 3B check and final verdicts

### EM / F1 (n=150 per dataset, K=4 paragraphs, one fixed prompt, greedy, 24 tokens)

| condition | MuSiQue EM | MuSiQue F1 | HotpotQA EM | HotpotQA F1 |
| :-- | --: | --: | --: | --: |
| Vector RAG | **0.180** | **0.269** | 0.473 | **0.582** |
| PPR | 0.113 | 0.187 | 0.387 | 0.506 |
| LLM-Trace (1.5B) | 0.153 | 0.232 | 0.427 | 0.534 |
| GNN | 0.113 | 0.191 | 0.420 | 0.516 |
| GNN+Vector (RRF) | 0.173 | 0.252 | **0.480** | 0.577 |
| Stacked (D19) | 0.153 | 0.250 | 0.460 | 0.568 |
| **Gold oracle** | **0.367** | **0.464** | **0.560** | **0.697** |

**Oracle sanity check PASSED on both datasets** — the oracle is the ceiling on EM
and F1, as it must be. Vector RAG − oracle is −0.187 EM (MuSiQue) and −0.087
(HotpotQA), both CIs excluding zero, so substantial reader headroom remains and
the reader is not saturating the comparison.

By hop bucket, MuSiQue EM: 2-hop 0.220 / 3-hop 0.220 / 4-hop 0.100 for Vector RAG
against the oracle's 0.460 / 0.380 / 0.260. HotpotQA by type: bridge 0.484,
comparison 0.429 (Vector RAG).

### Stage validity — 8 of 33 measured stages are INVALID under §7.3 (D26)

| stage | swap at start | swap growth | verdict |
| :-- | --: | --: | :-- |
| M7 indexing (extraction) | 264 MB | +0 MB | valid |
| M7c LLM-Trace / MuSiQue | 0 MB | **+3,389 MB** | **INVALID** |
| M7c routing / MuSiQue | 1,590 MB | +0 MB | valid¹ |
| M7c LLM-Trace / HotpotQA | 1,582 MB | **+2,195 MB** | **INVALID** |
| M7c routing / HotpotQA | 1,624 MB | +0 MB | valid¹ |
| M8 reader, 5 of 14 stages | — | +507 … +1,587 MB | **INVALID** |
| M7a routing, all 14 stages | — | +0 MB | valid |
| 3B check / HotpotQA | — | +782 MB | **INVALID** |

¹ **Start-state caveat (D25):** the M7c gate checks only at script start. The run
began at 0 MB and passed, but every stage after the first began at ~1.6 GB because
the preceding LLM-Trace stage had already pushed the machine into swap. The +0 MB
growth verdict stands on the §7.3 rule as written, but these were not measured on a
clean machine.

**EM/F1 and every retrieval metric are unaffected by any of this.** The reader and
the traversal both decode greedily at `temperature=0`, so identical context gives
identical output regardless of timing or paging. Only latency, token-rate and RSS
figures from the invalid stages are void.

### The 4.2 ms vs 43.8 ms routing discrepancy — resolved (agent.md §7.3 clarified)

`scripts/m7c_stage.py` builds its QueryGraphs and PyG tensors **before** the timer,
so its "routing" figure is **model forward + aggregation only**:

| | MuSiQue | HotpotQA |
| :-- | --: | --: |
| M7a routing, full §7.3 scope | **43.8 ms** | **35.6 ms** |
| — of which graph assembly | 38.9 ms | 30.9 ms |
| — of which model forward | 4.3 ms | 4.3 ms |
| M7c "routing" (forward only) | 4.2 ms | 4.1 ms |

§7.3 defines routing as "everything from question + graph to ranked paragraphs",
which includes question embedding, seed linking and ego-graph/feature assembly —
all inside `build_graph()`. **C3 uses the M7a numbers**; the M7c figure is
relabelled and must not be quoted as routing latency.

### LLM-Trace efficiency: four measurements, all indicative (D24)

| measurement | MuSiQue median | HotpotQA median | swap growth | valid |
| :-- | --: | --: | --: | :-- |
| M7 first run | 1,573 ms | 1,764 ms | +2,532 / +3,072 | INVALID |
| 10-question smoke | 1,639 ms | 2,157 ms | +0 / +0 | valid (smallest n) |
| D21 redo | 1,573 ms | 1,764 ms | +2,532 / +3,072 | INVALID |
| M7c clean start | 1,543 ms | 1,684 ms | +3,389 / +2,195 | INVALID |

Tokens 1,050–1,405 per question. **C3 latency ratio**, valid M7a routing against
the indicative LLM-Trace range: MuSiQue **35.2×–37.4×**, HotpotQA **47.4×–60.7×**.

### Post-hoc analyses, labelled

**P1 — edge-level evidence for C2's substance.** Edge-Precision@10, four CIs all
excluding zero: GNN − PPR +0.112 [+0.059, +0.168] and +0.199 [+0.146, +0.254];
GNN − Edge-Sim +0.129 [+0.068, +0.190] and +0.063 [+0.010, +0.113]. C2 was
registered on AllGold@4, one metric level above where the effect lives.

**P2 — answer level.** `GNN+Vector RRF − GNN` is **+0.060 EM [+0.007, +0.113] on
both datasets** (post-hoc): fusion helps the reader. `GNN − Vector` EM on MuSiQue
is **−0.067 [−0.133, +0.000]** — **borderline, upper bound exactly 0.000**, and its
F1 counterpart −0.078 [−0.147, −0.009] does exclude zero. `Stacked − Vector` EM on
MuSiQue is likewise borderline at [−0.060, +0.000].

**P3 — 3B strength check (n=30 per dataset, directional only).** `3B − 1.5B` on
MuSiQue: all CIs include zero, so **a 3B tracer is not materially better there** —
which weakens the "weak tracer" explanation for C3's MuSiQue result. On HotpotQA
the 3B is better on SP-Recall@4 (+0.117 [+0.017, +0.233]) yet the GNN still beats
it (+0.300 [+0.100, +0.500] AllGold@4).

**44 paired bootstraps, none corrected for multiplicity.** ~2 false positives are
expected at 95% confidence; two CIs have a bound of exactly 0.000. Single CIs that
only just exclude zero are not findings.

### Pareto analysis

`notebooks/pareto_analysis.py` → `results/pareto_{musique,hotpotqa}.png` and
`results/pareto_table.json`. Three cost axes (routing latency, LLM tokens, peak
RSS) against AllGold@4 and reader EM, indexing and query cost shown separately and
combined, hollow markers for §7.3-invalid or indicative points, the graph ceiling
drawn as a dashed line. The combined-cost panel is the least flattering: graph
methods pay **~22.5 s/question** of indexing on MuSiQue against Vector RAG's
**~0.02 s**, for no gain in AllGold@4.

## M7: validation results (one-time run)

Environment at the gate: swap 264 MB, RAM 2,620 MB free, pressure normal, AC power,
Low Power Mode off, `ollama ps` empty (`results/m7_env.json`).

### Indexing cost, re-measured clean (replaces the D11-voided figures)

| | measured |
| :-- | --: |
| seconds per paragraph | **1.603** (median 1.590, p95 2.153) |
| LLM tokens per paragraph | **408** (300 prompt + 108 generated) |
| triples per paragraph | 5.48 |
| JSON failures | 0 / 100 |
| peak tree RSS incl. Ollama | 1,882 MB (idle 1,305, delta **+577**) |
| swap growth | **+0 MB** → stage **valid** under §7.3 |
| entity/relation embedding | **23.3 ms per question** |

### MuSiQue validation (n=150) — graph ceiling **0.380**, answer-in-graph **0.293**

| method | SP@4 | AllGold@4 | cond (n=111) | EdgeP@10 | PathRec |
| :-- | --: | --: | --: | --: | --: |
| Vector RAG | 0.6333 | 0.267 | 0.252 | — | — |
| PPR | 0.5506 | 0.240 | 0.234 | 0.4933 | 0.0400 |
| Edge-Sim | 0.5517 | 0.213 | 0.207 | 0.4767 | 0.0733 |
| GNN (graph-only) | 0.5500 | 0.240 | 0.252 | 0.6053 | 0.0600 |
| GNN+Vector (RRF) | 0.6239 | 0.273 | 0.261 | 0.6053 | 0.0600 |
| PPR+Vector (RRF) | 0.6167 | 0.267 | 0.252 | 0.4933 | 0.0400 |
| GNN+hybrid-tail *(≡ GNN by construction)* | 0.5500 | 0.240 | 0.252 | 0.6053 | 0.0600 |
| PPR+hybrid-tail *(≡ PPR by construction)* | 0.5506 | 0.240 | 0.234 | 0.4933 | 0.0400 |
| Stacked (D19) | 0.6533 | 0.293 | 0.279 | 0.6053 | 0.0600 |
| LLM-Trace (1.5B) | 0.6433 | 0.300 | — | — | — |
| **ceiling (graph methods)** | — | **0.380** | — | — | **0.293** |

### HotpotQA validation (n=150) — graph ceiling **0.993**, answer-in-graph **0.500**

| method | SP@4 | AllGold@4 | cond (n=150) | EdgeP@10 | PathRec |
| :-- | --: | --: | --: | --: | --: |
| Vector RAG | 0.8933 | 0.793 | 0.793 | — | — |
| PPR | 0.7067 | 0.533 | 0.533 | 0.5493 | 0.1400 |
| Edge-Sim | 0.8033 | 0.613 | 0.613 | 0.6860 | 0.1867 |
| GNN (graph-only) | 0.7500 | 0.567 | 0.567 | 0.7487 | 0.1533 |
| GNN+Vector (RRF) | 0.8433 | 0.713 | 0.713 | 0.7487 | 0.1533 |
| PPR+Vector (RRF) | 0.8367 | 0.693 | 0.693 | 0.5493 | 0.1400 |
| GNN+hybrid-tail *(≡ GNN by construction)* | 0.7500 | 0.567 | 0.567 | 0.7487 | 0.1533 |
| PPR+hybrid-tail *(≡ PPR by construction)* | 0.7067 | 0.533 | 0.533 | 0.5493 | 0.1400 |
| Stacked (D19) | 0.8900 | 0.793 | 0.793 | 0.7487 | 0.1533 |
| LLM-Trace (1.5B) | 0.6200 | 0.367 | — | — | — |
| **ceiling (graph methods)** | — | **0.993** | — | — | **0.500** |

### Efficiency, routing stage (ms; RSS in MB)

**MuSiQue**

| method | median | p95 | assembly | forward | peak RSS | ΔRSS | tokens |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| Vector RAG | 0.2 | 7.7 | 0.0 | 0.22 | 300 | +42 | 0 |
| PPR | 40.3 | 70.8 | 39.0 | 1.04 | 801 | +543 | 0 |
| Edge-Sim | 583.5 | 942.1 | 34.9 | 548.02 | 1618 | +1359 | 0 |
| GNN (graph-only) | 43.8 | 83.3 | 38.9 | 4.34 | 852 | +599 | 0 |
| GNN+Vector (RRF) | 44.0 | 88.2 | 37.8 | 4.91 | 872 | +618 | 0 |
| PPR+Vector (RRF) | 38.7 | 70.9 | 35.8 | 1.60 | 822 | +569 | 0 |
| GNN+hybrid-tail *(≡ GNN by construction)* | 44.5 | 89.7 | 38.7 | 4.89 | 875 | +622 | 0 |
| PPR+hybrid-tail *(≡ PPR by construction)* | 40.1 | 122.7 | 36.3 | 1.78 | 818 | +564 | 0 |
| Stacked (D19) | 48.0 | 87.5 | 39.7 | 6.50 | 882 | +629 | 0 |
| LLM-Trace (1.5B) | 1573 | 2051 | — | — | 3234 | +1411 | 1101 |

**HotpotQA**

| method | median | p95 | assembly | forward | peak RSS | ΔRSS | tokens |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| Vector RAG | 0.5 | 7.5 | 0.0 | 0.51 | 293 | +39 | 0 |
| PPR | 27.5 | 69.5 | 26.5 | 0.87 | 791 | +537 | 0 |
| Edge-Sim | 674.3 | 944.4 | 26.4 | 638.21 | 1730 | +1476 | 0 |
| GNN (graph-only) | 35.6 | 105.6 | 30.9 | 4.28 | 850 | +598 | 0 |
| GNN+Vector (RRF) | 38.2 | 101.6 | 30.5 | 5.73 | 865 | +613 | 0 |
| PPR+Vector (RRF) | 29.0 | 75.4 | 26.6 | 1.70 | 810 | +558 | 0 |
| GNN+hybrid-tail *(≡ GNN by construction)* | 37.1 | 94.6 | 30.7 | 5.39 | 869 | +617 | 0 |
| PPR+hybrid-tail *(≡ PPR by construction)* | 29.1 | 76.6 | 26.7 | 1.64 | 813 | +561 | 0 |
| Stacked (D19) | 38.3 | 95.2 | 30.4 | 6.68 | 871 | +619 | 0 |
| LLM-Trace (1.5B) | 1764 | 2143 | — | — | 2472 | +660 | 1232 |

All nine LLM-free methods recorded **+0 MB swap growth** — no stage invalidated.
**LLM-Trace's efficiency stage IS invalid under §7.3** (swap +2,532 / +3,072 MB even
after being redone one dataset per process; see D21). Its latency is reported as
indicative: three independent runs agree within 13–23% (1.57–1.77 s MuSiQue,
1.76–2.16 s HotpotQA), and the smoke run with **zero** swap growth sits inside that
range, so the ~35–50× C3 margin is not an artefact. Edge-Sim's latency was **re-measured** after preloading
its edge-text embeddings before the timed stage: **583.5 → 10.6 ms** (MuSiQue) and
**674.3 → 6.2 ms** (HotpotQA). The original figure was ~98% sharded-cache I/O, not
the method. Its metrics are byte-identical across the two runs, as a deterministic
method must be; only the timing changed. The tables below carry the corrected
latency.


### Follow-up CIs computed from the saved per-question files (no model re-run)

**GNN vs LLM-Trace.** The GNN is significantly *worse* on SP-Recall@4 on MuSiQue
(−0.093 [−0.143, −0.046] uncond; −0.071 [−0.123, −0.017] cond) and significantly
*better* on HotpotQA (+0.130 [+0.067, +0.190]). On AllGold@4 the MuSiQue difference
includes zero and straddles C3's margin; HotpotQA is +0.200 [+0.107, +0.293].

**GNN vs the graph baselines on Edge-Precision@10 — four CIs, all excluding zero:**

| | MuSiQue | HotpotQA |
| :-- | :-- | :-- |
| GNN − PPR | **+0.112 [+0.059, +0.168]** | **+0.199 [+0.146, +0.254]** |
| GNN − Edge-Sim | **+0.129 [+0.068, +0.190]** | **+0.063 [+0.010, +0.113]** |
| GNN − PPR (Path-Recall) | +0.020 [−0.007, +0.053] | +0.013 [−0.033, +0.060] |
| GNN − Edge-Sim (Path-Recall) | −0.013 [−0.047, +0.020] | −0.033 [−0.087, +0.013] |

**How this changes C2.** The pre-registered verdict **does not change — it stays
INCONCLUSIVE** — because §1.1 pins C2 to AllGold@4, where GNN − PPR includes zero
on both datasets. But the claim's *substance* ("graph structure and learning both
matter") now has direct validation support at the edge level: the GNN beats the
non-learned graph scorer **and** the no-graph edge scorer on Edge-Precision@10 on
both datasets, with all four CIs excluding zero, alongside the CV `L=0` result
(−0.071/−0.098 edge AP). The honest summary is that **C2 was registered on the
wrong metric for what it asserts**: the effect is real and measurable one level
below where the claim looks for it. Path-Recall shows no difference between any
graph method.

**Edge AP is not reported on validation**: per-edge scores and labels were not
saved, and the GNN's would need a model forward, so it is not recoverable from the
saved files. The CV edge-AP figures stand as the train-side evidence.

### C1–C4 verdicts (pre-registered §1.1; C1 has D15's two rows, tested on D4-conditional)

| claim | verdict | evidence |
| :-- | :-- | :-- |
| **C1** | **NOT SUPPORTED** | (a) graph-only vs Vector RAG, conditional: MuSiQue **+0.000** [−0.090, +0.090] (a dead tie, n=111); HotpotQA **−0.227** [−0.320, −0.127], significantly worse. (b) matched pair GNN+Vec − PPR+Vec: +0.009 [−0.018, +0.045] and +0.020 [−0.040, +0.080] — no gain whose CI excludes zero. |
| **C2** | **INCONCLUSIVE as registered** (see below) | GNN − PPR on AllGold@4 includes zero on both (MuSiQue +0.018 [−0.036, +0.072] cond; HotpotQA +0.033 [−0.040, +0.107]). But Edge-Precision@10 is 0.605 vs 0.493 and 0.749 vs 0.549, and CV edge-AP CIs excluded zero, with `L=0` costing −0.071/−0.098. Learning clearly helps **edge scoring**; it does not reach the AllGold@4 metric C2 was registered on. |
| **C3** | **SUPPORTED on HotpotQA, INCONCLUSIVE on MuSiQue** (corrected, D23) | Half 1 (cost): **35.9×** and **49.6×** cheaper than LLM-Trace, **0** query-time LLM tokens by construction — passes on both. Half 2 (quality), testing the CI against the −3-point margin rather than the point estimate: MuSiQue **−0.060 [−0.133, +0.007]** unconditional and **−0.045 [−0.126, +0.036]** conditional (n=111) both **straddle** −0.03 → **INCONCLUSIVE**; HotpotQA **+0.200 [+0.107, +0.293]** lies entirely above the margin → **SUPPORTED**. |
| **C4** | **SUPPORTED** | Peak process-tree RSS incl. Ollama **3,234 MB (3.16 GB)** < 6 GB, worst case being LLM-Trace. Covers indexing and routing only — the reader (M8) has not run. |

**A 1.5B tracer is a weak LLM-Trace baseline** (§13), and the M7b 3B check that
would bound how much is the tracer versus the method was **not run**. C3's MuSiQue
failure should be read with that in mind: the comparison it loses is against a
baseline the spec itself flags as handicapped.

**The stacker's OOF-vs-final mismatch, stated precisely.** The final GNN models
never saw any validation question, so **their validation scores are out-of-sample
and the stacked row is a legitimate out-of-sample result.** The mismatch is
narrower than that: the D19 combiner's `gnn_score` feature was *fit* on
out-of-fold scores from models trained on **~108 of 135** training questions (4/5
folds minus the inner 10% early-stopping holdout), but at validation time that
feature is produced by the **final** models, trained on **131 of 135**. So the
feature the combiner sees at test time comes from slightly better-trained models
than the ones it was calibrated against, and may be better calibrated or sharper.
**The direction of that effect is not determined** — a weight fitted to noisier
OOF scores could under- or over-weight sharper final-model scores — so no claim is
made about which way it biases the stacked row. Two things bound it: the features
are min-max normalized per question, which absorbs scale shifts, and the gap is
108 vs 131 training questions, not OOF vs in-sample.

**Two variants are inert by construction:** `GNN+hybrid-tail` is identical to `GNN`
on every metric (likewise PPR), because the hybrid tail only reorders paragraphs
ranked *below* every graph-scored one, and with K=4 and ≥4 scored paragraphs on
nearly every question it can never alter the top-4.

**4-hop MuSiQue is exploratory (n=50):** every method lands at 0.04–0.08 against a
0.220 ceiling. The clearest single cell in the whole run is **3-hop MuSiQue, where
Vector RAG scores 0.260 against a graph ceiling of 0.200** — it beats the bound
that constrains every graph method, because it is not bounded by reachability.

## Methods that will be scored on validation (M7)

This is the complete list. Every variant that was evaluated appears here,
including the ones that lost on development data (the rule in §4.1.1 / D19).
Nothing below has touched validation yet — `results/validation_access.log` is
empty and the guard refuses the read without `--allow-validation`.

| # | method | kind | status |
| :-- | :-- | :-- | :-- |
| 1 | **Vector RAG** | baseline, no LLM | primary baseline |
| 2 | **PPR** | non-learned graph | §5 |
| 3 | **Edge-Sim** | no-graph edge ranker | §5 |
| 4 | **GNN Router (graph-only)** | ours | C1 primary (a), shown next to its ceiling |
| 5 | **GNN+Vector (RRF, k=60)** | matched hybrid | C1 primary (b) |
| 6 | **PPR+Vector (RRF, k=60)** | matched control | C1 primary (b) |
| 7 | **GNN+hybrid-tail** | secondary variant | **identical to GNN by construction** — inert at K=4 |
| 8 | **PPR+hybrid-tail** | secondary variant | **identical to PPR by construction** — inert at K=4 |
| 9 | **Stacked hybrid (D19)** | learned combiner | reported though it lost on CV |
| 10 | **LLM-Trace (1.5B)** | needs Ollama | **DONE** (efficiency stage §7.3-invalid, D21) |
| 11 | **LLM-Trace (3B)** | needs Ollama | **not run** — M7b optional, 60 questions |
| 12 | **Gold-paragraph oracle** | reader ceiling | **not run** — M8 |

**Not implemented, deliberately:** D20 (paragraph-aware GNN) — its gate closed.

Checkpoints ready: 6 GNN models in `results/checkpoints_final/`
(3 seeds × 2 datasets) and 2 stacker combiners (`stacker_musique.pkl`,
`stacker_hotpotqa.pkl`).

## D19 stacked hybrid — evaluated on train CV, does not beat Vector RAG

Logistic regression over paragraph rows, features min-max normalized per
question, fit under a second CV over the same fixed folds so the combiner never
scores a question it was fit on; the GNN feature is already out-of-fold.

| | MuSiQue | HotpotQA |
| :-- | --: | --: |
| SP-Recall@4 | 0.6883 | 0.8767 |
| AllGold@4 (uncond) | 0.213 | 0.760 |
| AllGold@4 (D4 cond) | 0.314 (102/150) | 0.760 (150/150) |
| coefficient: **vector cosine** | **+3.08** | **+4.72** |
| coefficient: GNN score | +0.75 | +1.94 |
| coefficient: PPR score | +0.27 | −0.29 |

| paired bootstrap | MuSiQue | HotpotQA |
| :-- | :-- | :-- |
| vs Vector RAG (AllGold@4) | -0.0133 [-0.0533, +0.0200] — includes 0 | +0.0267 [-0.0200, +0.0733] — includes 0 |
| vs RRF pair (AllGold@4) | +0.0133 [-0.0467, +0.0733] — includes 0 | +0.1000 [+0.0333, +0.1667] — **excludes 0** |
| vs RRF pair (SP-Recall@4) | +0.0494 [+0.0139, +0.0861] — **excludes 0** | +0.0533 [+0.0200, +0.0867] — **excludes 0** |

Two things follow. **A learned combiner beats fixed-weight RRF** — significantly
on SP-Recall@4 on both datasets — so D15's untuned `k=60` does leave value on the
table. And **the graph score carries real signal**: the combiner gives it a weight
of +0.75 / +1.94, not zero. But vector cosine outweighs it 4.1× and 2.4×, and the
stack still does not beat vector similarity alone. That is the answer to the
question D19 was built to ask.

**D20 was therefore skipped**, which is what its gate specified: no CI excluded
zero and the point estimates were not positive on both datasets.

## Open issues

### 1. C1 on MuSiQue is still not a fair fight on the unconditional metric

Connectivity did what it was meant to. MuSiQue train: components 13.5 → 2.3, BFS
reach 23.4% → 82.9%, evidence-only coverage 29.4% → **84.8%** (135-question train
subset; 85.0% over all 150), AllGold@4 ceiling 32.6% → 48.7%
unconditional. HotpotQA is essentially solved: 1.1 components, 99.4% coverage,
**99.3%** ceiling, from 65.2%.

But on MuSiQue **validation** the unconditional AllGold@4 ceiling is **38.0%**,
while Vector RAG — which scores every pool paragraph by cosine and needs no graph —
is capped only by all-gold-in-pool at **74.0%**. A perfect router still loses the
unconditional comparison on MuSiQue by construction. D4 anticipates this by
testing C1 on the *conditional* metric, which is the right call; see #2 for a
definition problem in exactly that metric.

Two specifics worth carrying into M6/M7 rather than rediscovering:

* **MuSiQue 3-hop validation is an outlier.** 5.3 components against 2.1 on train,
  coverage 71.6% against 86.6%, unconditional ceiling 20.0% against 48.0%. Its
  all-gold-in-pool is a healthy 80.0%, so this is a connectivity failure specific
  to 3-hop validation questions, not a pooling failure. Nothing was changed in
  response — the config was already frozen when this was measured.
* **MuSiQue 4-hop is pool-limited, not graph-limited**: all-gold-in-pool 54.0% is
  below even the 3-hop ceiling, so for 4-hop the binding constraint is §3.1 pool
  construction, not §4.1.1 reachability.

### 2. The dev split problem — RESOLVED by D17 cross-validation

Dev is 15 questions (§3's 10% carve). AllGold@4 therefore takes only 16 distinct
values, which is why all three MuSiQue seeds report exactly 0.200: they produce
genuinely *different* rankings (top-4 agrees on only 5/15 between seeds, and
matches raw pool order on 0/15), they just land on the same 3/15 count. Early
stopping runs on dev **AP** for this reason, being continuous, but AP on 15
questions is still noisy — MuSiQue's best epoch is 1–3 of 6–8 run, so the stopping
point comes off a short, jittery curve. This was resolved by replacing dev with 5-fold CV (D17). The concern was correct:
with CV's proper noise floor measured at ~0.031–0.038 edge-AP sd, the whole
hyperparameter grid and six of ten ablations landed inside it.

### 3. The "conditional" metric mismatch — RESOLVED

M5 reports `allgold_ceiling_conditional` as *P(gold ∩ pool ⊆ covered)* — it
conditions on pooling by **shrinking the gold set** to the part that survived
pooling. D4 defines C1's conditional AllGold@4 differently, as
*P(gold ⊆ selected | gold ⊆ pool)* — conditioning by **restricting the question
set** to those whose gold all survived.

These are not the same quantity, and the one C1 is to be tested on is D4's. The
mismatch was found while reconciling the ceilings (issue #3 below) and is reported
rather than fixed, because the config was frozen and validation already measured.
Both now done: `metrics.all_gold_conditional` implements D4's definition
(restricting the question set, returning `n_conditional` so the rate cannot be
quoted without its denominator), and the M5 key is renamed
`allgold_ceiling_conditional_on_pooling`. M6 reports the D4 metric on dev.

### 4. Ceiling reconciliation — resolved, and it was a real bug

The earlier report's 77.8% graph ceiling exceeded all-gold-in-pool (~68%), which
is impossible for two unconditional quantities. Recomputed on the same 135
MuSiQue train questions:

| | | |
| :-- | --: | :-- |
| A all-gold-in-pool | 67.4% | unconditional |
| B graph ceiling | 77.8% | **conditional on pooling** — what was reported |
| C graph ceiling | **54.1%** | **unconditional — the real cap** |

`C ≤ A` and `C ≤ B` both hold, so nothing was impossible: **B was the bug**.
`measure()` read `gold_pids` off the already-pooled `Example`, silently ignoring
gold paragraphs that pooling had discarded. All three are now reported with
explicit names and M5 asserts `C ≤ A` as an acceptance check.

### 5. M4 timings### 2. M4 timingsare not a valid efficiency measurement (D11)

t_p ≈ 1.90 s and the 3.30 h wall were measured while system swap grew to
**+8,239 MB**, far past §7.3's 500 MB invalidation rule. They are a planning
figure only. The §7.3 indexing cost and the C3 comparison need a dedicated
measurement run on an idle machine.

### 6. 3-hop MuSiQue remains the weakest cell (D16)

3-hop answer-in-graph is 12–42% and the 3-hop AllGold@4 ceiling is 8% on
validation. An earlier n=20 probe suggested 14% at every `max_triples`; on the
full 150-question split it is not that bad, but 3-hop remains worse than both
2-hop and 4-hop. Report it with C1 rather than discovering it at M10.

### 7. Minor

* M3's "bge freed" check passes trivially on a fully cached re-run, because the
  encoder is never loaded. The free path itself was verified separately (382 MB
  loaded, 84 MB returned to the OS, remainder held by torch's allocator).
* `EmbeddingCache` keeps every loaded shard resident, so own-process RSS after
  M5's embedding pass is ~761 MB. Intentional, but it means M6 should construct
  the cache lazily if memory gets tight.

## Exact next step

M6 is complete. Next is **M7** (retrieval eval on validation + LLM-Trace), which is the first phase allowed to score the validation split. Original M6 note follows — `src/gnn_router.py`: `EdgeRouter`, `train_router`, `route`; overfit
  test on 8 graphs to near-zero loss; 3 seeds (1234/1235/1236) plus the §5
  ablations; dev AllGold@4 logged per epoch; each run ≤ 5 min.

```bash
ollama stop qwen2.5:1.5b-instruct-q4_K_M    # M6 must not run alongside Ollama (§6.1)
scripts/run_phase.sh m6
```

Inputs M6 needs are already on disk: `data/cache/graphs/{dataset}_{split}_H3.pt`
(600 serialized PyG graphs), with `x[N, 778]`, `edge_attr[E, 390]`, `y_edge[E]`,
`q[1, 384]`, `seed_mask[N]`, `edge_pid`, `edge_kind` and `evidence_mask[E]`.

Two choices M6 must make deliberately rather than by default:

1. **`pos_weight`**: 18.6 over all edges, or **4.47** with the loss masked to
   `evidence_mask` (both train-only, in `results/m5_report.json`). Masking matches
   what paragraph scoring consumes, since scaffolding is 71% of edges and
   all-negative by construction.
2. **Ablations**: `L ∈ {0,1,2,3}` must reuse these exact tensors — §5 forbids
   rebuilding features, which is what keeps `L=0` an honest message-passing
   ablation — plus the four edge-type ablations (`no_para`, `no_same_as`,
   `no_title`, `triples_only`).
