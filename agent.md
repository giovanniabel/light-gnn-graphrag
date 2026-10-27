# AGENT SPECIFICATION: Resource-Efficient Subgraph Routing via Light GNNs (GraphRAG)

> Read this whole file before writing code. Follow the build order in Section 11. Do not start a phase until the previous phase's acceptance check passes. Every long-running step must be resumable, cached, and time-capped (Section 8).

## 1. Project Overview & Research Objective

* **Title:** Resource-Efficient Subgraph Routing via Light GNNs for Multi-Hop GraphRAG
* **Core objective:** Train and benchmark a lightweight, query-conditioned, edge-aware GNN that scores the edges of a small per-question knowledge graph, so that multi-hop supporting evidence is selected **without LLM calls at query time**.
* **Scope of the claim (state this in any write-up):** Per-question graphs built from each question's own candidate paragraphs (distractor-pool setting). This is easier than a global corpus-level GraphRAG. Results do **not** claim to transfer to a single global graph. A small global-graph check is an optional stretch (M9).
* **Target environment:** MacBook Pro M3, 8 GB unified memory, fully local, no cloud APIs.
* **Graph backend:** NetworkX (in-memory). No Docker, no external database.

### 1.1 Testable claims (pre-registered; report results whether or not they hold)

| ID | Claim | Test |
| :-- | :-- | :-- |
| C1 | The GNN router retrieves more complete multi-hop evidence than plain vector retrieval. | **Two primary rows (D15), both on AllGold@4-conditional** (D4: conditioned on all gold paragraphs being in the pool), with unconditional alongside and the datasets reported separately: **(a) graph-only**, always printed next to its graph ceiling, so a result is never read as a routing failure when it is a reachability cap; **(b) matched hybrid pair**, GNN+Vector vs PPR+Vector, fused by reciprocal-rank fusion at fixed `k=60`, paired-bootstrap 95% CI of the difference excluding 0. The pair differs only in the graph ranker, so the comparison isolates routing. |
| C2 | The graph structure and learning both matter. | GNN beats the non-learned graph baseline (PPR) and the 0-layer ablation (no message passing) on AllGold@4, by paired-bootstrap CI. |
| C3 | Query-time routing is far cheaper than LLM path tracing at similar quality. | Routing-stage median latency ≥10× lower than LLM-trace, with AllGold@4 not lower than LLM-trace by more than 3 points. LLM tokens at query time are 0 for the GNN by construction. |
| C4 | The full pipeline fits the hardware. | Peak process-tree RSS stays under 6 GB for every method (Section 7.3). |

Extraction (indexing) cost is reported separately from query-time cost, and an "indexing + query" total is included in the Pareto plot. Graph methods pay indexing cost that Vector RAG does not; do not hide it.

---

## 2. Technical Stack & Hardware Lock

| Component | Selected technology | Lock / config |
| :-- | :-- | :-- |
| Compute | PyTorch | `PYTORCH_ENABLE_MPS_FALLBACK=1` set **before** importing torch. Device chosen by a one-time benchmark (Section 6.5): `mps` or `cpu`. |
| Graph engine | `networkx` | `MultiDiGraph` per question. |
| GNN | `torch-geometric` | `GATv2Conv` with `edge_dim` (edge-aware GAT, not relational). Hidden 128, 4 heads, L=2 default. |
| Embedder | `BAAI/bge-small-en-v1.5` | Frozen, 384-dim, normalized. Loaded only in embedding phases, then freed. |
| SLM (extractor, reader, LLM-trace) | Ollama `qwen2.5:1.5b-instruct-q4_K_M` | Verify the exact tag with `ollama list`. Greedy decoding (`temperature=0`). |
| Strength-check LLM (subset only) | Ollama `qwen2.5:3b-instruct-q4_K_M` | Used only in M7b on a 60-question subset. Never loaded alongside training. |

**Ollama environment (set in `.env` / launch script):** `OLLAMA_NUM_PARALLEL=1` (pilot may test 2), `OLLAMA_MAX_LOADED_MODELS=1`, `OLLAMA_KEEP_ALIVE=30m` during LLM phases, and `num_ctx=2048` per request. Unload the model (`ollama stop <tag>`) before any GNN training phase.

**Pins:** all packages version-pinned in `requirements.txt`. Global seeds (`python`, `numpy`, `torch`) read from `.env` (default `SEED=1234`).

**Run long phases under `caffeinate -i`** with the laptop on power.

---

## 3. Data & Splits

| Dataset | Train items from | Eval items from | Notes |
| :-- | :-- | :-- | :-- |
| MuSiQue (answerable) | official `train` | official `validation` | Stratify by hop count (2/3/4). Report per hop count. |
| HotpotQA (distractor) | official `train`, **level=hard only** (D5) | official `validation` | Report bridge vs comparison separately. Official validation is 100% `hard` while official train is easy/medium/hard; train is restricted to `hard` so the splits match in difficulty. |

* Train and eval always come from **different official splits**. Assert that question ids are disjoint.
* Datasets are reported **separately**. Never pool into one headline number.
* Sample sizes are set by the **budget governor** (Section 8.2). Default targets: 150 train + 150 eval per dataset (600 questions total), range 60–300 per (dataset, split).
* Carve 10% of the training questions as a **dev split** for early stopping and hyperparameter choice. **Never tune on eval.**

### 3.1 Candidate pool (controls extraction cost)

* Pool size is **per dataset** (D1): HotpotQA keeps all **10** of its candidates, MuSiQue keeps the **top-14 of its 20 by cosine similarity to the question** (bge). The pool is built **without looking at gold labels**, identically for train and eval, and shared by every method.
* Questions carrying fewer paragraphs than their dataset's pool size are excluded, so every pool is exactly full (D6). This uses only the paragraph count, never a gold label.
* Report **pool recall ceiling** (fraction of gold paragraphs inside the pool) per dataset. Measured at the chosen sizes: MuSiQue 0.873 train / 0.898 validation, HotpotQA 1.000 / 1.000.
* All methods can only rank within this pool, which keeps comparisons fair.

---

## 4. Task Formulation

### 4.1 Graph construction (per question)

1. **Extraction (offline, cached):** For each pool paragraph, the SLM outputs JSON triples `(head, relation, tail)`. Provenance `pid` (source paragraph) is stored on every triple.
2. **Nodes:** normalized entity strings (lowercase, strip punctuation/articles). Merge exact matches. Optionally merge near-duplicates with embedding cosine ≥ 0.92 (config flag, default on).
3. **Edges:** one directed edge per triple, plus a reverse edge flagged by a direction bit in `edge_attr`. Each edge keeps `pid`.
4. **Seed linking (no LLM):** nodes whose normalized name is a substring of the normalized question; if fewer than 1 found, fall back to the top-3 nodes by cosine to the question embedding. Cap at 5 seeds.
5. **Ego-graph:** undirected BFS from seeds to `H=3` hops; cap at **300 nodes / 800 edges** (keep BFS order; raised from 200/400 by D14 once §4.1.1 connectivity was enabled). Record the **graph ceiling**: fraction of questions whose gold answer string matches a node reachable from a seed within H hops.

### 4.1.1 Cross-paragraph connectivity (added after M5; see D12)

Triples extracted paragraph-by-paragraph do not share entity surface forms, so the
per-question graph arrives as roughly one component per paragraph (measured: 15.0
components for MuSiQue's 14 paragraphs, 7.5 for HotpotQA's 10). BFS from
question-derived seeds then reaches only part of the graph, and **a graph method
cannot rank a paragraph it has no edge for**, which caps AllGold@4 for every graph
method independently of routing quality.

**Target (fixed before any mechanism was implemented, and measured on TRAIN):**

* **≥ 90% of pool paragraphs have at least one edge inside the ego-graph**, and
* the **cap-binding rate stays low** (≤ 10% of questions), so the fix adds
  reachability rather than just inflating graphs into the caps.

**Outcome: the first criterion was MISSED on MuSiQue (88.3%) and met on HotpotQA
(99.5%); the second is met on both (0.0% binding at the D14 caps of 300/800).**
The frozen configuration is accepted with the shortfall recorded in D13. It is not
a pass and must not be reported as one.

Mechanisms, implemented and measured **in this order**, stopping at the first
configuration that meets the target:

* **(a) Paragraph-title hub nodes.** One node per pool paragraph, named by its
  normalized title, linked to every entity extracted from that paragraph. Because
  node identity is the normalized string, a title node merges with an entity of
  the same name occurring in another paragraph — which is what bridges components.
  Title-to-entity linking also accepts fuzzy matches (token containment or token
  Jaccard above a train-chosen threshold).
* **(b) Looser entity merging with explicit `same_as` edges.** Rather than merging
  near-duplicates away, near-identical entity nodes are joined by a `same_as`
  edge, with the cosine threshold and the token-overlap/containment rule chosen on
  train/dev. `same_as` edges carry a sentinel pid, are always `y=0`, and are
  excluded from paragraph scoring: they add reachability, never evidence.
* **(c) Entity-to-paragraph links.** One pid-keyed node per paragraph, linked to
  every entity extracted from it. These never merge across paragraphs, so they
  guarantee each paragraph is representable even when nothing else links it.

All of this uses only `pid`, `title` and `text`. **No connectivity decision may
read `is_gold`**; a test asserts that flipping every gold flag leaves the graph
structure (nodes, edges, pids, seeds, hop distances) byte-identical.

**Every evaluated variant is reported (D19/D20).** Any variant that is run on the
validation split appears in the results tables, including the ones that lose.
Exploratory variants may be developed on train CV only, and the decision to
develop them was taken after seeing CV results but **before any validation
scoring**; a variant cannot be tried on validation and then dropped for being
unflattering.

**Hybrid fallback (reported as a separate variant, never as the main result).**
Paragraphs with no ego-graph edge may be appended to a method's ranking by bge
cosine to the question. This is applied to both PPR and the GNN router and is
always labelled `+hybrid`. The **C1 test stays graph-only**, since a hybrid row
partly measures Vector RAG.

### 4.2 Learning problem

* **Input:** ego-graph + question embedding `q`.
* **Output:** a relevance logit for every edge.
* **Supervision (weak, provenance-based):** edge label `y=1` iff the edge's source paragraph is a gold supporting paragraph for the question; else 0. Use `BCEWithLogitsLoss(pos_weight = #neg/#pos)` computed on the training set.
* **Unmappable examples:** if the extractor produced **no** edges from any gold paragraph, drop the question from **training only**. Keep it in **evaluation** and score it as-is (it counts against all graph methods). Log how many were dropped and report extraction coverage per dataset.
* **Inference (what "multi-hop path selection" means here):** the GNN's multi-layer message passing gives each edge a multi-hop receptive field. Edge scores are then converted to evidence by deterministic aggregation (Section 4.4). Scoring is differentiable; path assembly is not. Do not describe it as end-to-end differentiable path selection.

### 4.3 Features and model

* **Node features** `x_v` = `[e_v ; e_v ⊙ q ; cos(e_v, q) ; is_seed ; onehot(hop_dist 0..H) ; log(1+degree) ; kind(v)]` — **778 dims** at H=3 (384 + 384 + 1 + 1 + 4 + 1 + 3). `e_v` is the bge embedding of the node's text: the entity or title name, and for a pid-keyed paragraph node its paragraph's title, so **every node carries a real embedding and none enters the model as a zero block**. `kind(v)` is a 3-dim **multi-hot** indicator over (entity, title, para) — multi-hot rather than one-hot because a title node that merges with an identically named entity is genuinely both, and that merge is exactly what §4.1.1 relies on.
* **Edge features** `edge_attr` = `[r_e ; cos(r_e, q) ; direction_bit ; kind(e)]` — **390 dims** (384 + 1 + 1 + 4). `r_e` is the bge embedding of the relation phrase. `kind(e)` is a 4-dim one-hot over (triple, title, same_as, para), so the model can tell extracted evidence from §4.1.1 scaffolding instead of having to infer it.
* **Layers:** input linear → `L` × `GATv2Conv(hidden, hidden/heads, heads=4, edge_dim=edge_dim)` + residual + LayerNorm + ReLU/dropout 0.1.
* **Edge head:** MLP over `[h_u ; h_v ; proj(edge_attr) ; proj(q)]` → logit.
* **Size:** < 1M parameters. Hyperparameters: AdamW, lr 1e-3, weight decay 1e-2, batch 32 graphs, max 30 epochs, early stopping (patience 5) on dev AllGold@4.
* **No encoder or Ollama call inside training.** All embeddings (questions, entity names, relation phrases) are precomputed and cached to disk in M3.

### 4.4 From edge scores to context for the reader

1. Paragraph score = mean of its top-2 edge scores (config flag: `max`), computed over **`kind == triple` edges only**. The router scores every edge in the ego-graph, scaffolding included — message passing needs those edges, since they are what carries information between paragraphs — but §4.1.1 scaffolding exists to make paragraphs *reachable*, not to vouch for them. So a `title`, `same_as` or `para` edge is scored and used for message passing, yet contributes to no paragraph score, and `y=1` is restricted to `kind == triple` so no scaffolding edge can ever be a positive even when its own paragraph is gold. A paragraph reached only through its hub gets no score and is not rankable by the graph; it falls to the unreached tail.
2. Rank pool paragraphs by score; take the top `K=4`.
3. The reader receives those 4 paragraphs (plus the top-5 triples as a compact header, flagged by `INCLUDE_TRIPLE_HEADER`). **Every method gets the same K=4 paragraph budget** and the same prompt template, so differences come from routing only.

---

## 5. Methods Compared

| Method | Description | Uses LLM at query time |
| :-- | :-- | :-- |
| **Vector RAG** (primary baseline) | Rank pool paragraphs by bge cosine to the question; top-4. | No |
| **PPR (HippoRAG-style)** | Personalized PageRank on the same ego-graph from seed nodes; paragraph score from the PPR mass of its edges' endpoints. | No |
| **Edge-Sim (no graph)** | Rank edges by cosine of `[head; rel; tail]` text embedding to the question. Isolates "graph vs no graph". | No |
| **LLM-Trace** | Bounded iterative traversal: up to 3 steps; each step prompts the SLM with the question, current frontier, and ≤20 candidate outgoing edges (chosen by degree), and asks for ≤3 edge indices (JSON, `num_predict=40`). Collected edges → paragraphs. | Yes (≤3 calls) |
| **GNN Router (ours)** | Section 4. | No |
| **GNN+Vector (RRF)** | D15 primary hybrid: the GNN ranking fused with Vector RAG by reciprocal-rank fusion, `k=60` fixed and never tuned. | No |
| **PPR+Vector (RRF)** | D15 primary hybrid, the matched control: identical fusion, PPR in place of the GNN. | No |
| **GNN+hybrid-tail / PPR+hybrid-tail** | Secondary labelled variant: graph-scored paragraphs first, unreached tail ordered by cosine. Kept for comparison with the RRF pair; never a primary row. | No |
| **Gold-paragraph oracle** | Reader gets the gold paragraphs only. Gives the reader's ceiling. | n/a |

**Ablations (retrieval metrics only, no reader, so each costs seconds):** GNN layers `L ∈ {0, 1, 2, 3}` (L=0 is the MLP with no message passing); aggregator GATv2 vs SAGE-mean; hop radius `H ∈ {2, 3}`; with/without query conditioning in node features. Each ablation is run with the same 3 seeds as the main model.

**Feature parity for `L=0`.** The `L=0` ablation must consume **exactly** the same
`x` and `edge_attr` tensors as the full model — same 778-dim node features
including the paragraph-node embeddings, same 390-dim edge features including the
edge-kind one-hot — so that the only difference is the absence of message passing.
This is guaranteed structurally: every configuration reads the *same* serialized
`Data` objects from `data/cache/graphs/`, built once by `to_pyg`, and `L` only
changes how many `GATv2Conv` layers are applied on top. No ablation may rebuild
features.

**`H=2` ablation must mask, not rebuild.** The hop-radius ablation `H ∈ {2, 3}` is
run by **masking the cached `H=3` graphs on `hop_dist`** — dropping nodes with
`hop_dist > 2` and any edge touching them — never by rebuilding features at
`H=2`. Rebuilding would change the node feature width (`onehot(hop_dist 0..H)`)
and re-run near-duplicate merging and seeding, so the ablation would differ from
the full model in its inputs as well as its radius and would no longer isolate the
radius. Masking keeps the 778-dim layout intact: the unused hop slot is simply
always zero.

**Edge-type ablations (added with §4.1.1).** Because most of the graph is now
scaffolding (measured on MuSiQue train: 29% `triple`, 33% `title`, 5% `same_as`,
33% `para`), the contribution of each kind is reported separately:

| Ablation | What it removes | Question it answers |
| :-- | :-- | :-- |
| `no_para` | drop `para` edges and paragraph nodes | do pid-keyed hubs help, or only inflate the graph? |
| `no_same_as` | drop `same_as` edges | is cross-paragraph entity matching doing real work? |
| `no_title` | drop title hub nodes and their edges | how much of the connectivity gain is titles alone? |
| `triples_only` | drop all §4.1.1 scaffolding | the pre-connectivity graph, as a floor |

Each is run with the same 3 seeds. These bear directly on C2, since the
TRAIN-only link diagnostic shows ~62% of added cross-paragraph links join
distractor to distractor, so part of the added structure is noise by
construction and the ablations are what separate signal from scaffolding.

**Strength check (M7b, optional):** run LLM-Trace with the 3B model on a 60-question subset to show how much the 1.5B tracer is handicapping that baseline. If 3B is much better, say so; do not present the 1.5B result as the best an LLM can do.

---

## 6. Compute Strategy (why this does not take days)

### 6.1 Phase isolation (never run two heavy things at once)

| Phase | Loaded in memory | Ollama |
| :-- | :-- | :-- |
| Pool building / embeddings | bge + torch | **off** |
| Extraction | Python + Ollama model | **on** (nothing else heavy) |
| Graph build, GNN training, GNN/PPR/Edge-Sim/Vector eval | torch (+ cached embeddings) | **off** (unload) |
| LLM-Trace + reader | Python + Ollama model | **on** |

### 6.2 Extraction is the bottleneck; everything else is a few minutes

* GNN training runs on cached tensors for graphs of roughly ≤ 400 edges, so each run should take minutes.
* Extraction speed levers: `format="json"`, `temperature=0`, `num_predict=192`, paragraph truncated to 200 words, a short fixed prompt (one example at most) so Ollama can reuse the prefix cache, dedupe by paragraph hash across questions, strictly sequential unless the pilot shows `NUM_PARALLEL=2` is faster.
* Malformed JSON: one retry, then store an empty triple list and count it. Gate: failure rate ≤ 5% in the pilot (else fix the prompt first).

### 6.3 Time budget (planning estimates; the pilot replaces them with measured numbers)

Assumptions: ~2.5 s per paragraph extraction, ~7.5 s per LLM-Trace question, ~3 s per reader call. Treat these as guesses until M2 measures them.

| Phase | Work | Est. time |
| :-- | :-- | :-- |
| M0–M2 | Setup, smoke test, pilot | ~30 min |
| M3 | Pool building + embedding cache | < 10 min |
| M4 | **Extraction**: 600 questions × 10 paragraphs ≈ 6,000 paragraphs | **~4 h** (hard cap 5 h) |
| M5 | Graph build, labels, ceilings | < 10 min |
| M6 | GNN training: 3 seeds + ablation configs (~12 runs × ≤4 min) | ~50 min |
| M7 | Non-LLM retrieval eval (all methods + ablations) | < 10 min |
| M7 | LLM-Trace on 300 eval questions | ~40 min |
| M7b | 3B strength check (60 questions, optional) | ~15 min |
| M8 | Reader on 5 conditions × 300 questions ≈ 1,500 calls | ~75 min |
| M10 | Analysis/plots | minutes |
| **Total** | | **≈ 7–8 h machine time, in 3 sittings** (extraction overnight; training+retrieval eval one evening; reader+LLM-Trace another) |

### 6.4 Device choice

Tiny graphs may run faster on CPU than on `mps`. In M0, benchmark one training epoch on both and set `DEVICE` in config accordingly. If `mps` raises an unsupported-op error even with fallback enabled, use CPU and record it.

### 6.5 Hard limits

* Every phase accepts `--max-hours` and exits cleanly with its cache flushed.
* Every phase accepts `--resume`. Re-running a finished phase must be a no-op that takes seconds.
* No phase may exceed 5 hours in one run. If the governor (Section 8.2) predicts more, it shrinks the sample sizes instead.

---

## 7. Metrics & Measurement

### 7.1 Retrieval metrics (primary: independent of reader quality)

* **SP-Recall@4:** fraction of gold supporting paragraphs among the top-4 (macro-averaged over questions).
* **AllGold@4:** fraction of questions for which **all** gold paragraphs are in the top-4. This is the headline multi-hop metric, and it is reported **two ways** (D4):
  * **unconditional** — over all evaluated questions, so pool-construction loss is included;
  * **conditional** — over only those questions whose gold paragraphs all survived into the pool, which isolates routing quality from pool construction.
  Claim **C1 is tested on the conditional version**; the unconditional number is always reported next to it. The gap between them is exactly the pool-recall ceiling, so neither can be quoted without the other.
* **Edge-Precision@10:** fraction of the top-10 scored edges that come from gold paragraphs (not applicable to Vector RAG).
* **Path-Recall:** fraction of questions where the selected top-B edges (B=10) contain a connected path from a seed to a node matching the gold answer (normalized exact match). Report alongside the **graph ceiling** from 4.1.
* **Pool-recall ceiling** and **extraction coverage** reported per dataset as context for all numbers above.

### 7.2 Answer metrics (secondary)

* **EM / F1** with standard SQuAD answer normalization. Expect low absolute values from a 1.5B reader; compare **across methods** and against the gold-oracle row.

### 7.3 Efficiency metrics (per method, per stage)

Stages: `indexing` (extraction), `routing` (everything from question + graph to ranked paragraphs), `reading` (generation).

* **Latency:** median and p95 per question per stage (`time.perf_counter`). For graph methods the routing stage is reported **split** into graph assembly (question embedding, seed linking, ego-graph and feature assembly) and model forward + aggregation, because assembly dominates: 38.9 of 43.8 ms on MuSiQue. **Any figure that excludes assembly is not the routing stage** and must be labelled "model forward only" — `scripts/m7c_stage.py` builds its graphs outside the timer and so reports forward-only (4.1–4.2 ms), which is ~10× faster than the full stage and must not be quoted for C3.
* **LLM tokens:** `prompt_eval_count + eval_count` from each Ollama response, summed per stage.
* **Peak memory:** sample **RSS of the whole process tree plus the Ollama server/runner processes** every 100 ms with `psutil`, during each stage. Subtract idle baseline RSS measured before the stage and report both the absolute peak and the delta. Python-process-only RSS **must not** be used; it undercounts every method that calls Ollama.
* Also log `psutil.virtual_memory()` and swap usage during runs; any swap growth > 500 MB invalidates a run and it must be redone with a smaller batch.

### 7.4 Statistics

* **Paired bootstrap** (10,000 resamples over questions) for **differences between methods** on the same questions; report 95% CIs on both per-method values and differences.
* 3 training seeds for the GNN; report mean ± std over seeds and use the seed-averaged edge scores for the bootstrap.
* With ~150 questions per dataset the CIs will be wide; say so rather than over-interpreting small gaps.

---

## 8. Resumability & Budget Governor

### 8.1 Caching

* `data/cache/triples.jsonl`: append-only, one line per paragraph, key = `sha1(model_tag + prompt_version + paragraph_text)`. Flush after every paragraph.
* `data/cache/embeddings/`: `.pt` files for questions, entity names, relation phrases (keyed by text hash).
* `data/cache/graphs/`: serialized PyG `Data` per question.
* `results/`: one JSONL per (method, dataset, seed) with per-question outputs, so metrics can be recomputed without rerunning models.

### 8.2 Budget governor (runs at the end of the pilot)

1. Pilot extraction on 50 random pool paragraphs; measure `t_p` (seconds per paragraph) and JSON failure rate.
2. Compute per-(dataset, split) sample size (D1 generalizes the original `4 · POOL_SIZE` term, where 4 = 2 datasets × 2 splits, to a sum over datasets, since pool size now differs by dataset):
   `N = clamp(floor(EXTRACT_BUDGET_S / (2 · Σ_d POOL_SIZE_d · t_p)), 60, 300)` with `EXTRACT_BUDGET_S = 14400` (4 h).
   This reduces to the original formula when both pool sizes are equal, and reproduces its worked examples exactly: `t_p = 2.5 s` → N = 144; `t_p = 1.5 s` → N = 240; `t_p = 4 s` → N = 90.
   Report both the nominal paragraph count and the dedupe-adjusted one (measured dedupe: MuSiQue 15.2%, HotpotQA 1.0%).
3. **Gate on estimated total extraction hours, not on `t_p`** (D3). If the projected run exceeds `max_estimated_hours` (5 h) or JSON failure > 5% after one prompt fix: **stop and report**. Do not silently run longer. `t_p` is still reported, but a `t_p` threshold is not a gate: it is only meaningful against a fixed pool size and question count, whereas the budget constrains total hours.
4. Write the chosen N values and measured throughput to `results/budget.json`.

---

## 9. Repository Layout

```text
graphrag-light-gnn/
├── configs/default.yaml          # POOL_SIZE, H, L, K, seeds, budgets, DEVICE
├── data/
│   ├── raw/
│   └── cache/                    # triples.jsonl, embeddings/, graphs/
├── src/
│   ├── data.py                   # loading, splits, pool building
│   ├── embedder.py               # bge wrapper + disk cache
│   ├── extractor.py              # SLM OpenIE with cache + retries
│   ├── graph_builder.py          # NetworkX graph, seeds, ego-graph, PyG conversion
│   ├── gnn_router.py             # EdgeRouter model + training loop
│   ├── generator.py              # reader via Ollama
│   └── profiling.py              # stage timer, token counter, process-tree RSS sampler
├── evaluation/
│   ├── baselines.py              # vector_rag, ppr, edge_sim, llm_trace
│   ├── evaluate.py               # runs methods, writes results/*.jsonl
│   ├── metrics.py                # retrieval, EM/F1, paired bootstrap
│   └── ablations.py
├── scripts/
│   ├── pilot.py                  # throughput pilot + budget governor
│   └── run_phase.sh              # caffeinate wrapper, env, ollama load/unload
├── notebooks/pareto_analysis.ipynb   # latency / tokens / RAM vs AllGold@4
├── tests/
├── results/
├── .env
├── requirements.txt
├── main.py                       # CLI: `main.py <phase> [--resume] [--max-hours H]`
└── agent.md
```

---

## 10. Module Interfaces (keep these signatures stable)

```python
@dataclass
class Paragraph: pid: str; title: str; text: str; is_gold: bool
@dataclass
class Example: qid: str; dataset: str; question: str; answer: str
               paragraphs: list[Paragraph]; n_hops: int | None
@dataclass
class Triple: head: str; rel: str; tail: str; pid: str
@dataclass
class Routed: paragraph_ids: list[str]          # ranked, length >= K
              edges: list[tuple[int, float]]    # (edge_idx, score), ranked; empty for Vector RAG
              stats: StageStats                 # latency_s, llm_tokens, peak_rss_mb

# data.py
def load_examples(dataset: str, split: str, n: int, seed: int) -> list[Example]
def build_pool(ex: Example, embedder, pool_size: int) -> Example   # no gold access

# extractor.py
def extract_paragraph(p: Paragraph, cache: TripleCache) -> list[Triple]
def extract_all(examples, cache_path, max_hours: float, resume: bool) -> ExtractionStats

# graph_builder.py
def build_graph(ex, triples, embedder, hops=3, max_nodes=200, max_edges=400) -> QueryGraph
def to_pyg(qg, emb_cache) -> "torch_geometric.data.Data"   # x, edge_index, edge_attr, y_edge, edge_pid, seed_mask, q

# gnn_router.py
class EdgeRouter(nn.Module): def forward(self, data) -> Tensor   # [E] logits
def train_router(train, dev, cfg, seed) -> Path                 # checkpoint
def route(model, data, k: int = 4) -> Routed

# baselines.py  (all return Routed)
def vector_rag(ex, embedder, k=4); def ppr_rag(qg, k=4)
def edge_sim(qg, embedder, k=4);   def llm_trace(qg, ex, model_tag, k=4, steps=3)

# generator.py
def answer(question: str, paragraphs: list[Paragraph], triples=None) -> tuple[str, Usage]

# metrics.py
def sp_recall(routed, ex, k=4); def all_gold(routed, ex, k=4)
def edge_precision(routed, qg, b=10); def path_recall(routed, qg, b=10)
def em_f1(pred, gold); def paired_bootstrap(a: list[float], b: list[float], n=10_000) -> CI
```

---

## 11. Build Order, Milestones & Acceptance Checks

| # | Milestone | Acceptance check | Time |
| :-- | :-- | :-- | :-- |
| M0 | Env, pinned deps, `.env`, device benchmark (mps vs cpu) | Imports work; `DEVICE` set; Ollama model answers a test prompt | 20 min |
| M1 | `data.py`, `embedder.py`, `build_pool` | Train/eval id sets disjoint (assert); pool recall ceiling printed per dataset; `build_pool` never reads `is_gold` (unit test) | 20 min |
| M2 | **Pilot + budget governor** (`scripts/pilot.py`) | `results/budget.json` written; JSON failure ≤ 5%; `t_p ≤ 6 s`; optional NUM_PARALLEL=2 test recorded | 15 min |
| M3 | Embeddings cached for all pools | Re-run takes seconds; bge freed from memory | < 10 min |
| M4 | `extractor.py` full run | Resumable (kill mid-run, rerun → continues); cache lines match paragraph count; extraction coverage of gold paragraphs reported; **≤ 5 h** | ~4 h |
| M5 | `graph_builder.py`, labels, ceilings | Graph ceiling, seed-hit rate, # dropped train questions logged; PyG `Data` shapes validated on 10 graphs | < 10 min |
| M6 | `gnn_router.py` training (3 seeds) + ablation configs | Overfit test on 8 graphs reaches near-zero loss; dev AllGold@4 logged per epoch; each run ≤ 5 min; Ollama unloaded | ~50 min |
| M7 | Retrieval eval for all non-LLM methods; LLM-Trace on eval set | `results/*.jsonl` written; metrics recomputable from files alone; Vector RAG matches a hand-checked sample | ~50 min |
| M7b | (optional) 3B LLM-Trace on 60 questions | Reported as a separate row | ~15 min |
| M8 | Reader on Vector RAG, PPR, LLM-Trace, GNN, Oracle | EM/F1 + stage costs per method; oracle row present | ~75 min |
| M9 | (optional stretch) one shared global graph for ~200 MuSiQue questions | Only if M0–M8 finish early; reported as a separate, harder setting | open |
| M10 | Pareto notebook, paired-bootstrap tables, claims C1–C4 verdicts | Every claim marked supported / not supported / inconclusive with its CI | 1 h |

---

## 12. Rules for the Coding Agent

**Do**
* Pilot before committing to sizes; let the governor choose N.
* Read from caches; make every phase resumable and idempotent.
* Log pool recall, extraction coverage, graph ceiling, and dropped-example counts, so the reader can tell where accuracy is lost before the GNN even runs.
* Keep one config file as the single source of truth for hyperparameters and seeds.
* Write unit tests for: pool building (no gold leakage), seed linking, label construction, metric functions, cache keys.

**Don't**
* No cloud APIs or hosted models.
* Never run Ollama generation and GNN training at the same time.
* Never load a full dataset into memory; stream and slice.
* Never tune on the eval split or change K, POOL_SIZE, or prompts after seeing eval results (log any change as a deviation).
* Never measure memory from the Python process alone.
* Never pool MuSiQue and HotpotQA into one headline metric.
* Never extend a phase beyond its hour cap; shrink scope instead and report it.

---

## 13. Known Limitations (include in any write-up)

* Per-question graph setting (Section 1); not a global GraphRAG result.
* Weak provenance labels and a 1.5B extractor cap achievable recall; ceilings are reported.
* Small evaluation sets (~150 per dataset); wide CIs expected.
* 1.5B reader compresses EM/F1 differences; retrieval metrics are primary.
* The 1.5B LLM-Trace baseline may be a weak tracer; the 3B check (M7b) bounds this.
* Time estimates are planning assumptions until M2 measures throughput on the actual machine.

---

## 14. Deviation Log

Every departure from the specification as originally written, with the reason and
the measurement that motivated it. Section 12 requires changes to be logged rather
than made silently. Nothing here was changed after seeing eval results: all of it
predates M5.

| ID | Section | Change | Reason |
| :-- | :-- | :-- | :-- |
| **D1** | 3.1, 8.2 | `POOL_SIZE=10` for both datasets → **per-dataset**: HotpotQA **10**, MuSiQue **14**. | MuSiQue pool recall at 10 was **0.778 train / 0.791 validation**, below the 85% gate. §3.1's prescribed remedy (raise to 12) reaches only 0.836 train — still failing the same gate that triggered it, and §3.1 offers no fallback. 14 is the smallest size clearing it on both splits (**0.873 / 0.898**). HotpotQA stays at 10 because 10 is all it ships, so raising it would do nothing. Measured cost at t_p=1.9 s: none — N is budget-bound, not pool-bound (see the sensitivity table in `results/budget.json`). |
| **D2** | 6.2 | Extraction prompt **v1 → v2**: `format` is a JSON *schema* with `maxItems=6` (grammar-enforced) rather than the string `"json"`; paragraphs truncated to **120** words rather than 200; responses cut off at `num_predict` are **salvaged** for their complete leading triples. | v1 measured a **50% JSON failure rate**, every failure being `done_reason=length` — the output cap truncating valid JSON mid-array, not malformed generation. Asking for a triple limit in prose was ignored by the 1.5B model (measured: no change). The schema enforces it during decoding. 120 words affects only 2/15 pool paragraphs and keeps 98.8% of all words (median paragraph is 99 words). Post-fix failure rate: **0%** after salvage, **4%** strict. |
| **D3** | 8.2 step 3 | Gate on **estimated total extraction hours** (≤ 5 h) instead of `t_p ≤ 6 s`. `t_p` is still reported. | `t_p` is only interpretable against a fixed pool size and question count; what the budget actually constrains is total hours. The original gate also proved to measure the machine rather than the pipeline: identical code and inputs gave **t_p 7.24 s on battery with 3.3 GB of swap in use, and 1.91 s on AC power with swap at 0** — a 3.7× swing that would have failed or passed the gate for reasons unrelated to the extractor. |
| **D4** | 1.1, 7.1 | **AllGold@4** reported both **unconditional** and **conditional** on all gold paragraphs surviving into the pool. Claim **C1 is tested on the conditional** version. | Unconditional AllGold@4 is capped by pool recall, so it charges pool-construction loss to the router and cannot isolate routing quality. At MuSiQue pool 14 the all-gold-in-pool rate is ~0.68/0.74, so the unconditional metric is bounded well below 1 regardless of how good the router is. Both numbers are always reported together; their ratio is the pool ceiling. |
| **D5** | 3 | HotpotQA **train restricted to `level=hard`**. | Official validation is **100% hard**; official train is easy/medium/hard (measured: 17,709 easy / 56,349 medium / 15,551 hard at 10 paragraphs). Training on a mostly-medium distribution and evaluating on all-hard is an avoidable train/eval mismatch. Validation is untouched. |
| **D6** | 3.1 | Questions with **fewer paragraphs than their dataset's pool size** are excluded. | Both official splits carry a short-context tail: HotpotQA validation has 60 of 7,405 rows (0.8%) with 2–9 paragraphs, MuSiQue train 21 of 19,938 (0.1%). §3.1 requires equal-sized pools so every method ranks within the same budget. The filter reads only the paragraph count, never a gold label, and is applied identically to train and eval. |
| **D7** | 10 | `Example` gains a trailing `meta: dict`; `load_examples` gains a trailing optional `cfg=None`. | Additive only — no listed field or parameter changed meaning or order. `meta` carries HotpotQA's bridge/comparison type and level, which §3 requires reporting separately and which must survive `dataclasses.replace()` in `build_pool`. `cfg` lets the loader resolve paths and per-dataset pool sizes without a global. |
| **D8** | 8.1 | `prompt_version` encodes the triple cap, e.g. `v2-mt6`. | The §8.1 cache key is `sha1(model_tag + prompt_version + paragraph_text)` and does not include `max_triples`, which changes the output. Without this, a sweep over `max_triples` silently serves the first config's triples to every later one. The key formula itself is unchanged. |
| **D15** | 1.1, 5 | **Primary C1 reporting changed** from a single graph-only row to **(a)** graph-only, always shown next to its graph ceiling, and **(b)** a matched hybrid pair, **GNN+Vector vs PPR+Vector**, fused by reciprocal-rank fusion with **`k=60` fixed and untuned**. The pre-existing graph-first hybrid (graph-scored, then unreached tail by cosine) is kept as a separate labelled variant, never primary. Decided **after** seeing the validation graph ceilings but **before** any method produced a result. | Graph-only C1 on MuSiQue is **structurally capped and cannot be won by routing quality**: the graph ceiling is ~38% unconditional against Vector RAG's ~74% (all-gold-in-pool), and ~51% conditional against Vector RAG's 100% (conditioning on all-gold-in-pool means every gold paragraph is in the pool, and Vector RAG can rank the whole pool). A perfect router loses both comparisons by construction, so the pre-registered test would measure reachability, not routing. The matched RRF pair restores a fair test: both arms get the same vector ranker and the same fusion, so the only difference is the graph ranker. Graph-only is still reported — next to its ceiling, so the cap is visible rather than mistaken for a routing failure. `k=60` is the original RRF paper's value and is frozen; tuning it would make the hybrid partly a fitted result. HotpotQA needs none of this (99.3% ceiling) and is unaffected. |
| **D27** | 3, 7.4, 11 | **M11 validation extension — pre-registered BEFORE any new validation question was read.** **Purpose:** tighten the confidence intervals and attempt a confirmatory replication of C1–C3 on questions no part of this project has seen. The original 150-question results stay exactly as reported and are not revised. **Design:** +350 **new** validation questions per dataset (500 total = the original 150 + 350 new). **Sampling rule, fixed here before looking at any new question:** MuSiQue (answerable) validation stratified by hop count at **117 / 117 / 116** for 2/3/4-hop, seed **20261026**, excluding every qid in the original 150; HotpotQA (distractor) validation, **350** drawn at random from `level=hard`, same seed, same exclusion. Pool construction is unchanged and gold-blind (MuSiQue top-14 of 20 by bge cosine, HotpotQA all 10). Chosen qids are written to `results/ext_sample_<dataset>.json` with the seed and a sha256 of the list, and zero overlap with both the original 150 and train is asserted. **Analysis plan, fixed here before scoring:** the **primary, confirmatory** analysis is the **350 new questions alone**, using the identical tests that produced the original C1/C2/C3 verdicts (AllGold@4 unconditional and D4-conditional, SP-Recall@4, Edge-Precision@10, Path-Recall, reader EM/F1, paired bootstrap at 10,000 resamples, 95% CI); the **pooled 500** is **secondary**. **If the new-350 and original-150 verdicts disagree, both are reported and the disagreement is stated, not smoothed.** **Nothing is retrained, retuned or refitted:** the frozen config, the six final GNN checkpoints, the frozen D19 stacker weights, prompt version `v2-mt6`, K=4, the per-dataset pool sizes, the model tags and every connectivity setting are used exactly as frozen. No file in `results/` is overwritten; all new output carries an `ext_` prefix. **Multiplicity:** every new comparison is added to the running tally and the updated total is reported, still uncorrected. | Registered as a deviation because it adds validation data after the one-look discipline of D12 was already spent, which is a change to the evaluation protocol and must be visible as one. The honest framing: the original 150-question verdicts were produced under a protocol that permitted a single scored look, and this extension deliberately takes a second. Writing the sampling rule, the seed, the stratification and the analysis plan down **and committing them before reading a single new question** is what keeps the new-350 analysis confirmatory rather than exploratory — the intervals it produces are only trustworthy if the design could not have been adjusted after seeing the data. The new-350 set is the primary analysis precisely because pooling would let the original 150 dilute a disagreement; keeping them separable is what makes a replication failure detectable. |
| **D28** | 7.3, 11 | **M11 step 3: a retry-loop bug in `scripts/ext_extract.py` lost the timing and per-paragraph accounting for 12 of 13 extraction chunks; the triples themselves are complete and correct, and nothing was re-extracted.** The chunk loop was written to re-run a chunk once if its swap growth crossed the Section 7.3 threshold, on the theory that a cooled-down machine would yield a usable measurement. But extraction caches per paragraph, so on the second pass every paragraph in the chunk was already cached and skipped: the retry measured nothing and overwrote the chunk's record with `n_extracted: 0`, `t_p: null` and `swap_growth_mb: 0.0` -- the last of which marked a chunk that had in fact grown swap by +1.4 to +4.4 GB as Section 7.3-**valid**. The first report therefore claimed 272 paragraphs extracted (of 6,272), t_p 4.351 s measured on the final chunk alone, and 0 of 13 invalid chunks. **What was actually lost:** only the accounting. The cache flushes after every paragraph and records `seconds`, `json_failed`, `n_triples` and `llm_tokens` per paragraph, so `scripts/ext_extract_recover.py` rebuilds the full per-chunk table from the cache and recovers the real first-pass swap deltas from the run log, where they had been printed. **Corrected figures:** 6,272 of 6,272 paragraphs extracted in 7.21 h, t_p mean 4.020 s (median 4.011, p95 6.894), 0 JSON failures, 423.6 LLM tokens and 5.52 triples per paragraph, gold-paragraph coverage 1.0000 on both datasets, and **12 of 13 chunks timing-invalid** under Section 7.3. The retry was replaced by a single measured pass that keeps the triples, marks the timings invalid, and cools down before the next chunk, where a cooldown can still help. | Logged because the first `results/ext_extraction.json` was wrong in a direction that flattered the run -- it under-reported the work done and, worse, reported 0 invalid chunks when 12 of 13 were invalid -- and because the published t_p for this stage changes (4.020 s against the 1.767 s the step-2 timing check projected, which is why the run took 7.21 h rather than the projected 3.08 h; still inside the 12 h gate, so no stop condition fired). The triples are unaffected: decoding is greedy at temperature 0, each paragraph is extracted independently, and the recovered per-paragraph quality is indistinguishable from the original run (salvage 4.88% vs 5.35%, truncation 4.88% vs 5.39%, JSON failures 0.000% vs 0.015%, 423.6 vs 426.4 tokens, 5.52 vs 5.54 triples). No retrieval or reader metric depends on the lost timings, so no stage was re-run to recover them. |
| **D29** | 4.3, 7.3, 11 | **M11: the extension's first scoring sweep ran the router query-blind on HotpotQA, because `to_pyg` silently zero-filled a missing question embedding. Fixed by making it raise; the affected numbers were re-measured, not kept.** `to_pyg`'s `_emb` helper returned `torch.zeros(384)` for any cache miss. For the question that is not a recoverable miss: `q = 0` makes `e_v * q`, `cos(e_v, q)` and `edge_attr`'s `cos(r_e, q)` all zero, so every query-conditioned feature vanishes and the router scores edges with no knowledge of the question -- while still returning plausible-looking output. **Why it fired here:** `build_graph` does not cache the question (verified against an empty cache: the question embedding is absent afterwards, while all node texts are present), so the only thing that fills it is the M3 embedding pass, which was never re-run for the new pools. MuSiQue escaped by accident -- its top-14-of-20 pool selection embeds the question already -- while HotpotQA's pool is all 10 paragraphs and needs no selection, leaving 350 questions and 3,500 paragraphs uncached. **Effect, measured:** HotpotQA GNN-family AllGold@4 GNN 0.571 -> 0.583, GNN+Vector 0.717 -> 0.717, stacked 0.777 -> 0.786, SP-Recall@4 GNN 0.7486 -> 0.7586; MuSiQue and every non-GNN method byte-identical, which is the signature of a feature-only fault (the graphs themselves were unchanged). The cold-cache run is kept at `results/ext_cold_snapshot/` so the correction is auditable. **Fix:** `to_pyg` now raises `KeyError` on a missing question embedding, on the empty-graph path too, with two regression tests; `scripts/ext_embed.py` performs M3's pass for an extension split. | Logged because a published metric changed and because the bug class is worse than the numbers: it degrades silently, in the direction of looking like a weak-but-working router rather than a broken one, and nothing in the pipeline would have surfaced it. It is also the second instance of the same failure mode as the earlier paragraph-nodes-embedded-as-zeros fault, so the zero-fill default is now treated as the defect rather than the symptom. No frozen setting, hyperparameter, checkpoint or prompt changed: the embedding pass only restores the cache state the original 150-question run was measured under, which is what makes the two comparable. The re-measured sweep was verified deterministic (0 ranking or metric differences across 350 HotpotQA questions on a repeat run) and zero-fill-free (0 of 350 question and 0 of 19,904 node lookups missing, counted inside the real scoring path). |
| **D30** | 4.3, 7.3, 11 | **The same zero-fill fault again, one field over: 10% of the extension's edges were scored with `r_e = 0`. Found by auditing the rest of `to_pyg` after D29 instead of stopping at the question.** D29 made the *question* lookup strict and left the others zero-filling. An audit of every kind on the extension pools then measured **relation** cache misses of **10.2% (1,636/15,994) on MuSiQue and 9.1% (1,334/14,628) on HotpotQA**, against **0.0%** on the published train and validation splits. A missed relation means that edge's whole `edge_attr` carries no signal: `r_e = 0` and `cos(r_e, q) = 0`. Entity lookups missed 0% only because `build_graph` encodes entity names itself and they land in the in-memory cache before `to_pyg` reads it -- nothing encodes relation texts, and `scripts/ext_embed.py` covered only questions and paragraphs. **Also a latency fault:** those same uncached entity names were being encoded by bge-small *inside the timed routing region*, 895 encodes to 1,274 hits per 12 extension graphs against 186 to 2,025 for the published 150. The extension's graphs are *smaller* (median 270 edges vs 380 on MuSiQue) yet assembly cost **2.4x more per edge**, which is what prompted the audit. **Fix:** `_emb` now raises on a miss for every kind, with regression tests per kind; `ext_embed.py` gained a second pass that builds the graphs once and caches the relation and entity texts. The GNN-family metrics and all routing timings on the extension were re-measured. | Logged because it changed published extension metrics and because it is the *third* instance of one failure mode -- paragraph-nodes-as-zeros, then D29's query-blind router, now relation features. The lesson taken is that D29's fix was too narrow: patching the one field that had already burned us left the identical trap in the two beside it, and a cache miss never means "this text has no embedding", it means the embedding pass was not run. Published splits measuring 0% miss is what makes the strict version safe to adopt globally rather than a risk to earlier phases. Separately, and **correcting a first attribution of this to ambient machine state**: measuring in-loop encodes per graph showed the cause is the same artefact on *both* sides. The published validation split was itself doing **8.2 (MuSiQue) and 6.5 (HotpotQA) bge-small encodes per graph inside the timed region**, because the M3/M5 passes never persisted every text `build_graph` produces -- notably those that exist only after near-duplicate merging. With the graph texts of **both** splits cached (`scripts/ext_warm_baseline.py`, embedding-cache writes only), GNN routing medians are **29.0 / 20.9 ms** on the published 150 and **26.5 / 19.9 ms** on the new 350: the two question sets agree, and the extension's figures are not better than the original's. **Section 7's 43.8 / 35.6 ms are inflated by this artefact** -- the same fault class as D23's Edge-Sim 583.5 -> 10.6 ms, handled the same way (fix the measurement, change no metric, write down both numbers). Section 7 is not revised; §10 states the corrected baseline alongside it. The published 150's metrics reproduced **exactly** (SP@4 0.5500 / AllGold@4 0.240 MuSiQue, 0.7500 / 0.567 HotpotQA) across all three timings, which is what makes this a measurement change and not a rerun. C3c's ratio rises accordingly and should still be read as an order of magnitude, since its LLM-Trace numerator is §7.3-invalid on swap (D24). |
| **D26** | 7.3, 11 | **8 of 33 measured stages are invalid under §7.3 and are reported as such**, not quietly dropped: both M7c LLM-Trace stages (+3,389 and +2,195 MB), five M8 reader stages (MuSiQue ppr +1,587, oracle +926, gnn +830, llm_trace +568, gnn_rrf +507; HotpotQA vector +841) and the HotpotQA 3B stage (+782). **EM/F1 and every retrieval metric are unaffected**, because the reader and the traversal decode greedily at `temperature=0`: identical context gives identical output regardless of timing or paging. Only latency, token-rate and RSS figures from those stages are void. | Stated explicitly so no reader has to infer which numbers survive. The invalid set is also informative: it is almost entirely the Ollama-loaded stages, which is the predicted consequence of a 1.1–1.9 GB model on an 8 GB machine, and it is why C4's RSS figure is treated as a floor rather than a faithful total. |
| **D25** | 7.3 | **The M7c swap gate checks only at script start, so stages after the first ran from a degraded baseline.** The run began at **0 MB** swap and passed the 300 MB gate, but the second stage onwards started at **~1.6 GB** because the preceding LLM-Trace stage had already driven the machine into swap. A per-stage gate, or a separate process invocation per stage with its own gate, would have caught this. | Recorded because it changes how the "valid" verdicts should be read: the two routing stages are marked valid on the §7.3 growth rule (+0 MB) and that verdict stands, but they were measured on a machine already holding 1.6 GB of swap, so their absolute RSS and latency are not from a clean machine either. The growth rule is the specified criterion and is applied as written; the start-state caveat is reported alongside rather than used to override it. |
| **D24** | 7.3 | **LLM-Trace efficiency is reported as INDICATIVE, never valid.** All four measurements are tabulated (M7 first run, D21 redo, 10-question smoke, M7c clean-start): medians 1,543–1,639 ms (MuSiQue) and 1,684–2,157 ms (HotpotQA), 1,050–1,405 tokens/question. **Three of the four tripped §7.3**; only the 10-question smoke run (+0 MB swap) is formally valid, and it is the smallest sample. | Four attempts, including one that started from 0 MB swap, all drove swap up by 2–3.4 GB. On this hardware the stage cannot be measured within §7.3: a 1.1 GB model plus the graph-building process exceeds what 8 GB allows. Rather than keep re-running, the figure is reported as indicative with its full spread. This is safe for C3 because the medians agree within 6% (MuSiQue) and 28% (HotpotQA) across runs with wildly different swap, and the latency ratio is 35–61×, a margin no plausible paging artefact closes. |
| **D23** | 1.1, 7.3 | **Two M7 verdicts corrected after computing the CIs the claims actually call for.** (a) **C3's quality half on MuSiQue changes from NOT SUPPORTED to INCONCLUSIVE**: the test is "AllGold@4 not lower than LLM-Trace by more than 3 points", i.e. (GNN − Trace) ≥ −0.03, and the CI is **−0.060 [−0.133, +0.007]** unconditional (−0.045 [−0.126, +0.036] conditional), which **straddles** the −0.03 margin rather than falling below it. HotpotQA stays SUPPORTED (+0.200 [+0.107, +0.293], entirely above the margin). (b) **Edge-Sim's reported latency was an artefact and has been re-measured**: preloading its edge-text embeddings before the timed stage took it from **583.5 → 10.6 ms** (MuSiQue) and **674.3 → 6.2 ms** (HotpotQA); ~98% of the original figure was sharded-cache I/O. Its metrics are byte-identical across both runs, as a deterministic method must be. | (a) The first pass compared the point estimate to the margin instead of testing the CI against it, which is exactly the error §7.4's paired bootstrap exists to prevent — with n=150 the interval is wide enough that "6 points behind" cannot be distinguished from "within 3 points". (b) Reporting 583 ms would have overstated Edge-Sim's query-time cost by ~55x and made the C3 latency comparison look better for the GNN than it is. Re-timing changed no metric, only the timing, so this is a measurement fix rather than a rerun on validation results. |
| **D22** | 5 | **Correction to how the D19 stacked hybrid's train/test mismatch is described.** An earlier write-up said the mismatch "can only flatter the stacked row". That was wrong and is withdrawn. The final GNN models never saw a validation question, so the stacked row's validation score is a genuine out-of-sample result. The actual mismatch: the combiner's `gnn_score` feature was fit on out-of-fold scores from models trained on **~108 of 135** training questions, and at validation time is produced by final models trained on **131 of 135**. The direction of any resulting bias is **not determined** and no claim is made about it. | Recorded as a correction rather than edited away, because the original wording asserted a bias direction with no argument behind it, and a reader could have discounted the stacked result on that basis. Mitigations worth stating: features are min-max normalized per question, so scale shifts between OOF and final scores are largely absorbed, and the training-set gap is 108 vs 131 questions rather than out-of-sample vs in-sample. |
| **D21** | 7.3, 11 | **The first M7b LLM-Trace run was invalidated and redone.** Its efficiency stages recorded swap growth of **+3,169 MB (MuSiQue)** and **+4,020 MB (HotpotQA)** against §7.3's 500 MB rule, so the latency and RSS figures were void. Cause: both datasets were measured in one process, which held 300 QueryGraphs, the bge encoder and the resident embedding-cache shards **while Ollama's 1.1 GB model was also loaded** — on an 8 GB machine that forces paging. The redo runs **one dataset per process** and releases the encoder and cache **before** Ollama loads. | Logged because it is a rerun on validation, which the M7 protocol otherwise forbids. Two things make it a measurement redo rather than a re-selection: **nothing was tuned, re-chosen or re-fit** — the frozen config, the checkpoints and the method list are untouched — and the **retrieval metrics were never in question**, since greedy decoding at `temperature=0` makes the rankings deterministic and independent of timing, and they were already written per question before the invalidation was detected. Only latency, token and RSS figures were recomputed. The invalidation was caught by the §7.3 rule doing exactly its job, which is also why D10 kept that rule strict for M7/M8 while relaxing it for extraction. |
| **D20-SKIPPED** | 4.3, 5 | **D20 was gated on D19 and the gate closed, so the paragraph-aware GNN was never implemented.** D19's stacked hybrid vs Vector RAG on AllGold@4: MuSiQue **−0.013 [−0.053, +0.020]**, HotpotQA **+0.027 [−0.020, +0.073]**. Neither CI excludes zero, and the point estimates are not positive on both, so neither branch of the gate opened. | Recorded as a deliberate non-result rather than silently dropped. The evidence is informative: the learned combiner gives the GNN a clearly non-zero weight (+0.75 MuSiQue, +1.96 HotpotQA) so the graph score does carry signal, but vector cosine dominates it 4.1x and 2.4x, and the combination still does not beat vector similarity alone. Adding a paragraph-level feature and loss to the router was motivated by the hypothesis that paragraph-level signal was missing; D19 shows the paragraph-level signal that exists is already captured and is simply weaker than the vector baseline. Building D20 would have been fitting more capacity to a signal measured as insufficient. |
| **D20** | 4.3, 5 | **Paragraph-aware GNN**, an exploratory variant: a paragraph node gains a cosine-to-question feature computed from the cached paragraph-text embedding (not just its title), plus a paragraph-level term in the loss. Any feature-width change is recorded in §4.3. **Gated**: implemented only if D19 shows a gain over Vector RAG whose CI excludes zero on at least one dataset, or a clearly positive point estimate on both. | The CV ablations show the router scores edges well but that this does not reach paragraph selection, and that `para` links — which exist only to connect paragraphs — contribute nothing measurable. The natural hypothesis is that the model has no paragraph-level signal at all: it is supervised and scored edge-by-edge, and a paragraph node currently carries only its title's embedding. The gate exists so this is attempted only if the cheaper stacking variant first shows the paragraph-level signal is there to be had. |
| **D19-RESULT** | 5 | **D19 was implemented and evaluated on train CV; it does not beat Vector RAG and will still be reported on validation.** Stacked vs Vector RAG (AllGold@4): MuSiQue −0.013, HotpotQA +0.027, both CIs including zero. Stacked vs the RRF pair: **significantly better** on SP-Recall@4 on both datasets (+0.049 [+0.014, +0.086] and +0.053 [+0.020, +0.087]) and on AllGold@4 for HotpotQA (+0.100 [+0.033, +0.167]). | A learned combiner beats fixed-weight RRF, which is worth reporting: D15's `k=60` fusion is deliberately untuned and demonstrably leaves value on the table. But the combiner still cannot beat vector similarity alone, which is the answer to the question D19 was built to ask -- the graph scores are not complementary enough to the vector signal to change paragraph selection. Reported on validation regardless of being unflattering, per the rule added to §4.1.1. |
| **D19** | 5 | **Stacked hybrid**, an exploratory variant: a logistic regression over paragraph rows with three features — bge cosine to the question, the GNN's OOF paragraph score, and the PPR paragraph score — fit under a **second, nested CV over questions** (folds fixed and stratified as in D17) so the combiner is never scored on questions it was fit on. | The matched RRF pair gives every ranker equal weight at fixed `k=60`, which is deliberately untuned but also unable to learn that on these datasets the vector signal is strong and the graph signal is weak. A learned combiner tests whether the graph scores carry information *complementary* to vector similarity, which is the real question behind C1, without touching the frozen graph construction or the router. Three features only, and no tuning, so it cannot quietly become a fitted result. | 
| **D18** | 1.1, 4.3, 5 | **Findings from the D17 cross-validation that change what later phases should claim, recorded now so they are not presented as discoveries after seeing validation.** (a) The 8-config hyperparameter grid found **no signal**: the whole grid spanned 0.0093 mean OOF edge AP against a fold-to-fold sd of ~0.031–0.036, so the frozen config (lr 1e-3, dropout 0.3, weight decay 1e-1) is an arbitrary pick inside the noise and **must not be described as tuned**. (b) On CV, graph-only GNN is **significantly worse than Vector RAG on AllGold@4** (−0.107 [−0.187, −0.027] MuSiQue; −0.167 [−0.253, −0.080] HotpotQA), while it is **significantly better than PPR and Edge-Sim on edge AP** (+0.053 [+0.019, +0.087] and +0.092 [+0.051, +0.134] MuSiQue; +0.132 and +0.096 HotpotQA), and the `L=0` ablation costs −0.071/−0.098 edge AP. (c) Several §4.3/§4.1 design elements are **measurably inert**: query conditioning (−0.001 both datasets), `H=3` over `H=2` (within noise both), and `para` links (+0.000/+0.005, i.e. removing 33% of edges changes nothing). (d) The learning curve is **flat on MuSiQue** beyond 50% of training data. | These are development-set results on TRAIN under CV; the pre-registered C1–C4 verdicts still belong to M10 on validation and are **not** settled here. Recording them now matters because each one constrains an honest write-up: (a) forbids a tuning narrative; (b) means C1 as a graph-only claim is already failing on development data for the reason D15 predicted — the router scores edges well but that does not convert into better paragraph selection than vector retrieval, because coverage caps it — while C2's "learning matters" half has direct support; (c) means the ablation table should report query conditioning, hop radius and `para` links as design elements that did not earn their place, rather than quietly keeping them; (d) means more extraction would not help MuSiQue, so §6.3's budget is better spent on the entity-linking ceiling than on more questions. |
| **D17** | 3, 6, 11 | **The 10% dev split is replaced by 5-fold cross-validation over the training questions** for all hyperparameter selection and all ablations. Folds are fixed by the global seed, **stratified by hop count (MuSiQue) and question type (HotpotQA)**, and identical across every config so comparisons are paired. Within each fold's training portion a further **10% is held out for early stopping** on edge-level AP; **the fold's test questions are never used for stopping**, so out-of-fold (OOF) scores are clean. Every config is run as **3 seeds × 5 folds per dataset**, and metrics come from seed-averaged OOF scores (§7.4). The validation split remains untouched. | A 10% carve of 150 training questions leaves a **15-question** dev set. AllGold@4 on 15 questions takes only 16 distinct values, which is why all three M6 stage-A MuSiQue seeds reported exactly 0.200 despite producing different rankings (their top-4 agreed on only 5/15 questions and matched raw pool order on 0/15). Early stopping was already moved to edge-level AP for this reason, but AP on 15 questions is still noisy enough that stage A's best epoch landed at 1–3 of 6–8 epochs run. Ablation effects are expected to be smaller than that noise, so selecting configs or judging ablations on it would be measuring sampling error. CV uses every training question for evaluation exactly once, gives per-question OOF scores for paired bootstrap, and exposes fold-to-fold and seed-to-seed spread instead of hiding it. Cost is 15 runs per config per dataset, which the stage-A timings (2–9 s per run) make affordable. |
| **D16** | 4.1.1, 13 | **MuSiQue 3-hop validation connectivity is materially worse than 3-hop train, and no configuration change was made in response.** Validation 3-hop: **5.3** components per question against ~2.1 on train, pool coverage 71.6% vs 86.6%, unconditional ceiling **20.0%** vs 48.0%, conditional-on-all-gold-in-pool **25.0%** vs 72.7%. | The config was already frozen when this was measured (D12 reserves exactly one validation look), so acting on it would be tuning on eval. It is reported as a limitation instead. It is **not** a pooling artefact: 3-hop validation all-gold-in-pool is a healthy 80.0%, above 4-hop's 54.0%, so the gold paragraphs are present and the ego-graph simply fails to reach them. Any MuSiQue 3-hop result must be read against this, and the per-hop breakdown must always be shown rather than folded into a MuSiQue average. |
| **D14** | 4.1 | Ego-graph caps raised from **200 nodes / 400 edges to 300 nodes / 800 edges**. | The §4.1.1 connectivity mechanisms roughly triple edge count (MuSiQue mean 37 → 415 edges, HotpotQA 46 → 378). At 200/400 the edge cap binds on **67.4%** of MuSiQue train questions and 23.7% of HotpotQA's, so the measured coverage would reflect truncation rather than reachability — the cap would hide the failure instead of revealing it. Measured on train at 300/800: cap-binding **0.0%** on both datasets, pool coverage unchanged (88.2% → 88.3%), positive-edge rate slightly better (19.5% → 18.1% with the fuller graphs retained). Swept 200/400, 300/800, 400/1200, 500/1600: everything at or above 300/800 is identical, so 300/800 is the smallest cap that stops binding rather than a tuned value. |
| **D13** | 4.1.1 | The §4.1.1 connectivity target was **MISSED on MuSiQue and is accepted as missed, not passed**: the target is ≥90% of pool paragraphs carrying an ego-graph edge on train, and the frozen configuration reaches **88.3%** (HotpotQA reaches 99.5%, which does meet it). No configuration in the §4.1.1 ordering met the target on both datasets. | Accepted rather than chased because the residual is **reachability at H=3, not missing triples**: only **0.2%** of MuSiQue pool paragraphs yield no usable triple, so the hard coverage ceiling is 99.8% and the 11.5-point gap is entirely paragraphs sitting more than H hops from any seed. Closing it would mean raising `H`, which changes the §4.3 node feature width (`onehot(hop_dist 0..H)` widens, so the input dimension changes and every trained checkpoint and ablation would have to be re-run), for 1.7 points of coverage on one dataset. The AllGold@4 ceiling is already lifted from 32.6% to 77.8% on MuSiQue train and from 65.2% to 100% on HotpotQA, which is enough for routing quality rather than reachability to be what C1 measures. **Any report must state this as a missed target with the 88.3% figure, never as a satisfied one.** |
| **D12** | 3, 4.1, 12 | **Connectivity settings are chosen on TRAIN and DEV statistics only, frozen, and only then measured once on validation.** The disconnected-component problem was itself discovered from *validation*-split ceilings (M5's first report), which is a look at eval data that §12 forbids tuning on. That look is recorded here as having happened; nothing was tuned on it. Every subsequent choice — which connectivity mechanism, the `same_as` threshold, the token-overlap rule — is made against train/dev numbers, written into the config, and the validation ceilings are then re-run exactly once for reporting. | §12: "Never tune on the eval split." The fragmentation was found while reporting the §4.1 graph ceiling, which M5 computes on both splits, so the validation numbers were visible before a fix was designed. Pretending otherwise would be dishonest; the mitigation is to make every *decision* train/dev-only from here, so the reported validation ceiling stays an out-of-sample number rather than something a threshold was fitted to. |
| **D11** | 6.3, 7.3 | The M4 extraction timings — **t_p ≈ 1.90 s, 6,489 paragraphs, 3.30 h wall** — are a **planning figure only** and **must not be used as the indexing cost in the §7.3 efficiency tables or the §1.1 C3 comparison**. The indexing cost reported in any write-up must come from a dedicated measurement run taken under the §7.3 conditions (idle baseline, no competing load, swap growth within the 500 MB rule, which D10 keeps strict for exactly this purpose). | The run executed under heavy sustained system memory pressure: swap grew to a peak of **+8,239 MB** and stayed above 6 GB for most of the run, against a §7.3 invalidation threshold of 500 MB. §7.3 states plainly that any swap growth > 500 MB invalidates a run, so this run is invalid *as a cost measurement* even though it is perfectly valid as an *extraction* (0.02% JSON failure, 99.87%/100% gold coverage — the triples are correct and are kept). Latency measured while the OS is paging the process is not the pipeline's latency: t_p drifted 1.59→2.19 s with swap, and the same 50-paragraph pilot measured 7.24 s under pressure versus 1.91 s idle, a 3.7× swing. Reporting 3.30 h as the indexing cost would overstate it and make the C3 routing-vs-indexing comparison unreproducible. |
| **D10** | 7.3 | System-wide swap growth is **no longer an abort condition for extraction (M4)**. Extraction aborts instead on (a) own process-tree RSS, excluding Ollama, staying ≥1.5× baseline for 10 consecutive ticks, (b) rolling seconds/paragraph over the last 50 paragraphs exceeding 1.5× baseline, (c) OS memory pressure critical on 3 consecutive readings. System-wide swap and available RAM are **logged every 60 s as a warning**. The strict system-wide swap rule is **retained unchanged for the M7/M8(/M7b) efficiency measurement runs**. | Swap is a global counter, so on an 8 GB machine it reports other processes' behaviour as if it were ours. It aborted the first M4 attempt at 250/6489 paragraphs on a +979 MB jump while our own RSS was *falling* (1448 → 1089 MB) and t_p was flat at 1.65 s with 0% failures and a 2.97 h projection — the machine was paging us out for an unrelated allocation, the exact opposite of our run thrashing. The replacement conditions measure our own workload. For M7/M8 the rule stands: there, memory *is* the quantity under measurement (§7.3), not ambient noise. |
| **D9** | 2 | The `OLLAMA_NUM_PARALLEL=2` option is probed by issuing 2 concurrent requests, not by restarting the Ollama server with a different environment. | The setting is server-side and the server is the user's. The probe measures the concurrency actually available. Measured twice: **0.86×** and **1.01×** — never faster, so extraction stays sequential as §6.2 defaults. |

### Settings chosen by measurement, not deviation

| Setting | Value | Evidence |
| :-- | :-- | :-- |
| `DEVICE` | `cpu` | §6.4 benchmark: cpu **0.382 s/epoch** vs mps **0.534 s/epoch** (mps 0.72×, i.e. slower) on an EdgeRouter-shaped synthetic epoch. §6.4 anticipated exactly this for small graphs. |
| `max_triples` | **6** | Sweep over {6,5,4} at t_p {1.94, 1.48, 1.33} s. All three fit the 5 h cap, so the larger cap costs nothing; 6 gives the densest graphs (5.5 vs 3.8 triples/paragraph). Answer-in-graph was 35% / 25% / 25% on a hop-stratified n=20, but that 10-point gap is **two questions** and is not treated as evidence of a quality win. |
| `N` | **150** per (dataset, split) | Governor allows 157 at t_p=1.905 s; §3's default target of 150 is the binding constraint. 7,200 paragraphs ≈ **3.81 h** nominal, ~3.46 h after dedupe. |

### Known weak spot (not a deviation)

3-hop MuSiQue answer-in-graph is **14% (1/7)** at every `max_triples` value tested,
against 43% for 2-hop and 50% for 4-hop. Since §4.1's graph ceiling requires the
answer to appear as a node, this caps what any router can achieve on 3-hop
questions. It is a property of the 1.5B extractor, not of a setting chosen here,
and it should be reported with the C1 result rather than discovered at M10.
