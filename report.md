# Resource-Efficient Subgraph Routing via Light GNNs for Multi-Hop GraphRAG

**Status:** complete through M8 plus the 3B strength check. All numbers below are
generated from `results/` by `scripts/make_report_tables.py`; each table cites its
source file. Every claim C1–C4 was pre-registered in `agent.md` §1.1 before any
model was trained.

---

## Abstract

We train a small, query-conditioned, edge-aware GNN to score the edges of a
per-question knowledge graph, so that multi-hop supporting evidence can be
selected with no LLM call at query time. The graph for each question is built from
that question's own candidate paragraphs (the distractor-pool setting), with
triples extracted offline by a 1.5B SLM. On MuSiQue and HotpotQA validation
(n=150 each) the router learns edge scoring that is clearly better than a
non-learned graph baseline — Edge-Precision@10 of 0.6053
versus PPR's 0.4933 on MuSiQue
and 0.7487 versus
0.5493 on HotpotQA — but that
advantage does not reach paragraph selection: graph-only routing does not exceed
plain vector retrieval on AllGold@4 on either dataset. Routing is 35–61× cheaper
than LLM path tracing and uses no query-time LLM tokens. Of the four
pre-registered claims, **C1 is not supported, C2 is inconclusive, C3's cost half
is supported with its quality half supported on HotpotQA and inconclusive on
MuSiQue, and C4 is supported** subject to a memory-accounting caveat. The decisive
constraint is measured and reported: the ego-graph can only reach
0.791 of MuSiQue's pool
paragraphs, capping AllGold@4 for any graph method at
0.380 while vector retrieval
faces no such bound.

## 1. Setup

**Scope of the claim.** Graphs are per-question, built from each question's own
candidate paragraphs. This is an easier setting than a single corpus-level
GraphRAG graph, and none of these results should be read as transferring to one.

**Datasets.** MuSiQue (answerable) and HotpotQA (distractor), official train and
validation splits, question ids asserted disjoint. 150 train + 150 validation
questions per dataset. MuSiQue is stratified by hop count (2/3/4); HotpotQA keeps
its natural bridge/comparison ratio and its train split is restricted to
`level=hard` to match the all-hard validation split. Pool size is per dataset:
MuSiQue 14 of its 20 candidates by cosine to the question, HotpotQA all 10. Pool
construction never reads a gold label, and a unit test asserts that flipping every
gold flag leaves the pool unchanged. Measured pool recall:
0.791 (MuSiQue validation) and
1.000 (HotpotQA).

**Extractor.** Ollama `qwen2.5:1.5b-instruct-q4_K_M`, greedy, JSON-schema output
with the triple count enforced by the decoding grammar, paragraphs truncated to
120 words. 6,489 unique paragraphs extracted, JSON failure
rate 0.02%, gold-paragraph extraction coverage
0.999 (MuSiQue) and
1.000 (HotpotQA).

**Graph.** Nodes are normalized entity strings; one directed edge per triple plus
a flagged reverse. Because triples extracted paragraph-by-paragraph share almost
no entity surface forms, the raw per-question graph arrives as roughly one
component per paragraph. Three mechanisms were added to connect it (§4.1.1):
paragraph-title hub nodes, `same_as` edges between near-identical entities, and
pid-keyed paragraph nodes. Seeds are nodes whose normalized name is a substring of
the question, with an embedding fallback; the ego-graph is a 3-hop undirected BFS
ball capped at 300 nodes / 800 edges.

**Router.** `EdgeRouter`: input linear → 2 × `GATv2Conv` (hidden 128, 4 heads,
edge-conditioned) with residual, LayerNorm and dropout → MLP edge head over
`[h_u ; h_v ; proj(edge_attr) ; proj(q)]`. 431,745 parameters. Node features are
778-dim, edge features 390-dim, including kind indicators that let the model
distinguish extracted triples from connectivity scaffolding. Supervision is weak
and provenance-based: an edge is positive iff its source paragraph is gold, and
**only extracted edges are supervised or scored into paragraphs** — scaffolding
adds reachability, never evidence. Loss is `BCEWithLogitsLoss` with `pos_weight`
from training triple edges (4.447 / 4.081).

**Baselines.** Vector RAG (bge cosine over the pool, the primary baseline), PPR
from the seed nodes, Edge-Sim (triple-text cosine, no graph structure), LLM-Trace
(≤3 SLM calls, ≤20 candidate **triple** edges per step by degree), and a
gold-paragraph oracle for the reader ceiling. Two matched hybrids fuse a graph
ranker with Vector RAG by reciprocal-rank fusion at a fixed, untuned k=60, and a
learned stacked combiner (D19) is reported as an exploratory variant.

**Model selection.** All hyperparameters and ablations were chosen on 5-fold
cross-validation over the *training* questions (D17), not on validation. The
validation split was scored exactly once, after the configuration was frozen, and
every read is recorded in `results/validation_access.log`.

**One disclosure that matters for interpretation.** Validation *was* inspected
before the configuration was frozen — specifically, the §4.1 graph-ceiling and
connectivity statistics were computed on both splits while diagnosing the
fragmentation problem (D12). No threshold or hyperparameter was fitted to those
numbers, and every subsequent decision was made on train/dev only, but the
reported validation ceilings are not strictly naive and we do not present them as
such.

## 2. Pre-registered claims and verdicts

> **Extension (M11, D27).** These verdicts are the pre-registered 150-question ones and are *not* revised. Section 10 reports an independent replication on 350 **new** validation questions per dataset and says where the two disagree.

| claim | pre-registered test | verdict |
| :-- | :-- | :-- |
| **C1** | GNN retrieves more complete multi-hop evidence than vector retrieval; AllGold@4-conditional, paired-bootstrap CI of the difference excludes 0 | **NOT SUPPORTED** |
| **C2** | Graph structure and learning both matter: GNN beats PPR and the 0-layer ablation on AllGold@4 | **INCONCLUSIVE** |
| **C3** | Routing ≥10× cheaper than LLM-Trace at AllGold@4 no more than 3 points lower | **cost half SUPPORTED; quality half SUPPORTED (HotpotQA), INCONCLUSIVE (MuSiQue)** |
| **C4** | Peak process-tree RSS under 6 GB for every method | **SUPPORTED**, with a memory-accounting caveat |

**C1.** Graph-only versus Vector RAG on the conditional metric:
MuSiQue +0.0000 [-0.0901, +0.0901] against a graph ceiling of
0.380 on n=111; HotpotQA
-0.2267 [-0.3200, -0.1267] against a ceiling of 0.993.
The MuSiQue difference is a precise tie; the HotpotQA difference excludes zero in
the *unfavourable* direction. The matched hybrid pair (GNN+Vector − PPR+Vector) is
+0.0090 [-0.0180, +0.0450] and +0.0200 [-0.0400, +0.0800] — no gain whose CI excludes
zero. C1 is not supported on either dataset.

**C2.** GNN − PPR on AllGold@4 includes zero on both datasets, so the claim as
registered is inconclusive. Its 0-layer half was never run on validation, being a
§5 ablation; under cross-validation on train, `L=0` cost −0.071 and −0.098 edge AP
with CIs excluding zero. Post-hoc edge-level evidence is reported in §5.

**C3.** The cost half holds comfortably. The quality half is tested as
(GNN − Trace) ≥ −0.03: MuSiQue -0.0600 [-0.1333, +0.0067] straddles
the margin, so inconclusive; HotpotQA +0.2000 [+0.1067, +0.2933]
lies entirely above it, so supported. Query-time LLM tokens are 0 for the GNN by
construction against roughly 1,100–1,232 for LLM-Trace.

**C4.** Peak process-tree RSS including Ollama is
3482 MB (3.40 GB), under the
6 GB limit. See §6 for why that number is a floor rather than a faithful total.

## 3. Retrieval results

Ceilings appear in every table beneath the graph methods they bound. The ceiling
is the fraction of questions whose every gold paragraph has at least one extracted
edge in the ego-graph — a paragraph with no edge cannot be ranked by a graph
method at all.

### MuSiQue validation (n=150)

*Source: `results/m7a_report.json`, `results/val_llm_trace_musique.jsonl`*

| method | SP-Recall@4 | AllGold@4 | AllGold@4-cond | Edge-Prec@10 | Path-Recall |
| :-- | --: | --: | --: | --: | --: |
| Vector RAG | 0.6333 | 0.267 | 0.252 | — | — |
| PPR | 0.5506 | 0.240 | 0.234 | 0.4933 | 0.0400 |
| Edge-Sim | 0.5517 | 0.213 | 0.207 | 0.4767 | 0.0733 |
| GNN router (graph-only) | 0.5500 | 0.240 | 0.252 | 0.6053 | 0.0600 |
| GNN+Vector (RRF k=60) | 0.6239 | 0.273 | 0.261 | 0.6053 | 0.0600 |
| PPR+Vector (RRF k=60) | 0.6167 | 0.267 | 0.252 | 0.4933 | 0.0400 |
| GNN+hybrid-tail *(≡ GNN)* | 0.5500 | 0.240 | 0.252 | 0.6053 | 0.0600 |
| PPR+hybrid-tail *(≡ PPR)* | 0.5506 | 0.240 | 0.234 | 0.4933 | 0.0400 |
| Stacked combiner (D19) | 0.6533 | 0.293 | 0.279 | 0.6053 | 0.0600 |
| LLM-Trace (1.5B) | 0.6433 | 0.300 | — | — | — |
| **Ceiling for graph methods** | — | **0.380** | — | — | **0.293** |
| *context* | pool recall 0.898 | extraction coverage 0.998 | graph pool coverage 0.791 | n=150 | n_cond=111 |

### HotpotQA validation (n=150)

*Source: `results/m7a_report.json`, `results/val_llm_trace_hotpotqa.jsonl`*

| method | SP-Recall@4 | AllGold@4 | AllGold@4-cond | Edge-Prec@10 | Path-Recall |
| :-- | --: | --: | --: | --: | --: |
| Vector RAG | 0.8933 | 0.793 | 0.793 | — | — |
| PPR | 0.7067 | 0.533 | 0.533 | 0.5493 | 0.1400 |
| Edge-Sim | 0.8033 | 0.613 | 0.613 | 0.6860 | 0.1867 |
| GNN router (graph-only) | 0.7500 | 0.567 | 0.567 | 0.7487 | 0.1533 |
| GNN+Vector (RRF k=60) | 0.8433 | 0.713 | 0.713 | 0.7487 | 0.1533 |
| PPR+Vector (RRF k=60) | 0.8367 | 0.693 | 0.693 | 0.5493 | 0.1400 |
| GNN+hybrid-tail *(≡ GNN)* | 0.7500 | 0.567 | 0.567 | 0.7487 | 0.1533 |
| PPR+hybrid-tail *(≡ PPR)* | 0.7067 | 0.533 | 0.533 | 0.5493 | 0.1400 |
| Stacked combiner (D19) | 0.8900 | 0.793 | 0.793 | 0.7487 | 0.1533 |
| LLM-Trace (1.5B) | 0.6200 | 0.367 | — | — | — |
| **Ceiling for graph methods** | — | **0.993** | — | — | **0.500** |
| *context* | pool recall 1.000 | extraction coverage 1.000 | graph pool coverage 0.994 | n=150 | n_cond=150 |

The two `+hybrid-tail` rows are **identical to their base methods by
construction**: the hybrid tail only reorders paragraphs that rank below every
graph-scored one, so with K=4 and at least four scored paragraphs on nearly every
question it cannot alter the top-4.

### By hop bucket and question type

*Source: `results/m7a_report.json`*

| method | 2hop (n=50) | 3hop (n=50) | 4hop (n=50) |
| :-- | --: | --: | --: |
| Vector RAG | 0.480 | 0.260 | 0.060 |
| PPR | 0.500 | 0.140 | 0.080 |
| Edge-Sim | 0.480 | 0.120 | 0.040 |
| GNN router (graph-only) | 0.480 | 0.160 | 0.080 |
| GNN+Vector (RRF k=60) | 0.520 | 0.240 | 0.060 |
| PPR+Vector (RRF k=60) | 0.480 | 0.240 | 0.080 |
| GNN+hybrid-tail *(≡ GNN)* | 0.480 | 0.160 | 0.080 |
| PPR+hybrid-tail *(≡ PPR)* | 0.500 | 0.140 | 0.080 |
| Stacked combiner (D19) | 0.520 | 0.280 | 0.080 |
| **Ceiling (graph)** | **0.720** | **0.200** | **0.220** |

*Source: `results/m7a_report.json`*

| method | bridge (n=122) | comparison (n=28) |
| :-- | --: | --: |
| Vector RAG | 0.746 | 1.000 |
| PPR | 0.557 | 0.429 |
| Edge-Sim | 0.582 | 0.750 |
| GNN router (graph-only) | 0.525 | 0.750 |
| GNN+Vector (RRF k=60) | 0.664 | 0.929 |
| PPR+Vector (RRF k=60) | 0.680 | 0.750 |
| GNN+hybrid-tail *(≡ GNN)* | 0.525 | 0.750 |
| PPR+hybrid-tail *(≡ PPR)* | 0.557 | 0.429 |
| Stacked combiner (D19) | 0.746 | 1.000 |
| **Ceiling (graph)** | **1.000** | **0.964** |

The 3-hop MuSiQue column is the clearest single result in the study: Vector RAG
scores 0.260 against a
graph ceiling of 0.200. It
exceeds the bound that constrains every graph method, because it is not bounded by
reachability. The 4-hop column is exploratory at n=50 and every method sits
between 0.04 and 0.12.

Conditioning on all gold paragraphs surviving pooling (D4) changes the
denominators: MuSiQue validation conditions to n=111 of 150, and the ceiling there
is 51.4%.

## 4. Reader results (M8)

Every condition receives the same K=4 paragraph budget and the same prompt
template, greedy decoding, 24-token cap, so differences come from routing only.

*Source: `results/m8_emf1.json`*

| condition | MuSiQue EM | MuSiQue F1 | HotpotQA EM | HotpotQA F1 |
| :-- | --: | --: | --: | --: |
| Vector RAG | 0.180 | 0.269 | 0.473 | 0.582 |
| PPR | 0.113 | 0.187 | 0.387 | 0.506 |
| LLM-Trace (1.5B) | 0.153 | 0.232 | 0.427 | 0.534 |
| GNN router | 0.113 | 0.191 | 0.420 | 0.516 |
| GNN+Vector (RRF) | 0.173 | 0.252 | 0.480 | 0.577 |
| Stacked (D19) | 0.153 | 0.250 | 0.460 | 0.568 |
| **Gold-paragraph oracle** | **0.367** | **0.464** | **0.560** | **0.697** |

The oracle is the ceiling on both datasets and both metrics, which is the sanity
check this table exists to pass. Vector RAG − oracle excludes zero on both
datasets, so substantial reader headroom remains and the reader is not saturating
the comparison.

*Source: `results/m8_emf1.json`*

| comparison | metric | MuSiQue | HotpotQA | registered? |
| :-- | :-- | :-- | :-- | :-- |
| GNN − Vector RAG | EM | -0.0667 [-0.1333, +0.0000] *(borderline: a bound is exactly 0.000)* | -0.0533 [-0.1267, +0.0200] | pre-registered |
| GNN − Vector RAG | F1 | -0.0780 [-0.1467, -0.0091] | -0.0666 [-0.1377, +0.0066] | pre-registered |
| GNN − oracle | EM | -0.2533 [-0.3333, -0.1733] | -0.1400 [-0.2133, -0.0667] | pre-registered |
| GNN − oracle | F1 | -0.2722 [-0.3509, -0.1970] | -0.1817 [-0.2583, -0.1093] | pre-registered |
| GNN − PPR | EM | +0.0000 [-0.0400, +0.0400] | +0.0333 [-0.0133, +0.0867] | **post-hoc** |
| GNN − PPR | F1 | +0.0045 [-0.0404, +0.0488] | +0.0093 [-0.0430, +0.0623] | **post-hoc** |
| GNN − LLM-Trace | EM | -0.0400 [-0.1067, +0.0267] | -0.0067 [-0.0800, +0.0667] | **post-hoc** |
| GNN − LLM-Trace | F1 | -0.0408 [-0.1080, +0.0269] | -0.0187 [-0.0942, +0.0542] | **post-hoc** |
| GNN+Vector − GNN | EM | +0.0600 [+0.0067, +0.1133] | +0.0600 [+0.0067, +0.1133] | **post-hoc** |
| GNN+Vector − GNN | F1 | +0.0611 [+0.0041, +0.1183] | +0.0612 [+0.0032, +0.1207] | **post-hoc** |
| Stacked − GNN+Vector | EM | -0.0200 [-0.0733, +0.0333] | -0.0200 [-0.0733, +0.0267] | **post-hoc** |
| Stacked − GNN+Vector | F1 | -0.0027 [-0.0551, +0.0491] | -0.0092 [-0.0653, +0.0447] | **post-hoc** |
| Vector RAG − oracle | EM | -0.1867 [-0.2667, -0.1133] | -0.0867 [-0.1600, -0.0200] | pre-registered |
| Vector RAG − oracle | F1 | -0.1942 [-0.2717, -0.1200] | -0.1152 [-0.1883, -0.0457] | pre-registered |
| Stacked − Vector RAG | EM | -0.0267 [-0.0600, +0.0000] *(borderline: a bound is exactly 0.000)* | -0.0133 [-0.0600, +0.0333] | pre-registered |
| Stacked − Vector RAG | F1 | -0.0196 [-0.0572, +0.0159] | -0.0146 [-0.0607, +0.0306] | pre-registered |

Two results deserve emphasis and both are **post-hoc**. `GNN+Vector − GNN` is
about +0.060 EM with a CI excluding zero on *both* datasets: fusing with vector
retrieval helps the reader even where it did not help AllGold@4. And `GNN − Vector
RAG` EM on MuSiQue is −0.0667 with an upper bound of **exactly 0.000** — a
borderline interval that should not be read as a finding in either direction,
though its F1 counterpart does exclude zero.

## 5. Ablations and cross-validation findings

All of this is on train, under 5-fold CV, with a fold-to-fold edge-AP standard
deviation of 0.031 (MuSiQue) and 0.038 (HotpotQA) as the noise floor.

*Source: `results/m6_ablations.json`*

| ablation | MuSiQue Δ edge AP | verdict | HotpotQA Δ edge AP | verdict |
| :-- | --: | :-- | --: | :-- |
| L=0 (no message passing) | -0.0707 [-0.0967, -0.0462] | significant | -0.0982 [-0.1246, -0.0721] | significant |
| L=1 | -0.0208 [-0.0375, -0.0048] | significant | -0.0034 [-0.0132, +0.0060] | within noise |
| L=3 | -0.0044 [-0.0149, +0.0053] | within noise | -0.0059 [-0.0168, +0.0051] | within noise |
| SAGE-mean instead of GATv2 | -0.0155 [-0.0296, -0.0020] | significant | -0.0312 [-0.0473, -0.0158] | significant |
| no query conditioning | -0.0010 [-0.0081, +0.0063] | within noise | -0.0011 [-0.0061, +0.0039] | within noise |
| H=2 (hop_dist masking) | +0.0158 [-0.0060, +0.0383] | within noise | -0.0012 [-0.0110, +0.0090] | within noise |
| drop `para` links | +0.0001 [-0.0174, +0.0186] | within noise | +0.0053 [-0.0058, +0.0162] | within noise |
| drop `same_as` links | -0.0050 [-0.0173, +0.0075] | within noise | -0.0005 [-0.0107, +0.0096] | within noise |
| drop title hubs | -0.0212 [-0.0382, -0.0046] | significant | -0.0309 [-0.0481, -0.0134] | significant |
| triples only (no §4.1.1 scaffolding) | -0.0171 [-0.0334, -0.0012] | significant | -0.0342 [-0.0548, -0.0137] | significant |

Three design elements are measurably inert: query conditioning (−0.001 on both),
the H=3 radius over H=2, and the `para` links — which are 33% of all edges and the
mechanism that lifted evidence-only ego coverage to
84.8% on MuSiQue train, yet whose removal
changes edge AP by nothing *(source: `results/m5_diagnostics.json`)*. Message passing is the one component with a large, unambiguous effect.

*Source: `results/m6_grid.json`*

| lr | dropout | weight decay | MuSiQue edge AP | HotpotQA edge AP | mean |
| :-- | --: | --: | --: | --: | --: |
| 0.001 | 0.1 | 0.01 | 0.5804 | 0.6881 | 0.6342 |
| 0.001 | 0.1 | 0.1 | 0.5831 | 0.6881 | 0.6356 |
| 0.001 | 0.3 | 0.01 | 0.5837 | 0.6859 | 0.6348 |
| 0.001 **(selected)** | 0.3 | 0.1 | 0.5864 | 0.6872 | 0.6368 |
| 0.0003 | 0.1 | 0.01 | 0.5886 | 0.6794 | 0.6340 |
| 0.0003 | 0.1 | 0.1 | 0.5893 | 0.6793 | 0.6343 |
| 0.0003 | 0.3 | 0.01 | 0.5799 | 0.6752 | 0.6275 |
| 0.0003 | 0.3 | 0.1 | 0.5791 | 0.6773 | 0.6282 |
| **spread** | | | | | **0.0093** vs fold sd **0.0312** |

The entire 8-configuration grid spans less than half the fold-to-fold noise, and
the per-dataset winners disagree. The selected configuration is an arbitrary pick
inside that noise and **is not a tuned result**; the useful finding is that the
model is insensitive to these hyperparameters in this range.

*Source: `results/m6_curve.json`*

| fraction of fold training set | MuSiQue edge AP | MuSiQue AllGold@4 | HotpotQA edge AP | HotpotQA AllGold@4 |
| :-- | --: | --: | --: | --: |
| 25% | 0.5660 | 0.153 | 0.6391 | 0.520 |
| 50% | 0.5834 | 0.173 | 0.6624 | 0.553 |
| 75% | 0.5823 | 0.167 | 0.6683 | 0.560 |
| 100% | 0.5864 | 0.153 | 0.6872 | 0.567 |

MuSiQue is saturated beyond 50% of the training data. More extraction would not
help it; the binding constraint is elsewhere.

## 6. Why graph-only routing does not exceed vector retrieval

Three measured diagnostics, not speculation.

**Reachability caps.** The connectivity fix worked and was still not enough:

*Source: `results/m5_before_after_train.json`*

| TRAIN metric | MuSiQue before | MuSiQue after | HotpotQA before | HotpotQA after |
| :-- | --: | --: | --: | --: |
| components per question | 13.5 | 2.3 | 7.1 | 1.0 |
| BFS share of full graph | 23.4% | 82.9% | 41.5% | 99.7% |
| pool paragraphs with an ego edge | 29.4% | 88.3% | 46.8% | 99.5% |
| AllGold@4 ceiling (conditional on pooling) | 32.6% | 77.8% | 65.2% | 100.0% |
| positive-edge rate | 35.9% | 18.1% | 35.6% | 18.6% |
| cap-binding rate | 0.0% | 0.0% | 0.0% | 0.0% |

Even after it, only 0.791
of MuSiQue **validation** pool paragraphs carry an extracted edge inside the
ego-graph (84.8% on train), giving an
AllGold@4 ceiling of 0.380.
Vector RAG's own bound is all-gold-in-pool,
0.740. A perfect router
is therefore capped below a mediocre baseline. Diagnosis showed the residual is
hop-radius reachability, not missing triples: only
0.2% of MuSiQue pool paragraphs yield no
usable triple, so the hard coverage ceiling is
99.8% *(source: `results/m5_diagnostics.json`)*.

**Noisy scaffolding links.** Of the cross-paragraph links the connectivity
mechanisms add on train, only 3.6% (MuSiQue)
and 2.6% (HotpotQA) join gold to gold, while
62.3% and
61.3% join distractor to
distractor. The `para` mechanism is both the bulk of them and the noisiest. This
matches the ablation result that removing `para` costs nothing.

**The learned combiner's own weights.** Given vector cosine, the GNN paragraph
score and the PPR score as three features, a logistic regression fitted under
nested CV assigns:

*Source: `results/m6_stack.json`, `results/m6_stack_final.json`*

| feature | MuSiQue coefficient (CV) | HotpotQA coefficient (CV) | MuSiQue (final fit) | HotpotQA (final fit) |
| :-- | --: | --: | --: | --: |
| `vector_cos` | +3.083 | +4.719 | +3.101 | +4.798 |
| `gnn_score` | +0.748 | +1.941 | +0.755 | +1.962 |
| `ppr_score` | +0.265 | -0.290 | +0.255 | -0.327 |

The GNN feature gets a clearly non-zero positive weight, so the graph score does
carry signal — but vector cosine outweighs it by roughly 4× on MuSiQue and 2.4× on
HotpotQA, and the combination still does not beat vector similarity alone. That is
the most direct evidence available that the graph signal is real but not
complementary enough to change paragraph selection.

## 7. Efficiency

*Source: `results/m7_indexing_cost.json`, `results/val_vector_musique_stats.json`, `results/val_ppr_musique_stats.json`, `results/val_edge_sim_musique_stats.json`, `results/val_gnn_musique_stats.json`, `results/val_gnn_rrf_musique_stats.json`, `results/val_ppr_rrf_musique_stats.json`, `results/val_gnn_hybrid_tail_musique_stats.json`, `results/val_ppr_hybrid_tail_musique_stats.json`, `results/val_stacked_musique_stats.json`, `results/val_vector_hotpotqa_stats.json`, `results/val_ppr_hotpotqa_stats.json`, `results/val_edge_sim_hotpotqa_stats.json`, `results/val_gnn_hotpotqa_stats.json`, `results/val_gnn_rrf_hotpotqa_stats.json`, `results/val_ppr_rrf_hotpotqa_stats.json`, `results/val_gnn_hybrid_tail_hotpotqa_stats.json`, `results/val_ppr_hybrid_tail_hotpotqa_stats.json`, `results/val_stacked_hotpotqa_stats.json`, `results/m7c_llm_trace_musique_full.json`, `results/m7c_llm_trace_hotpotqa_full.json`, `results/read_vector_musique.json`, `results/read_ppr_musique.json`, `results/read_llm_trace_musique.json`, `results/read_gnn_musique.json`, `results/read_gnn_rrf_musique.json`, `results/read_stacked_musique.json`, `results/read_oracle_musique.json`, `results/read_vector_hotpotqa.json`, `results/read_ppr_hotpotqa.json`, `results/read_llm_trace_hotpotqa.json`, `results/read_gnn_hotpotqa.json`, `results/read_gnn_rrf_hotpotqa.json`, `results/read_stacked_hotpotqa.json`, `results/read_oracle_hotpotqa.json`*

| stage | method | median | p95 | LLM tokens/q | peak tree RSS | swap growth | valid? |
| :-- | :-- | --: | --: | --: | --: | --: | :-- |
| indexing | extraction (all graph methods) | 1.603 s/paragraph | 2.153 s | 408 /paragraph | 1882 MB | +0 MB | yes |
| indexing | entity/relation embedding | 23.3 ms/question | — | 0 | 885 MB | +0 MB | yes |
| routing (musique) | Vector RAG | 0.2 ms | 7.7 ms | 0 | 300 MB | +0 MB | yes |
| routing (musique) | PPR | 40.3 ms | 70.8 ms | 0 | 801 MB | +0 MB | yes |
| routing (musique) | Edge-Sim | 10.6 ms | 18.0 ms | 0 | 1196 MB | +0 MB | yes |
| routing (musique) | GNN router (graph-only) | 43.8 ms | 83.3 ms | 0 | 852 MB | +0 MB | yes |
| routing (musique) | GNN+Vector (RRF k=60) | 44.0 ms | 88.2 ms | 0 | 872 MB | +0 MB | yes |
| routing (musique) | PPR+Vector (RRF k=60) | 38.7 ms | 70.9 ms | 0 | 822 MB | +0 MB | yes |
| routing (musique) | GNN+hybrid-tail | 44.5 ms | 89.7 ms | 0 | 875 MB | +0 MB | yes |
| routing (musique) | PPR+hybrid-tail | 40.1 ms | 122.7 ms | 0 | 818 MB | +0 MB | yes |
| routing (musique) | Stacked combiner (D19) | 48.0 ms | 87.5 ms | 0 | 882 MB | +0 MB | yes |
| routing (musique) | LLM-Trace (1.5B) *indicative* | 1543 ms | 2041 ms | 1101 | 2005 MB | +3389 MB | **NO** |
| reading (musique) | Vector RAG | 0.67 s | 0.94 s | 619 | 2608 MB | -122 MB | yes |
| reading (musique) | PPR | 0.64 s | 1.03 s | 600 | 1680 MB | +1587 MB | **NO** |
| reading (musique) | LLM-Trace (1.5B) | 0.51 s | 0.89 s | 615 | 1940 MB | +568 MB | **NO** |
| reading (musique) | GNN router | 0.54 s | 0.86 s | 604 | 2016 MB | +830 MB | **NO** |
| reading (musique) | GNN+Vector (RRF) | 0.47 s | 0.88 s | 614 | 2152 MB | +507 MB | **NO** |
| reading (musique) | Stacked (D19) | 0.37 s | 0.72 s | 613 | 2300 MB | +156 MB | yes |
| reading (musique) | Gold-paragraph oracle | 0.41 s | 0.86 s | 457 | 2072 MB | +926 MB | **NO** |
| routing (hotpotqa) | Vector RAG | 0.5 ms | 7.5 ms | 0 | 293 MB | +0 MB | yes |
| routing (hotpotqa) | PPR | 27.5 ms | 69.5 ms | 0 | 791 MB | +0 MB | yes |
| routing (hotpotqa) | Edge-Sim | 6.2 ms | 13.8 ms | 0 | 1477 MB | +0 MB | yes |
| routing (hotpotqa) | GNN router (graph-only) | 35.6 ms | 105.6 ms | 0 | 850 MB | +0 MB | yes |
| routing (hotpotqa) | GNN+Vector (RRF k=60) | 38.2 ms | 101.6 ms | 0 | 865 MB | +0 MB | yes |
| routing (hotpotqa) | PPR+Vector (RRF k=60) | 29.0 ms | 75.4 ms | 0 | 810 MB | +0 MB | yes |
| routing (hotpotqa) | GNN+hybrid-tail | 37.1 ms | 94.6 ms | 0 | 869 MB | +0 MB | yes |
| routing (hotpotqa) | PPR+hybrid-tail | 29.1 ms | 76.6 ms | 0 | 813 MB | +0 MB | yes |
| routing (hotpotqa) | Stacked combiner (D19) | 38.3 ms | 95.2 ms | 0 | 871 MB | +0 MB | yes |
| routing (hotpotqa) | LLM-Trace (1.5B) *indicative* | 1684 ms | 2035 ms | 1232 | 3482 MB | +2195 MB | **NO** |
| reading (hotpotqa) | Vector RAG | 0.67 s | 0.94 s | 570 | 1904 MB | +841 MB | **NO** |
| reading (hotpotqa) | PPR | 0.60 s | 0.91 s | 564 | 1933 MB | -212 MB | yes |
| reading (hotpotqa) | LLM-Trace (1.5B) | 0.55 s | 0.92 s | 583 | 2178 MB | -103 MB | yes |
| reading (hotpotqa) | GNN router | 0.49 s | 0.87 s | 565 | 2268 MB | -126 MB | yes |
| reading (hotpotqa) | GNN+Vector (RRF) | 0.39 s | 0.77 s | 560 | 2340 MB | -8 MB | yes |
| reading (hotpotqa) | Stacked (D19) | 0.30 s | 0.68 s | 564 | 2493 MB | -131 MB | yes |
| reading (hotpotqa) | Gold-paragraph oracle | 0.25 s | 0.47 s | 295 | 2146 MB | +355 MB | yes |

### What the routing stage includes

*Source: `results/m8_analysis.json`*

| | MuSiQue | HotpotQA |
| :-- | --: | --: |
| M7a routing, full §7.3 scope | 43.8 ms | 35.6 ms |
| — graph assembly | 38.9 ms | 30.9 ms |
| — model forward | 4.3 ms | 4.3 ms |
| M7c figure (forward only, **not** routing) | 4.2 ms | 4.1 ms |

§7.3 defines routing as everything from question and graph to ranked paragraphs,
which includes question embedding, seed linking and ego-graph assembly. Graph
assembly dominates: 38.9 of 43.8 ms on MuSiQue, with the model forward only
4.3 ms. A figure that excludes assembly is not the routing stage.

### LLM-Trace timings are indicative, not valid

*Source: `results/m7b_smoke.json`, `results/m7b_full_musique.json`, `results/m7b_full_hotpotqa.json`, `results/m7c_llm_trace_musique_full.json`, `results/m7c_llm_trace_hotpotqa_full.json`, `results/m8_analysis.json`*

| measurement | MuSiQue median | HotpotQA median | tokens/q | swap growth | §7.3 valid? |
| :-- | --: | --: | --: | --: | :-- |
| M7 first run | 1573 ms | 1764 ms | 1101 / 1232 | +2532 / +3072 MB | **no** |
| D21 redo (same figures) | 1573 ms | 1764 ms | 1101 / 1232 | +2532 / +3072 MB | **no** |
| 10-question smoke | 1639 ms | 2157 ms | 1050 / 1405 | +0 / +0 MB | yes |
| M7c clean start | 1543 ms | 1684 ms | 1101 / 1232 | +3389 / +2195 MB | **no** |
| **C3 ratio (musique)** | GNN 43.8 ms | Trace 1543–1639 ms | — | — | **35.2×–37.4×** |
| **C3 ratio (hotpotqa)** | GNN 35.6 ms | Trace 1684–2157 ms | — | — | **47.4×–60.7×** |

Three of four measurements tripped the §7.3 swap rule, including one that started
from 0 MB swap. On this hardware the stage cannot be measured within the rule: a
1.1 GB model plus the graph-building process exceeds what 8 GB allows. The C3
latency ratio therefore combines **valid** M7a routing medians with **indicative**
LLM-Trace medians, and is quoted as a range. The medians agree within 6% (MuSiQue)
and 28% (HotpotQA) across runs with very different swap behaviour, and the ratio is
35–61×, a margin no plausible paging artefact closes.

## 8. Limitations

1. **Per-question graphs only.** Not a corpus-level GraphRAG result.
2. **Validation was inspected before freezing** (D12): the connectivity ceilings
   were computed on both splits while diagnosing fragmentation. Nothing was fitted
   to them, but the reported validation ceilings are not strictly naive.
3. **8 of 33 measured stages are invalid** under §7.3 on swap
   growth, including both LLM-Trace stages and five reader stages. EM/F1 and every
   retrieval metric are unaffected, because the reader and the traversal decode
   greedily at temperature 0, so identical context gives identical output
   regardless of timing or paging. Only latency, token-rate and RSS figures from
   those stages are void.
4. **Process RSS under-reports Ollama on Apple Silicon.** Model weights sit in
   GPU-wired memory managed by the Metal driver and are not fully attributed to the
   `llama-server` process. C4's figure is a floor; the swap evidence shows the
   machine was over-committed even where RSS looked comfortable.
5. **The M7c swap gate checks only at script start** (D25), so stages after the
   first began from a ~1.6 GB baseline.
6. **Multiple comparisons, uncorrected.** **80 paired bootstraps were run on
   the validation split** (24 retrieval, 12 follow-up, 32 reader EM/F1, 12 for the
   3B check), plus 62 more on train under cross-validation, for **142 across the
   project**. None is corrected for multiplicity. At 95% confidence roughly
   4 false positives are expected among the validation set alone, so
   an interval that only just excludes zero is not a finding. Two intervals have a
   bound of exactly 0.000. (An earlier note cited "44 comparisons"; that was the
   final M8 analysis only — the reader table plus the 3B check — not the total.)
7. **Small samples.** 150 questions per dataset per split; the 3B check is n=30 per
   dataset, where AllGold@4 takes 31 values and the CIs are wide by construction.
8. **A 1.5B reader compresses EM/F1 differences**, as anticipated. Retrieval
   metrics are the primary evidence.
9. **The 1.5B tracer is a weak LLM-Trace baseline**, though the 3B check found it
   is not materially better on MuSiQue, which weakens that excuse for C3's MuSiQue
   result.
10. **Weak provenance labels.** An edge is positive iff its source paragraph is
    gold, which is a proxy for relevance, not relevance itself.

## 9. Conclusion

The router learns what it was asked to learn. It scores edges better than a
non-learned graph baseline and better than a no-graph edge baseline, with CIs
excluding zero on both datasets, and removing message passing costs it 7–10 points
of edge AP. What it does not do is convert that into better paragraph selection
than plain vector retrieval, and the reason is measured rather than conjectured:
the ego-graph reaches too few of the candidate paragraphs, so the headline metric
is capped below where the baseline already sits. The connectivity mechanisms that
raised the cap did so largely by adding links between distractors, and the
ablations show the one carrying the most edges contributes nothing.

Two things follow for anyone extending this. First, in the per-question
distractor-pool setting the binding constraint is entity linking across
paragraphs, not router capacity — the learning curve is flat and the hyperparameter
grid is noise, while the reachability ceiling is the live variable. Second, the
cost argument stands independently of the quality argument: routing at 36–61× less
latency with zero query-time LLM tokens is a real property, and the fusion result
suggests the graph signal is most useful as a cheap additional input to a vector
ranker rather than as a replacement for one.

---

<!-- M11-VALIDATION-EXTENSION:START -->

## 10. Validation extension (M11)

Pre-registered as **D27** and committed before a single new validation question was read. The original 150-question results in Sections 2–9 stand unrevised; this section is an independent replication on questions no part of the project had seen, plus a secondary pooled analysis.

The **primary, confirmatory** column throughout is the **new 350 alone**. The pooled 500 is secondary and is reported because it tightens the intervals, not because it settles a disagreement — where the two columns disagree, both are shown and the disagreement is stated.

### 10.1 Sampling

*Source: `results/ext_sample_musique.json`, `results/ext_sample_hotpotqa.json`*

| | eligible | available after exclusion | drawn | strata | sha256 of qid list |
|---|---:|---:|---:|---|---|
| musique | 2,417 | 2,267 | 350 | 2-hop 117, 3-hop 117, 4-hop 116 | `4fa6b8a7db237ee2…` |
| hotpotqa | 7,345 | 7,195 | 350 | bridge 279, comparison 71 | `aba802ed38f67334…` |

Seed **20261026**, fixed in D27. Zero overlap with the original 150 and zero with the training questions, asserted at sampling time. Pool construction is the frozen, gold-blind one (MuSiQue top-14 of 20 by bge cosine, HotpotQA all 10), and every pool came out exactly full.

Of MuSiQue's 355 available 4-hop validation questions this draw takes 116, about a third of the stock, so a further extension has little 4-hop headroom left.

### 10.2 Extraction on the new pools

*Source: `results/ext_extraction.json`*

| | new paragraphs | t_p mean | JSON failures | gold coverage | pool coverage |
|---|---:|---:|---:|---:|---:|
| both datasets | 6,272 | 4.020 s | 0 (0.0000%) | — | — |
| musique | 4,900 slots | — | — | 1.0000 (941/941) | 1.0000 |
| hotpotqa | 3,500 slots | — | — | 1.0000 (700/700) | 1.0000 |

Neither stop condition fired. Gold-paragraph coverage is **1.0000** on both datasets against 0.999 / 1.000 originally, and the JSON failure rate is **0.0000%** against 0.02%, so the new questions are not harder to extract from. Per-paragraph behaviour matches the original run closely (salvage 4.88% vs 5.35%, truncation 4.88% vs 5.39%, 423.6 vs 426.4 LLM tokens, 5.52 vs 5.54 triples).

The run took **7.21 h** against the 3.08 h the step-2 timing check projected, because t_p came out at 4.020 s rather than 1.767 s: the machine held ~1.2–1.9 GB of swap throughout. Still inside D27's 12 h gate. **12 of 13 chunks are §7.3 timing-invalid**; the triples are unaffected, since decoding is greedy and each paragraph is extracted independently. The accounting for those chunks was lost to a retry-loop bug and rebuilt from the cache — see **D28**.

### 10.3 Three measurement faults found and corrected

The extension's first scoring sweep was wrong twice over, in ways that produced entirely plausible numbers; a third fault hit the extraction accounting. All three are recorded because published figures changed, and because the second was only found by going looking for it after the first.

**D29 — the router ran query-blind on HotpotQA.** `to_pyg` zero-filled any embedding-cache miss. For the *question* that is not a degraded feature but a silent amputation: `q = 0` zeroes `e_v * q`, `cos(e_v, q)` and `cos(r_e, q)`, so every query-conditioned feature vanishes. The M3 embedding pass had never been run for the new pools, and `build_graph` does not cache the question, so HotpotQA's 350 questions had no embedding at all. MuSiQue escaped only because its top-14-of-20 pool selection embeds the question anyway. Re-measured: HotpotQA GNN AllGold@4 0.571 → 0.583, stacked 0.777 → 0.786, with MuSiQue and every non-graph method byte-identical.

**D30 — 10% of edges were scored with no relation feature.** D29's fix made only the *question* lookup strict. An audit of the remaining kinds then measured relation cache misses of **10.2%** (MuSiQue) and **9.1%** (HotpotQA) on the new pools, against **0.0%** on the published splits: a missed relation leaves that edge's whole `edge_attr` signal-free. The trigger was a latency anomaly, not a metric one — the new questions' graphs are *smaller* (median 270 vs 380 edges on MuSiQue) yet assembly cost 2.4× more per edge, because uncached entity names were being encoded by bge-small inside the timed routing region. `_emb` now raises on a miss for every kind, and the embedding pass caches relation and entity texts. Re-measured, the effect on AllGold@4 is small but runs in both directions — MuSiQue GNN 0.217 → 0.211, GNN+Vector 0.271 → 0.274; HotpotQA GNN+Vector 0.717 → 0.723, stacked 0.786 → 0.780 — which is what a 10% feature corruption looks like when it is noise rather than bias. All GNN-family metrics and all routing timings in this section are from after the fix; the superseded runs are kept at `results/ext_cold_snapshot/` and `results/ext_pre_d30_snapshot/`.

**D28 — the extraction chunk accounting was lost to a retry-loop bug** and rebuilt from the triple cache; the triples themselves were never in doubt.

The common thread is that a zero vector is not a neutral default. Three times now (paragraph nodes, then the question, then relations) a cache miss has been silently absorbed into a feature and the model has gone on returning plausible scores. Nothing in the pipeline announced any of them.

### 10.4 The two question sets are not equally hard

*Source: `results/ext_pooled_report.json`*

AllGold@4 is bounded above by whether the ego-graph reaches every gold paragraph, so if that bound differs between the two sets, absolute AllGold@4 cannot be compared across them. It does differ, and the interval is unpaired because the two sets are disjoint samples.

| dataset | quantity | original 150 | new 350 | difference (95% CI) |
|---|---|---:|---:|---|
| musique | graph ceiling (AllGold@4) | 0.380 | 0.257 | -0.1229 [-0.2133, -0.0343] |
| musique | graph pool coverage | 0.791 | 0.674 | -0.1171 [-0.1639, -0.0704] |
| hotpotqa | graph ceiling (AllGold@4) | 0.993 | 0.980 | -0.0133 [-0.0333, +0.0067] |
| hotpotqa | graph pool coverage | 0.994 | 0.990 | -0.0040 [-0.0119, +0.0028] |

**MuSiQue's new draw is materially harder for every graph method**: the ceiling falls 0.380 → 0.257 and graph pool coverage 0.791 → 0.674, both intervals excluding zero. This is not a stratification artefact — both sets are uniform thirds by hop count, pool recall is unchanged at 0.897 (so the gold paragraphs are present), and the ceiling drops inside *every* hop bucket (2-hop 0.720 → 0.590, 3-hop 0.200 → 0.128, 4-hop 0.220 → 0.052). HotpotQA is unchanged, as expected when the ceiling is already 0.98+.

Two consequences, which govern how the rest of this section may be read. First, **only the within-set paired contrasts replicate across the two sets**; an absolute AllGold@4 from one set must not be compared to the other's. Second, the coverage bottleneck that Sections 5–8 identify as this project's central limitation is **worse on fresh questions than the original 150 suggested**, so that finding strengthens rather than softens on replication.

### 10.5 Retrieval, original 150 vs new 350

**musique** — AllGold@4 unconditional, with the graph ceiling.

*Source: `results/m7a_report.json`, `results/ext_m7a_report.json`*

| method | SP-Recall@4 (150) | SP-Recall@4 (350) | AllGold@4 (150) | AllGold@4 (350) | Δ |
|---|---:|---:|---:|---:|---:|
| Vector RAG | 0.6333 | 0.6483 | 0.267 | 0.303 | +0.036 |
| PPR | 0.5506 | 0.5279 | 0.240 | 0.220 | -0.020 |
| Edge-Sim | 0.5517 | 0.5462 | 0.213 | 0.223 | +0.010 |
| GNN router (graph-only) | 0.5500 | 0.5224 | 0.240 | 0.211 | -0.029 |
| GNN+Vector (RRF k=60) | 0.6239 | 0.6181 | 0.273 | 0.274 | +0.001 |
| PPR+Vector (RRF k=60) | 0.6167 | 0.6183 | 0.267 | 0.269 | +0.002 |
| Stacked combiner (D19) | 0.6533 | 0.6517 | 0.293 | 0.303 | +0.010 |
| *graph ceiling* | — | — | 0.380 | 0.257 | -0.123 |

The Δ column is **descriptive only and carries no interval**: by §10.4 the two sets are not equally hard, so a cross-set Δ mixes a method difference with a sample difference and cannot be read as a contrast. On MuSiQue the ceiling itself moved by −0.123, which is larger than every method Δ in the table, so the per-method numbers here say more about the draw than about the methods.

The replicating quantities are the **within-set paired contrasts** in §10.7, each computed inside one question set.

**hotpotqa** — AllGold@4 unconditional, with the graph ceiling.

*Source: `results/m7a_report.json`, `results/ext_m7a_report.json`*

| method | SP-Recall@4 (150) | SP-Recall@4 (350) | AllGold@4 (150) | AllGold@4 (350) | Δ |
|---|---:|---:|---:|---:|---:|
| Vector RAG | 0.8933 | 0.8829 | 0.793 | 0.774 | -0.019 |
| PPR | 0.7067 | 0.6957 | 0.533 | 0.489 | -0.045 |
| Edge-Sim | 0.8033 | 0.7743 | 0.613 | 0.586 | -0.028 |
| GNN router (graph-only) | 0.7500 | 0.7557 | 0.567 | 0.583 | +0.016 |
| GNN+Vector (RRF k=60) | 0.8433 | 0.8471 | 0.713 | 0.723 | +0.010 |
| PPR+Vector (RRF k=60) | 0.8367 | 0.8357 | 0.693 | 0.697 | +0.004 |
| Stacked combiner (D19) | 0.8900 | 0.8829 | 0.793 | 0.780 | -0.013 |
| *graph ceiling* | — | — | 0.993 | 0.980 | -0.013 |

The Δ column is **descriptive only and carries no interval**: by §10.4 the two sets are not equally hard, so a cross-set Δ mixes a method difference with a sample difference and cannot be read as a contrast.

The replicating quantities are the **within-set paired contrasts** in §10.7, each computed inside one question set.

### 10.6 Reader answer quality

*Source: `results/ext_pooled_report.json`*

EM/F1 from the frozen 1.5B reader over each method's top-4 paragraphs. Paired bootstrap on the new 350, F1.

**musique**

| method | EM (150) | F1 (150) | EM (350) | F1 (350) | F1 vs Vector RAG (350) |
|---|---:|---:|---:|---:|---|
| Vector RAG | 0.1800 | 0.2694 | 0.1571 | 0.2359 | — |
| PPR | 0.1133 | 0.1869 | 0.1143 | 0.1813 | — |
| LLM-Trace | 0.1533 | 0.2322 | 0.1629 | 0.2319 | — |
| GNN router (graph-only) | 0.1133 | 0.1914 | 0.1057 | 0.1671 | -0.0688 [-0.1074, -0.0309] |
| GNN+Vector (RRF k=60) | 0.1733 | 0.2525 | 0.1457 | 0.2198 | -0.0161 [-0.0466, +0.0135] |
| Stacked combiner (D19) | 0.1533 | 0.2498 | 0.1571 | 0.2358 | -0.0001 [-0.0258, +0.0257] |
| Oracle (gold only) | 0.3667 | 0.4636 | 0.3657 | 0.4415 | — |

**hotpotqa**

| method | EM (150) | F1 (150) | EM (350) | F1 (350) | F1 vs Vector RAG (350) |
|---|---:|---:|---:|---:|---|
| Vector RAG | 0.4733 | 0.5822 | 0.3743 | 0.5111 | — |
| PPR | 0.3867 | 0.5063 | 0.3486 | 0.4736 | — |
| LLM-Trace | 0.4267 | 0.5344 | 0.2943 | 0.3988 | — |
| GNN router (graph-only) | 0.4200 | 0.5157 | 0.3343 | 0.4566 | -0.0545 [-0.1011, -0.0081] |
| GNN+Vector (RRF k=60) | 0.4800 | 0.5769 | 0.3743 | 0.4995 | -0.0117 [-0.0446, +0.0212] |
| Stacked combiner (D19) | 0.4600 | 0.5677 | 0.3800 | 0.5110 | -0.0001 [-0.0245, +0.0244] |
| Oracle (gold only) | 0.5600 | 0.6974 | 0.5200 | 0.6661 | — |

Both datasets reproduce the original ordering: no graph method's F1 beats Vector RAG, the GNN+Vector and stacked hybrids are indistinguishable from it, and the oracle gap is large on both (MuSiQue 0.442 F1 vs 0.236 for Vector RAG), so retrieval, not the reader, is the binding constraint. On HotpotQA the GNN router does beat LLM-Trace on F1 (+0.0579 [+0.0139, +0.1019]), matching C3's quality half there.

### 10.7 Verdicts on three question sets

*Source: `results/ext_pooled_report.json`*

The primary column is **new 350**. A row marked *disagreement* has a verdict that is not the same across all three.

**C1a** — GNN - Vector, AllGold@4 (D4-conditional)

| dataset | set | n | difference (95% CI) | verdict |
|---|---|---:|---|---|
| musique | original 150 | 111 | +0.0000 [-0.0901, +0.0901] | INCONCLUSIVE |
| musique | **new 350** | 257 | -0.1012 [-0.1595, -0.0428] | NOT SUPPORTED |
| musique | pooled 500 | 368 | -0.0707 [-0.1196, -0.0217] | NOT SUPPORTED |
| musique | *disagreement* | | | {'orig150': 'INCONCLUSIVE', 'new350': 'NOT SUPPORTED', 'pooled500': 'NOT SUPPORTED'} |
| hotpotqa | original 150 | 150 | -0.2267 [-0.3200, -0.1267] | NOT SUPPORTED |
| hotpotqa | **new 350** | 350 | -0.1914 [-0.2457, -0.1343] | NOT SUPPORTED |
| hotpotqa | pooled 500 | 500 | -0.2020 [-0.2500, -0.1540] | NOT SUPPORTED |

**C1a_uncond** — GNN - Vector, AllGold@4 (unconditional)

| dataset | set | n | difference (95% CI) | verdict |
|---|---|---:|---|---|
| musique | original 150 | 150 | -0.0267 [-0.1000, +0.0467] | INCONCLUSIVE |
| musique | **new 350** | 350 | -0.0914 [-0.1400, -0.0429] | NOT SUPPORTED |
| musique | pooled 500 | 500 | -0.0720 [-0.1140, -0.0320] | NOT SUPPORTED |
| musique | *disagreement* | | | {'orig150': 'INCONCLUSIVE', 'new350': 'NOT SUPPORTED', 'pooled500': 'NOT SUPPORTED'} |
| hotpotqa | original 150 | 150 | -0.2267 [-0.3200, -0.1267] | NOT SUPPORTED |
| hotpotqa | **new 350** | 350 | -0.1914 [-0.2457, -0.1343] | NOT SUPPORTED |
| hotpotqa | pooled 500 | 500 | -0.2020 [-0.2500, -0.1540] | NOT SUPPORTED |

**C1b** — GNN+Vec - PPR+Vec, AllGold@4 (D15 matched, conditional)

| dataset | set | n | difference (95% CI) | verdict |
|---|---|---:|---|---|
| musique | original 150 | 111 | +0.0090 [-0.0180, +0.0450] | INCONCLUSIVE |
| musique | **new 350** | 257 | +0.0039 [-0.0233, +0.0311] | INCONCLUSIVE |
| musique | pooled 500 | 368 | +0.0054 [-0.0163, +0.0272] | INCONCLUSIVE |
| hotpotqa | original 150 | 150 | +0.0200 [-0.0400, +0.0800] | INCONCLUSIVE |
| hotpotqa | **new 350** | 350 | +0.0257 [-0.0086, +0.0600] | INCONCLUSIVE |
| hotpotqa | pooled 500 | 500 | +0.0240 [-0.0060, +0.0540] | INCONCLUSIVE |

**C2** — GNN - PPR, AllGold@4 (unconditional)

| dataset | set | n | difference (95% CI) | verdict |
|---|---|---:|---|---|
| musique | original 150 | 150 | +0.0000 [-0.0467, +0.0467] | INCONCLUSIVE |
| musique | **new 350** | 350 | -0.0086 [-0.0343, +0.0143] | INCONCLUSIVE |
| musique | pooled 500 | 500 | -0.0060 [-0.0280, +0.0160] | INCONCLUSIVE |
| hotpotqa | original 150 | 150 | +0.0333 [-0.0400, +0.1067] | INCONCLUSIVE |
| hotpotqa | **new 350** | 350 | +0.0943 [+0.0514, +0.1400] | SUPPORTED |
| hotpotqa | pooled 500 | 500 | +0.0760 [+0.0380, +0.1140] | SUPPORTED |
| hotpotqa | *disagreement* | | | {'orig150': 'INCONCLUSIVE', 'new350': 'SUPPORTED', 'pooled500': 'SUPPORTED'} |

**C2_cond** — GNN - PPR, AllGold@4 (conditional)

| dataset | set | n | difference (95% CI) | verdict |
|---|---|---:|---|---|
| musique | original 150 | 111 | +0.0180 [-0.0360, +0.0721] | INCONCLUSIVE |
| musique | **new 350** | 257 | -0.0195 [-0.0506, +0.0117] | INCONCLUSIVE |
| musique | pooled 500 | 368 | -0.0082 [-0.0353, +0.0190] | INCONCLUSIVE |
| hotpotqa | original 150 | 150 | +0.0333 [-0.0400, +0.1067] | INCONCLUSIVE |
| hotpotqa | **new 350** | 350 | +0.0943 [+0.0514, +0.1400] | SUPPORTED |
| hotpotqa | pooled 500 | 500 | +0.0760 [+0.0380, +0.1140] | SUPPORTED |
| hotpotqa | *disagreement* | | | {'orig150': 'INCONCLUSIVE', 'new350': 'SUPPORTED', 'pooled500': 'SUPPORTED'} |

**C3q** — GNN - LLM-Trace, AllGold@4 vs -3pt margin

| dataset | set | n | difference (95% CI) | verdict |
|---|---|---:|---|---|
| musique | original 150 | 150 | -0.0600 [-0.1333, +0.0067] | INCONCLUSIVE |
| musique | **new 350** | 350 | -0.1000 [-0.1457, -0.0571] | NOT SUPPORTED |
| musique | pooled 500 | 500 | -0.0880 [-0.1260, -0.0520] | NOT SUPPORTED |
| musique | *disagreement* | | | {'orig150': 'INCONCLUSIVE', 'new350': 'NOT SUPPORTED', 'pooled500': 'NOT SUPPORTED'} |
| hotpotqa | original 150 | 150 | +0.2000 [+0.1067, +0.2933] | SUPPORTED |
| hotpotqa | **new 350** | 350 | +0.2686 [+0.2114, +0.3257] | SUPPORTED |
| hotpotqa | pooled 500 | 500 | +0.2480 [+0.1980, +0.2980] | SUPPORTED |

**C3, cost half** — LLM-Trace median latency over GNN routing median.

| dataset | set | GNN routing | LLM-Trace | ratio | verdict |
|---|---|---:|---:|---:|---|
| musique | original 150 | 43.8 ms | 1573 ms | 35.9× | SUPPORTED ⚠ |
| musique | **new 350** | 26.5 ms | 3729 ms | 140.7× | SUPPORTED ⚠ |
| hotpotqa | original 150 | 35.6 ms | 1764 ms | 49.6× | SUPPORTED ⚠ |
| hotpotqa | **new 350** | 19.9 ms | 3578 ms | 179.4× | SUPPORTED ⚠ |

⚠ = at least one of the two stages is §7.3-invalid on swap, so the ratio is indicative.

**The GNN side of this ratio is not comparable to Section 7's, and Section 7's is the inflated one.** Counting bge-small calls per graph showed the published validation split was doing 8.2 (MuSiQue) and 6.5 (HotpotQA) encodes *inside* the timed region, because the M3/M5 passes never persisted every text `build_graph` produces; the extension, after D30, does zero. With the graph texts of both splits cached, GNN routing medians are **29.0 / 20.9 ms** on the published 150 and **26.5 / 19.9 ms** on the new 350 — the two question sets agree, and the extension is not faster than the original. Section 7's 43.8 / 35.6 ms carry the artefact; they are left as published and corrected here rather than restated there. The published 150's metrics reproduced exactly across all three timings, which is what makes this a measurement fix and not a rerun (D30, same fault class as D23's Edge-Sim 583.5 → 10.6 ms). The ratio's numerator is §7.3-invalid on swap in every run (D24), so read it as an order of magnitude: the margin is ~36–180× against a threshold of 10.

**C4** — peak process-tree RSS over the extension's stages: **1575 MB** (1.54 GB) → **SUPPORTED** against the 6 GB budget. The same caveat as Section 7 applies: RSS under-reports Ollama's GPU-wired memory.

### 10.8 Multiplicity and stage validity

The extension computes **78** new confidence intervals: 26 in the claim table (new 350 and pooled 500), 24 in the aggregation's own method contrasts, 24 on reader F1/EM and 4 on set comparability. A further 14 recompute already-published original-150 comparisons through this code path and are not new evidence.

They remain **uncorrected**, as everywhere else in this report. At this count, roughly 4 intervals would be expected to exclude zero by chance alone, so **no single borderline interval here is evidence of anything**.

The verdict changes in §10.7 are not borderline. They are **3** distinct comparisons (5 rows below, since two appear in both their conditional and unconditional readings). Each flipped interval and its distance from the decision threshold:

| claim | dataset | original 150 | new 350 | new-350 interval vs threshold |
|---|---|---|---|---|
| C1a | musique | INCONCLUSIVE | **NOT SUPPORTED** | -0.1012 [-0.1595, -0.0428], nearer bound -0.0428 vs 0 (0.0428 clear) |
| C1a_uncond | musique | INCONCLUSIVE | **NOT SUPPORTED** | -0.0914 [-0.1400, -0.0429], nearer bound -0.0429 vs 0 (0.0429 clear) |
| C2 | hotpotqa | INCONCLUSIVE | **SUPPORTED** | +0.0943 [+0.0514, +0.1400], nearer bound +0.0514 vs 0 (0.0514 clear) |
| C2_cond | hotpotqa | INCONCLUSIVE | **SUPPORTED** | +0.0943 [+0.0514, +0.1400], nearer bound +0.0514 vs 0 (0.0514 clear) |
| C3q | musique | INCONCLUSIVE | **NOT SUPPORTED** | -0.1000 [-0.1457, -0.0571], nearer bound -0.0571 vs −0.03 (0.0271 clear) |

Every one is a case where the original 150 could not resolve the comparison and 350 questions could — the direction of the point estimate did not reverse in any of them.

**Stage validity under §7.3.** The extension's own measured stages:

| stage group | measured | §7.3-invalid on swap |
|---|---:|---:|
| retrieval routing (§10.5, §10.7) | 18 | **0** |
| extraction chunks (§10.2) | 13 | 12 |
| LLM-Trace routing | 2 | 2 |
| reader (§10.6) | 14 | 8 |

Retrieval is fully valid this time, which it was not before D30 — removing bge-small from the timed region cut the sweep from 3,462 s to 768 s and with it the memory pressure. The 22 invalid stages are all Ollama-loaded or extraction stages, the same pattern as D26, and they void **only** latency, token-rate and RSS figures: every EM/F1 and retrieval metric is unaffected, because the reader and the traversal decode greedily at `temperature=0`, so identical context gives identical output however the machine was paging.

<!-- M11-VALIDATION-EXTENSION:END -->

# Appendix

## A. Deviation log

*Source: `agent.md` §14 (full reasons and measurements are there; abridged here)*

| ID | §  | when, relative to validation | decision (abridged) |
| :-- | :-- | :-- | :-- |
| **D1** | 3.1, 8.2 | before any validation access | `POOL_SIZE=10` for both datasets → **per-dataset**: HotpotQA **10**, MuSiQue **14**. |
| **D2** | 6.2 | before any validation access | Extraction prompt **v1 → v2**: `format` is a JSON *schema* with `maxItems=6` (grammar-enforced) rather than the string `"json"`; paragraphs truncated to **120** words rather than 200; responses cut off at… |
| **D3** | 8.2 step 3 | before any validation access | Gate on **estimated total extraction hours** (≤ 5 h) instead of `t_p ≤ 6 s`. `t_p` is still reported. |
| **D4** | 1.1, 7.1 | before any validation access | **AllGold@4** reported both **unconditional** and **conditional** on all gold paragraphs surviving into the pool. Claim **C1 is tested on the conditional** version. |
| **D5** | 3 | before any validation access | HotpotQA **train restricted to `level=hard`**. |
| **D6** | 3.1 | before any validation access | Questions with **fewer paragraphs than their dataset's pool size** are excluded. |
| **D7** | 10 | before any validation access | `Example` gains a trailing `meta: dict`; `load_examples` gains a trailing optional `cfg=None`. |
| **D8** | 8.1 | before any validation access | `prompt_version` encodes the triple cap, e.g. `v2-mt6`. |
| **D15** | 1.1, 5 | after the ceiling look, before any method was scored | **Primary C1 reporting changed** from a single graph-only row to **(a)** graph-only, always shown next to its graph ceiling, and **(b)** a matched hybrid pair, **GNN+Vector vs PPR+Vector**, fused by reciprocal-rank… |
| **D26** | 7.3, 11 | after M7c/M8 | **8 of 33 measured stages are invalid under §7.3 and are reported as such**, not quietly dropped: both M7c LLM-Trace stages (+3,389 and +2,195 MB), five M8 reader stages (MuSiQue ppr +1,587, oracle +926, gnn +830,… |
| **D25** | 7.3 | after M7c/M8 | **The M7c swap gate checks only at script start, so stages after the first ran from a degraded baseline.** The run began at **0 MB** swap and passed the 300 MB gate, but the second stage onwards started at **~1.6… |
| **D24** | 7.3 | after M7c/M8 | **LLM-Trace efficiency is reported as INDICATIVE, never valid.** All four measurements are tabulated (M7 first run, D21 redo, 10-question smoke, M7c clean-start): medians 1,543–1,639 ms (MuSiQue) and 1,684–2,157 ms… |
| **D23** | 1.1, 7.3 | after M7 scoring | **Two M7 verdicts corrected after computing the CIs the claims actually call for.** (a) **C3's quality half on MuSiQue changes from NOT SUPPORTED to INCONCLUSIVE**: the test is "AllGold@4 not lower than LLM-Trace by… |
| **D22** | 5 | after M7 scoring | **Correction to how the D19 stacked hybrid's train/test mismatch is described.** An earlier write-up said the mismatch "can only flatter the stacked row". That was wrong and is withdrawn. The final GNN models never… |
| **D21** | 7.3, 11 | during M7 (validation scoring) | **The first M7b LLM-Trace run was invalidated and redone.** Its efficiency stages recorded swap growth of **+3,169 MB (MuSiQue)** and **+4,020 MB (HotpotQA)** against §7.3's 500 MB rule, so the latency and RSS… |
| **D20-SKIPPED** | 4.3, 5 | after the ceiling look, before any method was scored | **D20 was gated on D19 and the gate closed, so the paragraph-aware GNN was never implemented.** D19's stacked hybrid vs Vector RAG on AllGold@4: MuSiQue **−0.013 [−0.053, +0.020]**, HotpotQA **+0.027 [−0.020,… |
| **D20** | 4.3, 5 | after the ceiling look, before any method was scored | **Paragraph-aware GNN**, an exploratory variant: a paragraph node gains a cosine-to-question feature computed from the cached paragraph-text embedding (not just its title), plus a paragraph-level term in the loss.… |
| **D19-RESULT** | 5 | after the ceiling look, before any method was scored | **D19 was implemented and evaluated on train CV; it does not beat Vector RAG and will still be reported on validation.** Stacked vs Vector RAG (AllGold@4): MuSiQue −0.013, HotpotQA +0.027, both CIs including zero.… |
| **D19** | 5 | after the ceiling look, before any method was scored | **Stacked hybrid**, an exploratory variant: a logistic regression over paragraph rows with three features — bge cosine to the question, the GNN's OOF paragraph score, and the PPR paragraph score — fit under a… |
| **D18** | 1.1, 4.3, 5 | after the ceiling look, before any method was scored | **Findings from the D17 cross-validation that change what later phases should claim, recorded now so they are not presented as discoveries after seeing validation.** (a) The 8-config hyperparameter grid found **no… |
| **D17** | 3, 6, 11 | after the ceiling look, before any method was scored | **The 10% dev split is replaced by 5-fold cross-validation over the training questions** for all hyperparameter selection and all ablations. Folds are fixed by the global seed, **stratified by hop count (MuSiQue) and… |
| **D16** | 4.1.1, 13 | after the ceiling look, before any method was scored | **MuSiQue 3-hop validation connectivity is materially worse than 3-hop train, and no configuration change was made in response.** Validation 3-hop: **5.3** components per question against ~2.1 on train, pool coverage… |
| **D14** | 4.1 | after the ceiling look, before any method was scored | Ego-graph caps raised from **200 nodes / 400 edges to 300 nodes / 800 edges**. |
| **D13** | 4.1.1 | after the ceiling look, before any method was scored | The §4.1.1 connectivity target was **MISSED on MuSiQue and is accepted as missed, not passed**: the target is ≥90% of pool paragraphs carrying an ego-graph edge on train, and the frozen configuration reaches… |
| **D12** | 3, 4.1, 12 | **records the validation look** (connectivity ceilings, before freezing) | **Connectivity settings are chosen on TRAIN and DEV statistics only, frozen, and only then measured once on validation.** The disconnected-component problem was itself discovered from *validation*-split ceilings… |
| **D11** | 6.3, 7.3 | before any validation access | The M4 extraction timings — **t_p ≈ 1.90 s, 6,489 paragraphs, 3.30 h wall** — are a **planning figure only** and **must not be used as the indexing cost in the §7.3 efficiency tables or the §1.1 C3 comparison**. The… |
| **D10** | 7.3 | before any validation access | System-wide swap growth is **no longer an abort condition for extraction (M4)**. Extraction aborts instead on (a) own process-tree RSS, excluding Ollama, staying ≥1.5× baseline for 10 consecutive ticks, (b) rolling… |
| **D9** | 2 | before any validation access | The `OLLAMA_NUM_PARALLEL=2` option is probed by issuing 2 concurrent requests, not by restarting the Ollama server with a different environment. |

## B. Stage validity

*Source: `results/m8_analysis.json`*

| stage | swap at start | swap growth | peak tree RSS | §7.3 verdict |
| :-- | --: | --: | --: | :-- |
| M7 indexing (extraction) | 264 MB | +0 MB | 1882 MB | valid |
| M7c llm_trace/musique | 0 MB | +3389 MB | 2005 MB | **INVALID** |
| M7c routing/musique | 1590 MB | +0 MB | 894 MB | valid |
| M7c llm_trace/hotpotqa | 1582 MB | +2195 MB | 3482 MB | **INVALID** |
| M7c routing/hotpotqa | 1624 MB | +0 MB | 885 MB | valid |
| M8 reader vector/musique | — | -122 MB | 2608 MB | valid |
| M8 reader ppr/musique | — | +1587 MB | 1680 MB | **INVALID** |
| M8 reader llm_trace/musique | — | +568 MB | 1940 MB | **INVALID** |
| M8 reader gnn/musique | — | +830 MB | 2016 MB | **INVALID** |
| M8 reader gnn_rrf/musique | — | +507 MB | 2152 MB | **INVALID** |
| M8 reader stacked/musique | — | +156 MB | 2300 MB | valid |
| M8 reader oracle/musique | — | +926 MB | 2072 MB | **INVALID** |
| M8 reader vector/hotpotqa | — | +841 MB | 1904 MB | **INVALID** |
| M8 reader ppr/hotpotqa | — | -212 MB | 1933 MB | valid |
| M8 reader llm_trace/hotpotqa | — | -103 MB | 2178 MB | valid |
| M8 reader gnn/hotpotqa | — | -126 MB | 2268 MB | valid |
| M8 reader gnn_rrf/hotpotqa | — | -8 MB | 2340 MB | valid |
| M8 reader stacked/hotpotqa | — | -131 MB | 2493 MB | valid |
| M8 reader oracle/hotpotqa | — | +355 MB | 2146 MB | valid |
| M7a routing vector/musique | — | +0 MB | 300 MB | valid |
| M7a routing ppr/musique | — | +0 MB | 801 MB | valid |
| M7a routing edge_sim/musique | — | +0 MB | 1196 MB | valid |
| M7a routing gnn/musique | — | +0 MB | 852 MB | valid |
| M7a routing gnn_rrf/musique | — | +0 MB | 872 MB | valid |
| M7a routing ppr_rrf/musique | — | +0 MB | 822 MB | valid |
| M7a routing stacked/musique | — | +0 MB | 882 MB | valid |
| M7a routing vector/hotpotqa | — | +0 MB | 293 MB | valid |
| M7a routing ppr/hotpotqa | — | +0 MB | 791 MB | valid |
| M7a routing edge_sim/hotpotqa | — | +0 MB | 1477 MB | valid |
| M7a routing gnn/hotpotqa | — | +0 MB | 850 MB | valid |
| M7a routing gnn_rrf/hotpotqa | — | +0 MB | 865 MB | valid |
| M7a routing ppr_rrf/hotpotqa | — | +0 MB | 810 MB | valid |
| M7a routing stacked/hotpotqa | — | +0 MB | 871 MB | valid |
| **8 of 33 stages invalid** | | | | |

## C. All paired comparisons

*Source: `results/m7a_report.json`, `results/m7_recheck.json`, `results/m8_emf1.json`, `results/m6_ablations.json`, `results/m6_compare.json`, `results/m6_stack.json`*

| # | comparison | metric | dataset | CI | registered? |
| --: | :-- | :-- | :-- | :-- | :-- |
| 1 | gnn − vector | allgold4 | musique | -0.0267 [-0.1000, +0.0467] | pre-registered |
| 2 | gnn − vector | sp_recall4 | musique | -0.0833 [-0.1367, -0.0311] | pre-registered |
| 3 | gnn − ppr | allgold4 | musique | +0.0000 [-0.0467, +0.0467] | pre-registered |
| 4 | gnn − ppr | sp_recall4 | musique | -0.0006 [-0.0328, +0.0300] | pre-registered |
| 5 | gnn − edge_sim | allgold4 | musique | +0.0267 [-0.0333, +0.0867] | **post-hoc** |
| 6 | gnn − edge_sim | sp_recall4 | musique | -0.0017 [-0.0428, +0.0389] | **post-hoc** |
| 7 | gnn_rrf − ppr_rrf | allgold4 | musique | +0.0067 [-0.0200, +0.0333] | pre-registered |
| 8 | gnn_rrf − ppr_rrf | sp_recall4 | musique | +0.0072 [-0.0156, +0.0300] | pre-registered |
| 9 | stacked − vector | allgold4 | musique | +0.0267 [-0.0133, +0.0667] | **post-hoc** |
| 10 | stacked − vector | sp_recall4 | musique | +0.0200 [-0.0028, +0.0433] | **post-hoc** |
| 11 | stacked − gnn_rrf | allgold4 | musique | +0.0200 [-0.0133, +0.0533] | **post-hoc** |
| 12 | stacked − gnn_rrf | sp_recall4 | musique | +0.0294 [+0.0061, +0.0539] | **post-hoc** |
| 13 | gnn − vector | allgold4 | hotpotqa | -0.2267 [-0.3200, -0.1267] | pre-registered |
| 14 | gnn − vector | sp_recall4 | hotpotqa | -0.1433 [-0.1967, -0.0900] | pre-registered |
| 15 | gnn − ppr | allgold4 | hotpotqa | +0.0333 [-0.0400, +0.1067] | pre-registered |
| 16 | gnn − ppr | sp_recall4 | hotpotqa | +0.0433 [-0.0067, +0.0933] | pre-registered |
| 17 | gnn − edge_sim | allgold4 | hotpotqa | -0.0467 [-0.1400, +0.0467] | **post-hoc** |
| 18 | gnn − edge_sim | sp_recall4 | hotpotqa | -0.0533 [-0.1067, +0.0000] *(borderline: a bound is exactly 0.000)* | **post-hoc** |
| 19 | gnn_rrf − ppr_rrf | allgold4 | hotpotqa | +0.0200 [-0.0400, +0.0800] | pre-registered |
| 20 | gnn_rrf − ppr_rrf | sp_recall4 | hotpotqa | +0.0067 [-0.0267, +0.0400] | pre-registered |
| 21 | stacked − vector | allgold4 | hotpotqa | +0.0000 [-0.0400, +0.0400] | **post-hoc** |
| 22 | stacked − vector | sp_recall4 | hotpotqa | -0.0033 [-0.0233, +0.0167] | **post-hoc** |
| 23 | stacked − gnn_rrf | allgold4 | hotpotqa | +0.0800 [+0.0267, +0.1400] | **post-hoc** |
| 24 | stacked − gnn_rrf | sp_recall4 | hotpotqa | +0.0467 [+0.0133, +0.0800] | **post-hoc** |
| 25 | allgold4_uncond | — | musique | -0.0600 [-0.1333, +0.0067] | pre-registered |
| 26 | allgold4_cond | — | musique | -0.0450 [-0.1261, +0.0360] | pre-registered |
| 27 | sp_recall4_uncond | — | musique | -0.0933 [-0.1433, -0.0456] | **post-hoc** |
| 28 | sp_recall4_cond | — | musique | -0.0706 [-0.1231, -0.0173] | **post-hoc** |
| 29 | edgep10_vs_PPR | — | musique | +0.1120 [+0.0593, +0.1680] | **post-hoc** |
| 30 | edgep10_vs_Edge-Sim | — | musique | +0.1287 [+0.0680, +0.1900] | **post-hoc** |
| 31 | allgold4_uncond | — | hotpotqa | +0.2000 [+0.1067, +0.2933] | pre-registered |
| 32 | allgold4_cond | — | hotpotqa | +0.2000 [+0.1067, +0.2933] | pre-registered |
| 33 | sp_recall4_uncond | — | hotpotqa | +0.1300 [+0.0667, +0.1900] | **post-hoc** |
| 34 | sp_recall4_cond | — | hotpotqa | +0.1300 [+0.0667, +0.1900] | **post-hoc** |
| 35 | edgep10_vs_PPR | — | hotpotqa | +0.1993 [+0.1460, +0.2540] | **post-hoc** |
| 36 | edgep10_vs_Edge-Sim | — | hotpotqa | +0.0627 [+0.0100, +0.1133] | **post-hoc** |
| 37 | gnn − vector | EM | musique | -0.0667 [-0.1333, +0.0000] *(borderline: a bound is exactly 0.000)* | pre-registered |
| 38 | gnn − vector | F1 | musique | -0.0780 [-0.1467, -0.0091] | pre-registered |
| 39 | gnn − oracle | EM | musique | -0.2533 [-0.3333, -0.1733] | pre-registered |
| 40 | gnn − oracle | F1 | musique | -0.2722 [-0.3509, -0.1970] | pre-registered |
| 41 | gnn − ppr | EM | musique | +0.0000 [-0.0400, +0.0400] | **post-hoc** |
| 42 | gnn − ppr | F1 | musique | +0.0045 [-0.0404, +0.0488] | **post-hoc** |
| 43 | gnn − llm_trace | EM | musique | -0.0400 [-0.1067, +0.0267] | **post-hoc** |
| 44 | gnn − llm_trace | F1 | musique | -0.0408 [-0.1080, +0.0269] | **post-hoc** |
| 45 | gnn_rrf − gnn | EM | musique | +0.0600 [+0.0067, +0.1133] | **post-hoc** |
| 46 | gnn_rrf − gnn | F1 | musique | +0.0611 [+0.0041, +0.1183] | **post-hoc** |
| 47 | stacked − gnn_rrf | EM | musique | -0.0200 [-0.0733, +0.0333] | **post-hoc** |
| 48 | stacked − gnn_rrf | F1 | musique | -0.0027 [-0.0551, +0.0491] | **post-hoc** |
| 49 | vector − oracle | EM | musique | -0.1867 [-0.2667, -0.1133] | pre-registered |
| 50 | vector − oracle | F1 | musique | -0.1942 [-0.2717, -0.1200] | pre-registered |
| 51 | stacked − vector | EM | musique | -0.0267 [-0.0600, +0.0000] *(borderline: a bound is exactly 0.000)* | pre-registered |
| 52 | stacked − vector | F1 | musique | -0.0196 [-0.0572, +0.0159] | pre-registered |
| 53 | 3B − 1.5B | allgold4 | musique | -0.0333 [-0.1333, +0.0667] | **post-hoc** |
| 54 | GNN − 3B | allgold4 | musique | +0.0000 [-0.1333, +0.1333] | **post-hoc** |
| 55 | GNN − 1.5B | allgold4 | musique | -0.0333 [-0.2000, +0.1333] | **post-hoc** |
| 56 | 3B − 1.5B | sp_recall4 | musique | +0.0000 [-0.0556, +0.0556] | **post-hoc** |
| 57 | GNN − 3B | sp_recall4 | musique | -0.0194 [-0.1222, +0.0750] | **post-hoc** |
| 58 | GNN − 1.5B | sp_recall4 | musique | -0.0194 [-0.1333, +0.0944] | **post-hoc** |
| 59 | gnn − vector | EM | hotpotqa | -0.0533 [-0.1267, +0.0200] | pre-registered |
| 60 | gnn − vector | F1 | hotpotqa | -0.0666 [-0.1377, +0.0066] | pre-registered |
| 61 | gnn − oracle | EM | hotpotqa | -0.1400 [-0.2133, -0.0667] | pre-registered |
| 62 | gnn − oracle | F1 | hotpotqa | -0.1817 [-0.2583, -0.1093] | pre-registered |
| 63 | gnn − ppr | EM | hotpotqa | +0.0333 [-0.0133, +0.0867] | **post-hoc** |
| 64 | gnn − ppr | F1 | hotpotqa | +0.0093 [-0.0430, +0.0623] | **post-hoc** |
| 65 | gnn − llm_trace | EM | hotpotqa | -0.0067 [-0.0800, +0.0667] | **post-hoc** |
| 66 | gnn − llm_trace | F1 | hotpotqa | -0.0187 [-0.0942, +0.0542] | **post-hoc** |
| 67 | gnn_rrf − gnn | EM | hotpotqa | +0.0600 [+0.0067, +0.1133] | **post-hoc** |
| 68 | gnn_rrf − gnn | F1 | hotpotqa | +0.0612 [+0.0032, +0.1207] | **post-hoc** |
| 69 | stacked − gnn_rrf | EM | hotpotqa | -0.0200 [-0.0733, +0.0267] | **post-hoc** |
| 70 | stacked − gnn_rrf | F1 | hotpotqa | -0.0092 [-0.0653, +0.0447] | **post-hoc** |
| 71 | vector − oracle | EM | hotpotqa | -0.0867 [-0.1600, -0.0200] | pre-registered |
| 72 | vector − oracle | F1 | hotpotqa | -0.1152 [-0.1883, -0.0457] | pre-registered |
| 73 | stacked − vector | EM | hotpotqa | -0.0133 [-0.0600, +0.0333] | pre-registered |
| 74 | stacked − vector | F1 | hotpotqa | -0.0146 [-0.0607, +0.0306] | pre-registered |
| 75 | 3B − 1.5B | allgold4 | hotpotqa | +0.1000 [-0.0333, +0.2333] | **post-hoc** |
| 76 | GNN − 3B | allgold4 | hotpotqa | +0.3000 [+0.1000, +0.5000] | **post-hoc** |
| 77 | GNN − 1.5B | allgold4 | hotpotqa | +0.4000 [+0.2000, +0.6000] | **post-hoc** |
| 78 | 3B − 1.5B | sp_recall4 | hotpotqa | +0.1167 [+0.0167, +0.2333] | **post-hoc** |
| 79 | GNN − 3B | sp_recall4 | hotpotqa | +0.1667 [+0.0667, +0.2667] | **post-hoc** |
| 80 | GNN − 1.5B | sp_recall4 | hotpotqa | +0.2833 [+0.1667, +0.4000] | **post-hoc** |
| | **80 validation comparisons listed above, none corrected for multiplicity.** A further 62 were run on train under cross-validation (40 ablations, 14 method comparisons, 8 stacker), for 142 across the project. | | | | |

## D. 3B strength check

*Source: `results/m8_emf1.json`*

| comparison | metric | MuSiQue (n=30) | HotpotQA (n=30) |
| :-- | :-- | :-- | :-- |
| 3B − 1.5B | allgold4 | -0.0333 [-0.1333, +0.0667] | +0.1000 [-0.0333, +0.2333] |
| 3B − 1.5B | sp_recall4 | +0.0000 [-0.0556, +0.0556] | +0.1167 [+0.0167, +0.2333] |
| GNN − 3B | allgold4 | +0.0000 [-0.1333, +0.1333] | +0.3000 [+0.1000, +0.5000] |
| GNN − 3B | sp_recall4 | -0.0194 [-0.1222, +0.0750] | +0.1667 [+0.0667, +0.2667] |
| GNN − 1.5B | allgold4 | -0.0333 [-0.2000, +0.1333] | +0.4000 [+0.2000, +0.6000] |
| GNN − 1.5B | sp_recall4 | -0.0194 [-0.1333, +0.0944] | +0.2833 [+0.1667, +0.4000] |

## E. Figures

`results/pareto_musique.png` and `results/pareto_hotpotqa.png`, generated by
`notebooks/pareto_analysis.py` from `results/pareto_table.json`. Three cost axes
(routing latency, LLM tokens, peak RSS) against AllGold@4 and reader EM, with
indexing and query cost shown separately and combined. Hollow markers are stages
that failed §7.3 or whose timings are indicative; the dashed line is the graph
ceiling.

## F. Reproducing the numbers

Every table above is generated by `scripts/make_report_tables.py` from the files
cited beneath it. See `results/RUNBOOK.md` for how to re-run the measurement
phases themselves.
