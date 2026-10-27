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
