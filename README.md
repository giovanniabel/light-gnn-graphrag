# Light-GNN GraphRAG — resource-efficient subgraph routing for multi-hop QA

A fully local study of whether a small, query-conditioned, edge-aware GNN can
select multi-hop supporting evidence from a per-question knowledge graph **without
any LLM call at query time**. Everything runs on one laptop: Ollama for the small
language model, `BAAI/bge-small-en-v1.5` for embeddings, NetworkX + PyTorch
Geometric for the graphs. No cloud APIs.

## Preprint

This repository is the code and data release accompanying a preprint:

> Christian, G. A. (2026). *Light GNN Subgraph Routing for Multi-Hop GraphRAG : A
> Pre-Registered Study*. Zenodo. https://doi.org/10.5281/zenodo.24489118

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24489118.svg)](https://doi.org/10.5281/zenodo.24489118)

* **Record:** <https://zenodo.org/records/24489118> · published 2026-11-24 · CC BY 4.0
* **Version DOI** (this version): [10.5281/zenodo.24489118](https://doi.org/10.5281/zenodo.24489118)
* **Concept DOI** (always resolves to the latest version): [10.5281/zenodo.24489117](https://doi.org/10.5281/zenodo.24489117)
* **ORCID:** [0009-0006-3589-2055](https://orcid.org/0009-0006-3589-2055)

The Zenodo record carries the manuscript, the source code, the per-question
outputs for every method, the deviation log, the validation-read audit log and the
scripts that regenerate every table. This repository is the live version of the
same material; cite the DOI rather than the repository URL.

```bibtex
@misc{christian2026lightgnn,
  author       = {Christian, Giovanni Abel},
  title        = {Light GNN Subgraph Routing for Multi-Hop {GraphRAG} : A Pre-Registered Study},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.24489118},
  url          = {https://doi.org/10.5281/zenodo.24489118},
  note         = {Preprint}
}
```

The preprint reports the pre-registered study on 150 validation questions per
dataset **and** the replication on 350 new questions per dataset (M11 / D27).
Section 10 of [report.md](report.md) is that replication; the headline below is
the original 150-question result, which the replication revised in three places.

## Headline result

**The router learns to score edges well, and that advantage does not reach
paragraph selection.** Edge-Precision@10 beats non-learned PPR on both datasets and
in both samples — 0.605 against 0.493 (MuSiQue) and 0.749 against 0.549 (HotpotQA)
on the original 150 questions, 0.492 against 0.370 and 0.714 against 0.560 on the
350 replication questions — and removing message passing costs 7–10 points of edge
average precision. But graph-only routing does not exceed plain vector retrieval on
AllGold@4 on either dataset, in either sample.

**The reason is measured, not assumed.** The 3-hop ego-graph reaches only 79% of
MuSiQue's candidate paragraphs in the original sample and **67% in the replication**,
capping AllGold@4 at 0.380 and **0.257** respectively for any graph method, while
vector retrieval faces no such bound. The replication found this bottleneck *worse*
on unseen questions than the first measurement suggested, so the central limitation
strengthens rather than softens.

Routing is **35–180× cheaper** than LLM path tracing and uses **zero query-time LLM
tokens** by construction.

Verdicts on the four pre-registered claims, as replicated on 350 new questions per
dataset:

| claim | verdict |
| :-- | :-- |
| **C1** — the router beats vector retrieval at selecting evidence | **not supported** on both datasets |
| **C2** — learning beats non-learned graph traversal | **supported** on HotpotQA (+0.094 [+0.051, +0.140]), **inconclusive** on MuSiQue |
| **C3** — cheaper than LLM tracing at comparable quality | **cost half supported**; quality half **supported** on HotpotQA, **not supported** on MuSiQue |
| **C4** — fits in 6 GB | **supported** (peak 1.54 GB over the replication's stages) |

C2 on HotpotQA and C3's MuSiQue quality half both *changed* from inconclusive on
the original 150 questions — in each case because 350 questions resolved a
comparison 150 could not, with no point estimate reversing direction.

Full write-up with every table traced to its source file: **[report.md](report.md)**
(§10 is the replication). Current state, open issues and next steps:
**[STATUS.md](STATUS.md)**.

## Reproducing

Phases M0–M11 are complete; `results/` holds every per-question output, so all
metrics can be recomputed without re-running a model:

```bash
.venv/bin/python scripts/make_report_tables.py   # regenerates every table in report.md
.venv/bin/python notebooks/pareto_analysis.py    # regenerates the Pareto figures
.venv/bin/python -m pytest tests/ -q             # 179 tests
```

To re-run the measurement phases themselves, see **[results/RUNBOOK.md](results/RUNBOOK.md)** —
it gives the exact command, expected duration, outputs and failure recovery for
each remaining script. Environment setup:

```bash
uv venv --python 3.12
VIRTUAL_ENV=.venv uv pip install -r requirements.txt
ollama pull qwen2.5:1.5b-instruct-q4_K_M
ollama pull qwen2.5:3b-instruct-q4_K_M      # only for the M7b strength check
```

Datasets download automatically from the Hugging Face Hub on first use.

## Repo layout

```
agent.md                  the specification and single source of truth; §14 is the
                          deviation log (D1-D30), every departure with its reason
STATUS.md                 current state, settings chosen, open issues, next step
report.md                 the write-up (generated; do not hand-edit)
CLAUDE.md                 working rules for agents on this repo
configs/default.yaml      the only place hyperparameters and seeds live
src/
  data.py                 loading, splits, gold-blind pool building
  embedder.py             frozen bge wrapper + sharded disk cache
  extractor.py            SLM OpenIE with append-only cache and salvage
  graph_builder.py        nodes, seeds, ego-graph, labels, PyG conversion
  gnn_router.py           EdgeRouter, training loop, route()
  generator.py            reader via Ollama
  routing.py              paragraph scoring, hybrid tail, RRF fusion
  profiling.py            stage timer, token counter, process-tree RSS sampler
  validation_guard.py     validation reads require --allow-validation and are logged
evaluation/
  metrics.py              SP-Recall, AllGold@4 (both conditionings), EM/F1, bootstrap
  baselines.py            vector_rag, ppr_rag, edge_sim, llm_trace, RRF pairs
  cv.py / oof.py          5-fold CV, out-of-fold training and evaluation
scripts/                  one runner per phase; *.sh are standalone; ext_*.py are
                          the M11 replication (sampling, extraction, verdicts)
tests/                    179 tests
results/                  all outputs, per-question files, figures, RUNBOOK.md
notebooks/pareto_analysis.py
```

## Data and model licences

Check and comply with these yourself before redistributing anything derived from
them; this repo ships no dataset or model weights.

| asset | source | licence |
| :-- | :-- | :-- |
| MuSiQue (answerable) | `dgslibisey/MuSiQue` on the HF Hub, the official `musique_ans_v1.0` files | CC BY 4.0 (per the original MuSiQue release) |
| HotpotQA (distractor) | `hotpotqa/hotpot_qa` on the HF Hub | CC BY-SA 4.0 |
| `BAAI/bge-small-en-v1.5` | HF Hub | MIT |
| `qwen2.5:*-instruct` | Ollama | Qwen licence (Apache 2.0 for the 1.5B and 3B sizes; verify for your use) |
| this code | this repository | **no licence file yet** — default copyright applies, so reuse rights are not granted; the Zenodo record is CC BY 4.0 |

Only derived artefacts are tracked in git: result JSON, per-question outputs and
figures. Raw datasets, the embedding cache and model checkpoints are gitignored.

## Limitations, honestly

* **Per-question graphs only.** Each graph is built from one question's own
  candidate paragraphs. This is an easier setting than corpus-level GraphRAG and
  the results do not transfer to one.
* **Validation was inspected before the configuration was frozen** (deviation D12):
  the graph-ceiling and connectivity statistics were computed on both splits while
  diagnosing why the ego-graphs were fragmented. No threshold was fitted to those
  numbers and every later decision used train/dev only, but the reported validation
  ceilings are not strictly naive and are not presented as such.
* **30 of 80 measured stages are invalid** under the §7.3 swap rule — 8 of 33 in
  the original run and 22 of 47 in the replication, almost all of them
  Ollama-loaded or extraction stages. EM/F1 and all retrieval metrics are
  unaffected — the reader and the traversal decode greedily, so identical context
  gives identical output regardless of paging — but the latency and RSS figures
  from those stages are void and marked as such. The replication's retrieval sweep
  is the one fully valid group (0 of 18 invalid), after D30 removed on-the-fly
  embedding from the timed region.
* **Process RSS under-reports Ollama on Apple Silicon.** Model weights live in
  GPU-wired memory managed by the Metal driver and are not fully attributed to the
  process. The C4 memory figure is a floor, not a faithful total.
* **The C3 latency ratio mixes valid and indicative numbers**: valid routing
  medians against LLM-Trace medians whose stage failed the swap rule in every
  attempt. Read it as an order of magnitude, not to two significant figures. The
  routing medians reported in §7 also carry a measurement artefact found later
  (D30) — they were timed with a handful of embedding calls still inside the timed
  region, so the like-for-like figures are 29.0/20.9 ms rather than the published
  43.8/35.6 ms. §7 is left as published and corrected in §10.
* **158 paired bootstraps on validation, 220 across the project, none corrected**
  for multiplicity — 80 in the original run plus 78 in the replication. At 95%
  confidence roughly one interval in twenty excludes zero by chance, so no single
  borderline interval is evidence on its own. Three intervals have a bound of
  exactly 0.000 and are flagged as borderline rather than reported as findings.
* **Small samples.** 500 validation questions per dataset in total — the original
  150 plus 350 new ones scored under a pre-committed analysis plan (D27) — and the
  3B strength check is n=30 per dataset. The two question sets are **not equally
  hard**: MuSiQue's graph ceiling differs by −0.123 [−0.213, −0.034] between them,
  so absolute AllGold@4 is not comparable across them and only the within-set
  paired contrasts replicate.
* **A 1.5B reader compresses EM/F1 differences**, so retrieval metrics are the
  primary evidence. The 1.5B tracer is also a weak LLM-Trace baseline, though the
  3B check found it is not materially better on MuSiQue.
* **Three measurement faults were found late, and all of them degraded silently**
  (D28–D30). Two were the same bug in different fields: `to_pyg` zero-filled
  missing embeddings, so on the replication pools the router first ran
  **query-blind** on HotpotQA (D29) and then scored **10% of edges with no relation
  feature** (D30). Both produced plausible-looking output and neither announced
  itself; every affected figure was re-measured and the superseded runs are kept in
  `results/ext_cold_snapshot/` and `results/ext_pre_d30_snapshot/`. The lesson is
  recorded rather than papered over: a zero vector is not a neutral default, and
  D29's fix was too narrow to catch D30.
* **Weak supervision.** An edge is labelled positive iff its source paragraph is a
  gold supporting paragraph — a provenance proxy for relevance, not relevance.
* **Hyperparameters are not tuned.** The 8-configuration grid spanned less than
  half the fold-to-fold noise, so the selected setting is an arbitrary pick inside
  that noise; the finding is insensitivity, not optimisation.
