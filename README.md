# Light-GNN GraphRAG — resource-efficient subgraph routing for multi-hop QA

A fully local study of whether a small, query-conditioned, edge-aware GNN can
select multi-hop supporting evidence from a per-question knowledge graph **without
any LLM call at query time**. Everything runs on one laptop: Ollama for the small
language model, `BAAI/bge-small-en-v1.5` for embeddings, NetworkX + PyTorch
Geometric for the graphs. No cloud APIs.

## Preprint

This repository is the code and data release accompanying a preprint:

> Christian, G. A. (2026). *Light GNN Subgraph Routing for Multi-Hop GraphRAG A
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
  title        = {Light GNN Subgraph Routing for Multi-Hop GraphRAG A Pre-Registered Study},
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

The router learns edge scoring that is clearly better than a non-learned graph
baseline — Edge-Precision@10 of 0.605 against PPR's 0.493 on MuSiQue and 0.749
against 0.549 on HotpotQA, with paired-bootstrap CIs excluding zero — and removing
message passing costs it 7–10 points of edge average precision. That advantage does
not reach paragraph selection: graph-only routing does not exceed plain vector
retrieval on AllGold@4 on either dataset, because the 3-hop ego-graph reaches only
79% of MuSiQue's candidate paragraphs and so caps AllGold@4 at 0.380 for any graph
method, while vector retrieval faces no such bound. Routing is 35–61× cheaper than
LLM path tracing and uses zero query-time LLM tokens, so of the four pre-registered
claims **C1 is not supported, C2 is inconclusive, C3's cost half is supported
(quality half supported on HotpotQA, inconclusive on MuSiQue), and C4 is
supported**.

Full write-up with every table traced to its source file: **[report.md](report.md)**.

## Reproducing

Phases M0–M8 are already complete; `results/` holds every per-question output, so
all metrics can be recomputed without re-running a model:

```bash
.venv/bin/python scripts/make_report_tables.py   # regenerates every table in report.md
.venv/bin/python notebooks/pareto_analysis.py    # regenerates the Pareto figures
.venv/bin/python -m pytest tests/ -q             # 174 tests
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
                          deviation log (D1-D26), every departure with its reason
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
scripts/                  one runner per phase; *.sh are standalone
tests/                    174 tests
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
| this code | — | add a licence file before publishing |

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
* **8 of 33 measured stages are invalid** under the §7.3 swap rule, including both
  LLM-Trace stages and five reader stages. EM/F1 and all retrieval metrics are
  unaffected — the reader and the traversal decode greedily, so identical context
  gives identical output regardless of paging — but the latency and RSS figures
  from those stages are void and marked as such.
* **Process RSS under-reports Ollama on Apple Silicon.** Model weights live in
  GPU-wired memory managed by the Metal driver and are not fully attributed to the
  process. The C4 memory figure is a floor, not a faithful total.
* **The C3 latency ratio mixes valid and indicative numbers**: valid routing
  medians against LLM-Trace medians whose stage failed the swap rule in three of
  four attempts.
* **80 paired bootstraps on validation, 142 across the project, none corrected**
  for multiplicity. Two intervals have a bound of exactly 0.000 and are flagged as
  borderline rather than reported as findings.
* **Small samples.** 150 questions per dataset per split; the 3B strength check is
  n=30 per dataset.
* **A 1.5B reader compresses EM/F1 differences**, so retrieval metrics are the
  primary evidence. The 1.5B tracer is also a weak LLM-Trace baseline, though the
  3B check found it is not materially better on MuSiQue.
* **Weak supervision.** An edge is labelled positive iff its source paragraph is a
  gold supporting paragraph — a provenance proxy for relevance, not relevance.
* **Hyperparameters are not tuned.** The 8-configuration grid spanned less than
  half the fold-to-fold noise, so the selected setting is an arbitrary pick inside
  that noise; the finding is insensitivity, not optimisation.
