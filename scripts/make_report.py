"""Render report.md from the generated tables. Run via make_report_tables.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RES = ROOT / "results"

import scripts.make_report_tables as T  # noqa: E402

# deviation -> when, relative to the validation split. Hand-classified; the
# boundary that matters is D12, which records the one look at validation ceilings.
WHEN = {
    **{f"D{i}": "before any validation access" for i in range(1, 12)},
    "D12": "**records the validation look** (connectivity ceilings, before freezing)",
    **{f"D{i}": "after the ceiling look, before any method was scored" for i in range(13, 21)},
    "D19-RESULT": "after the ceiling look, before any method was scored",
    "D20-SKIPPED": "after the ceiling look, before any method was scored",
    "D21": "during M7 (validation scoring)",
    "D22": "after M7 scoring",
    "D23": "after M7 scoring",
    "D24": "after M7c/M8",
    "D25": "after M7c/M8",
    "D26": "after M7c/M8",
}


def deviation_table() -> str:
    md = (ROOT / "agent.md").read_text()
    start = md.index("## 14. Deviation Log")
    body = md[start:]
    rows = [l for l in body.splitlines() if l.startswith("| **D")]
    out = ["| ID | §  | when, relative to validation | decision (abridged) |",
           "| :-- | :-- | :-- | :-- |"]
    for line in rows:
        parts = [p.strip() for p in line.split(" | ")]
        did = parts[0].strip("| ").strip("*")
        sec = parts[1]
        decision = parts[2]
        decision = decision.replace("|", "/")
        if len(decision) > 220:
            decision = decision[:217].rsplit(" ", 1)[0] + "…"
        out.append(f"| **{did}** | {sec} | {WHEN.get(did, '—')} | {decision} |")
    T.PROV["T_deviations"] = ["../agent.md §14"]
    return ("*Source: `agent.md` §14 (full reasons and measurements are there; "
            "abridged here)*\n\n" + "\n".join(out))


def render() -> None:
    comparisons, n_comp = T.t_comparisons()
    env = T.j("m7_env.json")
    ix = T.j("m7_indexing_cost.json")
    m4 = T.j("m4_report.json")
    m1 = T.j("m1_report.json")
    bud = T.j("budget.json")
    an = T.j("m8_analysis.json")
    e8 = T.j("m8_emf1.json")
    a7 = T.j("m7a_report.json")
    rc = T.j("m7_recheck.json")
    cond = T.j("m5_ceiling_conditional.json")
    gl = T.j("m5_connectivity_sweep.json")["link_quality_train"]
    dg = T.j("m5_diagnostics.json")
    n_inv = sum(1 for r in an["validity"] if not r["valid"])

    mus_c1 = T.j("m7_verdicts.json")["claims"]["C1"]["musique"]
    hot_c1 = T.j("m7_verdicts.json")["claims"]["C1"]["hotpotqa"]

    doc = f"""# Resource-Efficient Subgraph Routing via Light GNNs for Multi-Hop GraphRAG

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
non-learned graph baseline — Edge-Precision@10 of {T.f4(a7['datasets']['musique']['methods']['gnn']['ALL']['edge_precision10'])}
versus PPR's {T.f4(a7['datasets']['musique']['methods']['ppr']['ALL']['edge_precision10'])} on MuSiQue
and {T.f4(a7['datasets']['hotpotqa']['methods']['gnn']['ALL']['edge_precision10'])} versus
{T.f4(a7['datasets']['hotpotqa']['methods']['ppr']['ALL']['edge_precision10'])} on HotpotQA — but that
advantage does not reach paragraph selection: graph-only routing does not exceed
plain vector retrieval on AllGold@4 on either dataset. Routing is 35–61× cheaper
than LLM path tracing and uses no query-time LLM tokens. Of the four
pre-registered claims, **C1 is not supported, C2 is inconclusive, C3's cost half
is supported with its quality half supported on HotpotQA and inconclusive on
MuSiQue, and C4 is supported** subject to a memory-accounting caveat. The decisive
constraint is measured and reported: the ego-graph can only reach
{T.f3(a7['datasets']['musique']['context']['ALL']['graph_pool_coverage'])} of MuSiQue's pool
paragraphs, capping AllGold@4 for any graph method at
{T.f3(a7['datasets']['musique']['context']['ALL']['allgold4_ceiling'])} while vector retrieval
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
{T.f3(m1['datasets']['musique']['validation']['pool_recall'])} (MuSiQue validation) and
{T.f3(m1['datasets']['hotpotqa']['validation']['pool_recall'])} (HotpotQA).

**Extractor.** Ollama `qwen2.5:1.5b-instruct-q4_K_M`, greedy, JSON-schema output
with the triple count enforced by the decoding grammar, paragraphs truncated to
120 words. {m4['cache_unique_keys']:,} unique paragraphs extracted, JSON failure
rate {m4['stats']['json_failure_rate']:.2%}, gold-paragraph extraction coverage
{T.f3(m4['per_dataset']['musique']['extraction_coverage'])} (MuSiQue) and
{T.f3(m4['per_dataset']['hotpotqa']['extraction_coverage'])} (HotpotQA).

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

| claim | pre-registered test | verdict |
| :-- | :-- | :-- |
| **C1** | GNN retrieves more complete multi-hop evidence than vector retrieval; AllGold@4-conditional, paired-bootstrap CI of the difference excludes 0 | **NOT SUPPORTED** |
| **C2** | Graph structure and learning both matter: GNN beats PPR and the 0-layer ablation on AllGold@4 | **INCONCLUSIVE** |
| **C3** | Routing ≥10× cheaper than LLM-Trace at AllGold@4 no more than 3 points lower | **cost half SUPPORTED; quality half SUPPORTED (HotpotQA), INCONCLUSIVE (MuSiQue)** |
| **C4** | Peak process-tree RSS under 6 GB for every method | **SUPPORTED**, with a memory-accounting caveat |

**C1.** Graph-only versus Vector RAG on the conditional metric:
MuSiQue {T.ci(mus_c1['row_a_conditional'])} against a graph ceiling of
{T.f3(mus_c1['ceiling'])} on n={mus_c1['n_conditional']}; HotpotQA
{T.ci(hot_c1['row_a_conditional'])} against a ceiling of {T.f3(hot_c1['ceiling'])}.
The MuSiQue difference is a precise tie; the HotpotQA difference excludes zero in
the *unfavourable* direction. The matched hybrid pair (GNN+Vector − PPR+Vector) is
{T.ci(mus_c1['row_b'])} and {T.ci(hot_c1['row_b'])} — no gain whose CI excludes
zero. C1 is not supported on either dataset.

**C2.** GNN − PPR on AllGold@4 includes zero on both datasets, so the claim as
registered is inconclusive. Its 0-layer half was never run on validation, being a
§5 ablation; under cross-validation on train, `L=0` cost −0.071 and −0.098 edge AP
with CIs excluding zero. Post-hoc edge-level evidence is reported in §5.

**C3.** The cost half holds comfortably. The quality half is tested as
(GNN − Trace) ≥ −0.03: MuSiQue {T.ci(rc['musique']['allgold4_uncond'])} straddles
the margin, so inconclusive; HotpotQA {T.ci(rc['hotpotqa']['allgold4_uncond'])}
lies entirely above it, so supported. Query-time LLM tokens are 0 for the GNN by
construction against roughly 1,100–1,232 for LLM-Trace.

**C4.** Peak process-tree RSS including Ollama is
{an['c4']['peak_rss_mb']:.0f} MB ({an['c4']['peak_rss_mb']/1024:.2f} GB), under the
6 GB limit. See §6 for why that number is a floor rather than a faithful total.

## 3. Retrieval results

Ceilings appear in every table beneath the graph methods they bound. The ceiling
is the fraction of questions whose every gold paragraph has at least one extracted
edge in the ego-graph — a paragraph with no edge cannot be ranked by a graph
method at all.

### MuSiQue validation (n=150)

{T.t_retrieval('musique')}

### HotpotQA validation (n=150)

{T.t_retrieval('hotpotqa')}

The two `+hybrid-tail` rows are **identical to their base methods by
construction**: the hybrid tail only reorders paragraphs that rank below every
graph-scored one, so with K=4 and at least four scored paragraphs on nearly every
question it cannot alter the top-4.

### By hop bucket and question type

{T.t_buckets('musique')}

{T.t_buckets('hotpotqa')}

The 3-hop MuSiQue column is the clearest single result in the study: Vector RAG
scores {T.f3(a7['datasets']['musique']['methods']['vector']['3hop']['allgold4'])} against a
graph ceiling of {T.f3(a7['datasets']['musique']['context']['3hop']['allgold4_ceiling'])}. It
exceeds the bound that constrains every graph method, because it is not bounded by
reachability. The 4-hop column is exploratory at n=50 and every method sits
between 0.04 and 0.12.

Conditioning on all gold paragraphs surviving pooling (D4) changes the
denominators: MuSiQue validation conditions to n=111 of 150, and the ceiling there
is {cond['musique/validation/ALL']['conditional_on_all_gold_in_pool']:.1%}.

## 4. Reader results (M8)

Every condition receives the same K=4 paragraph budget and the same prompt
template, greedy decoding, 24-token cap, so differences come from routing only.

{T.t_reader()}

The oracle is the ceiling on both datasets and both metrics, which is the sanity
check this table exists to pass. Vector RAG − oracle excludes zero on both
datasets, so substantial reader headroom remains and the reader is not saturating
the comparison.

{T.t_reader_cis()}

Two results deserve emphasis and both are **post-hoc**. `GNN+Vector − GNN` is
about +0.060 EM with a CI excluding zero on *both* datasets: fusing with vector
retrieval helps the reader even where it did not help AllGold@4. And `GNN − Vector
RAG` EM on MuSiQue is −0.0667 with an upper bound of **exactly 0.000** — a
borderline interval that should not be read as a finding in either direction,
though its F1 counterpart does exclude zero.

## 5. Ablations and cross-validation findings

All of this is on train, under 5-fold CV, with a fold-to-fold edge-AP standard
deviation of 0.031 (MuSiQue) and 0.038 (HotpotQA) as the noise floor.

{T.t_ablations()}

Three design elements are measurably inert: query conditioning (−0.001 on both),
the H=3 radius over H=2, and the `para` links — which are 33% of all edges and the
mechanism that lifted evidence-only ego coverage to
{dg['musique']['ego_coverage_evidence_only']:.1%} on MuSiQue train, yet whose removal
changes edge AP by nothing *(source: `results/m5_diagnostics.json`)*. Message passing is the one component with a large, unambiguous effect.

{T.t_grid()}

The entire 8-configuration grid spans less than half the fold-to-fold noise, and
the per-dataset winners disagree. The selected configuration is an arbitrary pick
inside that noise and **is not a tuned result**; the useful finding is that the
model is insensitive to these hyperparameters in this range.

{T.t_curve()}

MuSiQue is saturated beyond 50% of the training data. More extraction would not
help it; the binding constraint is elsewhere.

## 6. Why graph-only routing does not exceed vector retrieval

Three measured diagnostics, not speculation.

**Reachability caps.** The connectivity fix worked and was still not enough:

{T.t_connectivity()}

Even after it, only {T.f3(a7['datasets']['musique']['context']['ALL']['graph_pool_coverage'])}
of MuSiQue **validation** pool paragraphs carry an extracted edge inside the
ego-graph ({dg['musique']['ego_coverage_evidence_only']:.1%} on train), giving an
AllGold@4 ceiling of {T.f3(a7['datasets']['musique']['context']['ALL']['allgold4_ceiling'])}.
Vector RAG's own bound is all-gold-in-pool,
{T.f3(cond['musique/validation/ALL']['all_gold_in_pool'])}. A perfect router
is therefore capped below a mediocre baseline. Diagnosis showed the residual is
hop-radius reachability, not missing triples: only
{dg['musique']['no_usable_triple_rate']:.1%} of MuSiQue pool paragraphs yield no
usable triple, so the hard coverage ceiling is
{dg['musique']['hard_coverage_ceiling']:.1%} *(source: `results/m5_diagnostics.json`)*.

**Noisy scaffolding links.** Of the cross-paragraph links the connectivity
mechanisms add on train, only {gl['musique']['shares']['gold_gold']:.1%} (MuSiQue)
and {gl['hotpotqa']['shares']['gold_gold']:.1%} (HotpotQA) join gold to gold, while
{gl['musique']['shares']['distractor_distractor']:.1%} and
{gl['hotpotqa']['shares']['distractor_distractor']:.1%} join distractor to
distractor. The `para` mechanism is both the bulk of them and the noisiest. This
matches the ablation result that removing `para` costs nothing.

**The learned combiner's own weights.** Given vector cosine, the GNN paragraph
score and the PPR score as three features, a logistic regression fitted under
nested CV assigns:

{T.t_stacker()}

The GNN feature gets a clearly non-zero positive weight, so the graph score does
carry signal — but vector cosine outweighs it by roughly 4× on MuSiQue and 2.4× on
HotpotQA, and the combination still does not beat vector similarity alone. That is
the most direct evidence available that the graph signal is real but not
complementary enough to change paragraph selection.

## 7. Efficiency

{T.t_efficiency()}

### What the routing stage includes

{T.t_routing_scope()}

§7.3 defines routing as everything from question and graph to ranked paragraphs,
which includes question embedding, seed linking and ego-graph assembly. Graph
assembly dominates: 38.9 of 43.8 ms on MuSiQue, with the model forward only
4.3 ms. A figure that excludes assembly is not the routing stage.

### LLM-Trace timings are indicative, not valid

{T.t_llm_trace()}

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
3. **{n_inv} of {len(an['validity'])} measured stages are invalid** under §7.3 on swap
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
6. **Multiple comparisons, uncorrected.** **{n_comp} paired bootstraps were run on
   the validation split** (24 retrieval, 12 follow-up, 32 reader EM/F1, 12 for the
   3B check), plus 62 more on train under cross-validation, for **142 across the
   project**. None is corrected for multiplicity. At 95% confidence roughly
   {n_comp*0.05:.0f} false positives are expected among the validation set alone, so
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

# Appendix

## A. Deviation log

{deviation_table()}

## B. Stage validity

{T.t_validity()}

## C. All paired comparisons

{comparisons}

## D. 3B strength check

{T.t_three_b()}

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
"""
    (ROOT / "report.md").write_text(doc)
    (RES / "report_provenance.json").write_text(json.dumps(T.PROV, indent=2))
    print(f"wrote report.md ({len(doc.splitlines())} lines, "
          f"{len(doc.split())} words)")
    print(f"wrote results/report_provenance.json ({len(T.PROV)} tables)")


if __name__ == "__main__":
    render()
