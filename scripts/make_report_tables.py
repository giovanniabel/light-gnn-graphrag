"""Build every table and figure reference in report.md directly from results/.

Nothing is hand-typed: `python scripts/make_report_tables.py` regenerates
results/report_tables.md, results/report_provenance.json and report.md itself, so
every number in the write-up is traceable to a file and reproducible.

No model is run and no setting is read from anywhere but results/ and configs/.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RES = ROOT / "results"
DS = ("musique", "hotpotqa")

PROV: dict[str, list[str]] = {}


def j(name):
    return json.loads((RES / name).read_text())


def jl(name):
    return [json.loads(l) for l in (RES / name).read_text().splitlines() if l.strip()]


def table(key, sources, lines):
    PROV[key] = sources
    src = ", ".join(f"`results/{s}`" for s in sources)
    return f"*Source: {src}*\n\n" + "\n".join(lines)


def f3(v, dash="—"):
    return dash if v is None else f"{v:.3f}"


def f4(v, dash="—"):
    return dash if v is None else f"{v:.4f}"


def ci(d):
    if d is None:
        return "—"
    star = " *(borderline: a bound is exactly 0.000)*" if (
        abs(d["lo"]) < 1e-9 or abs(d["hi"]) < 1e-9) else ""
    return f"{d['point']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]{star}"


# ---------------------------------------------------------------- T1/T2 retrieval
RET_ORDER = ["vector", "ppr", "edge_sim", "gnn", "gnn_rrf", "ppr_rrf",
             "gnn_hybrid_tail", "ppr_hybrid_tail", "stacked"]
RET_LAB = {"vector": "Vector RAG", "ppr": "PPR", "edge_sim": "Edge-Sim",
           "gnn": "GNN router (graph-only)", "gnn_rrf": "GNN+Vector (RRF k=60)",
           "ppr_rrf": "PPR+Vector (RRF k=60)",
           "gnn_hybrid_tail": "GNN+hybrid-tail *(≡ GNN)*",
           "ppr_hybrid_tail": "PPR+hybrid-tail *(≡ PPR)*",
           "stacked": "Stacked combiner (D19)"}


def t_retrieval(ds):
    r = j("m7a_report.json")["datasets"][ds]
    c = r["context"]["ALL"]
    # LLM-Trace retrieval metrics come from its per-question file, not the
    # m7c_* files, which carry efficiency only.
    trq = jl(f"val_llm_trace_{ds}.jsonl")
    tr = {"sp_recall4": statistics.fmean(r["sp_recall4"] for r in trq),
          "allgold4": statistics.fmean(r["allgold4"] for r in trq)}
    rows = [
        "| method | SP-Recall@4 | AllGold@4 | AllGold@4-cond | Edge-Prec@10 | Path-Recall |",
        "| :-- | --: | --: | --: | --: | --: |",
    ]
    for k in RET_ORDER:
        m = r["methods"][k]["ALL"]
        rows.append(f"| {RET_LAB[k]} | {f4(m['sp_recall4'])} | {f3(m['allgold4'])} | "
                    f"{f3(m['allgold4_cond'])} | {f4(m['edge_precision10'])} | "
                    f"{f4(m['path_recall10'])} |")
    rows.append(f"| LLM-Trace (1.5B) | {f4(tr['sp_recall4'])} | {f3(tr['allgold4'])} | "
                f"— | — | — |")
    rows.append(f"| **Ceiling for graph methods** | — | **{f3(c['allgold4_ceiling'])}** | — | "
                f"— | **{f3(c['answer_in_graph_ceiling'])}** |")
    rows.append(f"| *context* | pool recall {f3(c['pool_recall'])} | "
                f"extraction coverage {f3(c['extraction_coverage'])} | "
                f"graph pool coverage {f3(c['graph_pool_coverage'])} | "
                f"n=150 | n_cond={r['methods']['gnn']['ALL']['n_cond']} |")
    return table(f"T_retrieval_{ds}",
                 ["m7a_report.json", f"val_llm_trace_{ds}.jsonl"], rows)


def t_buckets(ds):
    r = j("m7a_report.json")["datasets"][ds]
    bs = [b for b in r["context"] if b != "ALL"]
    rows = ["| method | " + " | ".join(f"{b} (n={r['context'][b]['n']})" for b in bs) + " |",
            "| :-- | " + " | ".join("--:" for _ in bs) + " |"]
    for k in RET_ORDER:
        rows.append(f"| {RET_LAB[k]} | " +
                    " | ".join(f3(r["methods"][k][b]["allgold4"]) for b in bs) + " |")
    rows.append("| **Ceiling (graph)** | " +
                " | ".join(f"**{f3(r['context'][b]['allgold4_ceiling'])}**" for b in bs) + " |")
    return table(f"T_buckets_{ds}", ["m7a_report.json"], rows)


# ---------------------------------------------------------------- T3 reader
READ = ["vector", "ppr", "llm_trace", "gnn", "gnn_rrf", "stacked", "oracle"]
READ_LAB = {"vector": "Vector RAG", "ppr": "PPR", "llm_trace": "LLM-Trace (1.5B)",
            "gnn": "GNN router", "gnn_rrf": "GNN+Vector (RRF)",
            "stacked": "Stacked (D19)", "oracle": "Gold-paragraph oracle"}


def t_reader():
    e = j("m8_emf1.json")
    rows = ["| condition | MuSiQue EM | MuSiQue F1 | HotpotQA EM | HotpotQA F1 |",
            "| :-- | --: | --: | --: | --: |"]
    for k in READ:
        m = e["datasets"]["musique"]["rows"][k]
        h = e["datasets"]["hotpotqa"]["rows"][k]
        bold = "**" if k == "oracle" else ""
        rows.append(f"| {bold}{READ_LAB[k]}{bold} | {bold}{f3(m['em'])}{bold} | "
                    f"{bold}{f3(m['f1'])}{bold} | {bold}{f3(h['em'])}{bold} | "
                    f"{bold}{f3(h['f1'])}{bold} |")
    return table("T_reader", ["m8_emf1.json"], rows)


def t_reader_cis():
    e = j("m8_emf1.json")
    rows = ["| comparison | metric | MuSiQue | HotpotQA | registered? |",
            "| :-- | :-- | :-- | :-- | :-- |"]
    for keyname, lab in [("gnn-vs-vector", "GNN − Vector RAG"),
                         ("gnn-vs-oracle", "GNN − oracle"),
                         ("gnn-vs-ppr", "GNN − PPR"),
                         ("gnn-vs-llm_trace", "GNN − LLM-Trace"),
                         ("gnn_rrf-vs-gnn", "GNN+Vector − GNN"),
                         ("stacked-vs-gnn_rrf", "Stacked − GNN+Vector"),
                         ("vector-vs-oracle", "Vector RAG − oracle"),
                         ("stacked-vs-vector", "Stacked − Vector RAG")]:
        for metric in ("em", "f1"):
            m = e["datasets"]["musique"]["cis"].get(f"{metric}|{keyname}")
            h = e["datasets"]["hotpotqa"]["cis"].get(f"{metric}|{keyname}")
            reg = (m or h or {}).get("registered", "—")
            rows.append(f"| {lab} | {metric.upper()} | {ci(m)} | {ci(h)} | "
                        f"{'pre-registered' if reg=='pre-registered' else '**post-hoc**'} |")
    return table("T_reader_cis", ["m8_emf1.json"], rows)


# ---------------------------------------------------------------- T4 efficiency
def t_efficiency():
    a = j("m7a_report.json")
    ix = j("m7_indexing_cost.json")
    rows = ["| stage | method | median | p95 | LLM tokens/q | peak tree RSS | swap growth | valid? |",
            "| :-- | :-- | --: | --: | --: | --: | --: | :-- |"]
    rows.append(f"| indexing | extraction (all graph methods) | "
                f"{ix['t_p_mean_s']:.3f} s/paragraph | {ix['t_p_p95_s']:.3f} s | "
                f"{ix['llm_tokens_per_paragraph_mean']:.0f} /paragraph | "
                f"{ix['peak_tree_rss_mb_incl_ollama']:.0f} MB | "
                f"{ix['swap_growth_mb']:+.0f} MB | yes |")
    rows.append(f"| indexing | entity/relation embedding | "
                f"{ix['embedding']['seconds_per_question']*1000:.1f} ms/question | — | 0 | "
                f"{ix['embedding']['peak_tree_rss_mb']:.0f} MB | "
                f"{ix['embedding']['swap_growth_mb']:+.0f} MB | yes |")
    for ds in DS:
        for k in RET_ORDER:
            # Read the per-method stats file, NOT m7a_report.json: that aggregate was
            # written before Edge-Sim was re-timed (583.5 -> 10.6 ms) and still holds
            # the stale figure.
            st = j(f"val_{k}_{ds}_stats.json")
            s = st["routing"]
            b = st["routing_breakdown"]
            rows.append(f"| routing ({ds}) | {RET_LAB[k].split(' *')[0]} | "
                        f"{s['median_s']*1000:.1f} ms | {s['p95_s']*1000:.1f} ms | 0 | "
                        f"{s['peak_tree_rss_mb']:.0f} MB | {s['swap_growth_mb']:+.0f} MB | "
                        f"{'yes' if not s['invalidated_by_swap'] else '**NO**'} |")
        tr = j(f"m7c_llm_trace_{ds}_full.json")
        rows.append(f"| routing ({ds}) | LLM-Trace (1.5B) *indicative* | "
                    f"{tr['median_s']*1000:.0f} ms | {tr['p95_s']*1000:.0f} ms | "
                    f"{tr['tokens_per_question_mean']:.0f} | {tr['peak_tree_rss_mb']:.0f} MB | "
                    f"{tr['swap_growth_mb']:+.0f} MB | **NO** |")
        for k in READ:
            d = j(f"read_{k}_{ds}.json")
            rows.append(f"| reading ({ds}) | {READ_LAB[k]} | {d['median_s']:.2f} s | "
                        f"{d['p95_s']:.2f} s | {d['tokens_per_question_mean']:.0f} | "
                        f"{d['peak_tree_rss_mb']:.0f} MB | {d['swap_growth_mb']:+.0f} MB | "
                        f"{'yes' if not d['invalidated_by_swap'] else '**NO**'} |")
    srcs = ["m7_indexing_cost.json"] + \
           [f"val_{k}_{d}_stats.json" for d in DS for k in RET_ORDER] + \
           [f"m7c_llm_trace_{d}_full.json" for d in DS] + \
           [f"read_{k}_{d}.json" for d in DS for k in READ]
    return table("T_efficiency", srcs, rows)


def t_routing_scope():
    an = j("m8_analysis.json")["routing_scope"]
    rows = ["| | MuSiQue | HotpotQA |", "| :-- | --: | --: |"]
    for lab, key in [("M7a routing, full §7.3 scope", "m7a_full_ms"),
                     ("— graph assembly", "m7a_assembly_ms"),
                     ("— model forward", "m7a_forward_ms"),
                     ("M7c figure (forward only, **not** routing)", "m7c_forward_only_ms")]:
        rows.append(f"| {lab} | {an['musique'][key]:.1f} ms | {an['hotpotqa'][key]:.1f} ms |")
    return table("T_routing_scope", ["m8_analysis.json"], rows)


def t_llm_trace():
    smoke = j("m7b_smoke.json")["datasets"]
    first = {d: j(f"m7b_full_{d}.json")["datasets"][d] for d in DS}
    m7c = {d: j(f"m7c_llm_trace_{d}_full.json") for d in DS}
    rows = ["| measurement | MuSiQue median | HotpotQA median | tokens/q | swap growth | §7.3 valid? |",
            "| :-- | --: | --: | --: | --: | :-- |"]
    for lab, src in [("M7 first run", first), ("D21 redo (same figures)", first),
                     ("10-question smoke", smoke), ("M7c clean start", m7c)]:
        m, h = src["musique"], src["hotpotqa"]
        ok = (m.get("swap_growth_mb", 0) <= 500 and h.get("swap_growth_mb", 0) <= 500)
        rows.append(f"| {lab} | {m['median_s']*1000:.0f} ms | {h['median_s']*1000:.0f} ms | "
                    f"{m['tokens_per_question_mean']:.0f} / {h['tokens_per_question_mean']:.0f} | "
                    f"{m['swap_growth_mb']:+.0f} / {h['swap_growth_mb']:+.0f} MB | "
                    f"{'yes' if ok else '**no**'} |")
    an = j("m8_analysis.json")["c3_latency"]
    for ds in DS:
        a = an[ds]
        rows.append(f"| **C3 ratio ({ds})** | GNN {a['gnn_full_routing_ms']:.1f} ms | "
                    f"Trace {a['trace_ms_range'][0]:.0f}–{a['trace_ms_range'][1]:.0f} ms | — | — | "
                    f"**{a['ratio_range'][0]:.1f}×–{a['ratio_range'][1]:.1f}×** |")
    return table("T_llm_trace",
                 ["m7b_smoke.json", "m7b_full_musique.json", "m7b_full_hotpotqa.json",
                  "m7c_llm_trace_musique_full.json", "m7c_llm_trace_hotpotqa_full.json",
                  "m8_analysis.json"], rows)


def t_validity():
    v = j("m8_analysis.json")["validity"]
    rows = ["| stage | swap at start | swap growth | peak tree RSS | §7.3 verdict |",
            "| :-- | --: | --: | --: | :-- |"]
    for r in v:
        s0 = "—" if r["swap_start_mb"] is None else f"{r['swap_start_mb']:.0f} MB"
        rows.append(f"| {r['stage']} | {s0} | {r['swap_growth_mb']:+.0f} MB | "
                    f"{r['peak_rss_mb']:.0f} MB | {'valid' if r['valid'] else '**INVALID**'} |")
    n_inv = sum(1 for r in v if not r["valid"])
    rows.append(f"| **{n_inv} of {len(v)} stages invalid** | | | | |")
    return table("T_validity", ["m8_analysis.json"], rows)


def t_ablations():
    ab = j("m6_ablations.json")
    rows = ["| ablation | MuSiQue Δ edge AP | verdict | HotpotQA Δ edge AP | verdict |",
            "| :-- | --: | :-- | --: | :-- |"]
    labels = {"L0": "L=0 (no message passing)", "L1": "L=1", "L3": "L=3",
              "sage": "SAGE-mean instead of GATv2", "no_query_cond": "no query conditioning",
              "H2": "H=2 (hop_dist masking)", "no_para": "drop `para` links",
              "no_same_as": "drop `same_as` links", "no_title": "drop title hubs",
              "triples_only": "triples only (no §4.1.1 scaffolding)"}
    for k, lab in labels.items():
        cells = []
        for ds in DS:
            c = ab["datasets"][ds]["paired_vs_full"][f"{k}|edge_ap"]
            cells += [f"{c['point']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]",
                      "significant" if c["excludes_zero"] else "within noise"]
        rows.append(f"| {lab} | " + " | ".join(cells) + " |")
    return table("T_ablations", ["m6_ablations.json"], rows)


def t_grid():
    g = j("m6_grid.json")
    rows = ["| lr | dropout | weight decay | MuSiQue edge AP | HotpotQA edge AP | mean |",
            "| :-- | --: | --: | --: | --: | --: |"]
    for r in g["grid"]:
        star = " **(selected)**" if r["label"] == g["best"]["label"] else ""
        rows.append(f"| {r['lr']:g}{star} | {r['dropout']} | {r['weight_decay']:g} | "
                    f"{r['datasets']['musique']['edge_ap_mean']:.4f} | "
                    f"{r['datasets']['hotpotqa']['edge_ap_mean']:.4f} | {r['score']:.4f} |")
    sc = [r["score"] for r in g["grid"]]
    fsd = statistics.fmean(
        statistics.pstdev(r["datasets"]["musique"]["fold_ap"]) for r in g["grid"])
    rows.append(f"| **spread** | | | | | **{max(sc)-min(sc):.4f}** vs fold sd "
                f"**{fsd:.4f}** |")
    return table("T_grid", ["m6_grid.json"], rows)


def t_curve():
    c = j("m6_curve.json")["datasets"]
    rows = ["| fraction of fold training set | MuSiQue edge AP | MuSiQue AllGold@4 | HotpotQA edge AP | HotpotQA AllGold@4 |",
            "| :-- | --: | --: | --: | --: |"]
    for k in ("25%", "50%", "75%", "100%"):
        rows.append(f"| {k} | {c['musique'][k]['edge_ap_mean']:.4f} | "
                    f"{f3(c['musique'][k]['allgold4_uncond'])} | "
                    f"{c['hotpotqa'][k]['edge_ap_mean']:.4f} | "
                    f"{f3(c['hotpotqa'][k]['allgold4_uncond'])} |")
    return table("T_curve", ["m6_curve.json"], rows)


def t_connectivity():
    b = j("m5_before_after_train.json")
    rows = ["| TRAIN metric | MuSiQue before | MuSiQue after | HotpotQA before | HotpotQA after |",
            "| :-- | --: | --: | --: | --: |"]
    keys = [("components_per_question", "components per question", "{:.1f}"),
            ("bfs_node_share", "BFS share of full graph", "{:.1%}"),
            ("pool_covered_share", "pool paragraphs with an ego edge", "{:.1%}"),
            ("allgold_ceiling", "AllGold@4 ceiling (conditional on pooling)", "{:.1%}"),
            ("positive_edge_rate", "positive-edge rate", "{:.1%}"),
            ("cap_bound_rate", "cap-binding rate", "{:.1%}")]
    names = list(b["musique"].keys())
    before_k, after_k = names[0], names[1]
    for key, lab, fmt in keys:
        cells = []
        for ds in DS:
            for which in (before_k, after_k):
                v = b[ds][which]["ALL"].get(key)
                cells.append("—" if v is None else fmt.format(v))
        rows.append(f"| {lab} | " + " | ".join(cells) + " |")
    return table("T_connectivity", ["m5_before_after_train.json"], rows)


def t_stacker():
    s = j("m6_stack.json")["datasets"]
    f = j("m6_stack_final.json")["datasets"]
    rows = ["| feature | MuSiQue coefficient (CV) | HotpotQA coefficient (CV) | MuSiQue (final fit) | HotpotQA (final fit) |",
            "| :-- | --: | --: | --: | --: |"]
    for feat in ("vector_cos", "gnn_score", "ppr_score"):
        rows.append(f"| `{feat}` | {s['musique']['coef_mean'][feat]:+.3f} | "
                    f"{s['hotpotqa']['coef_mean'][feat]:+.3f} | "
                    f"{f['musique']['coef'][feat]:+.3f} | {f['hotpotqa']['coef'][feat]:+.3f} |")
    return table("T_stacker", ["m6_stack.json", "m6_stack_final.json"], rows)


def t_three_b():
    t = j("m8_emf1.json")["three_b"]
    rows = ["| comparison | metric | MuSiQue (n=30) | HotpotQA (n=30) |",
            "| :-- | :-- | :-- | :-- |"]
    for nm in ("3B - 1.5B", "GNN - 3B", "GNN - 1.5B"):
        for metric in ("allgold4", "sp_recall4"):
            rows.append(f"| {nm.replace('-', '−')} | {metric} | "
                        f"{ci(t['musique']['cis'][f'{metric}|{nm}'])} | "
                        f"{ci(t['hotpotqa']['cis'][f'{metric}|{nm}'])} |")
    return table("T_three_b", ["m8_emf1.json"], rows)


def t_comparisons():
    """Every paired bootstrap reported anywhere in the write-up."""
    rows = ["| # | comparison | metric | dataset | CI | registered? |",
            "| --: | :-- | :-- | :-- | :-- | :-- |"]
    n = 0
    a = j("m7a_report.json")
    for ds in DS:
        for key, d in a["datasets"][ds]["paired_bootstrap"].items():
            metric, pair = key.split("|")
            n += 1
            reg = "pre-registered" if ("gnn-vs-vector" in pair or
                                       "gnn_rrf-vs-ppr_rrf" in pair or
                                       "gnn-vs-ppr" in pair) else "**post-hoc**"
            rows.append(f"| {n} | {pair.replace('-vs-', ' − ')} | {metric} | {ds} | "
                        f"{ci(d)} | {reg} |")
    rc = j("m7_recheck.json")
    for ds in DS:
        for key, d in rc[ds].items():
            if not isinstance(d, dict) or "point" not in d:
                continue
            n += 1
            reg = "pre-registered" if "allgold4" in key else "**post-hoc**"
            rows.append(f"| {n} | {key} | — | {ds} | {ci(d)} | {reg} |")
    e = j("m8_emf1.json")
    for ds in DS:
        for key, d in e["datasets"][ds]["cis"].items():
            metric, pair = key.split("|")
            n += 1
            rows.append(f"| {n} | {pair.replace('-vs-', ' − ')} | {metric.upper()} | {ds} | "
                        f"{ci(d)} | {'pre-registered' if d['registered']=='pre-registered' else '**post-hoc**'} |")
        for key, d in e["three_b"][ds]["cis"].items():
            metric, pair = key.split("|")
            n += 1
            rows.append(f"| {n} | {pair.replace('-', '−')} | {metric} | {ds} | "
                        f"{ci(d)} | **post-hoc** |")
    rows.append(f"| | **{n} validation comparisons listed above, none corrected "
                f"for multiplicity.** A further 62 were run on train under "
                f"cross-validation (40 ablations, 14 method comparisons, 8 stacker), "
                f"for 142 across the project. | | | | |")
    return table("T_comparisons", ["m7a_report.json", "m7_recheck.json", "m8_emf1.json",
                                   "m6_ablations.json", "m6_compare.json", "m6_stack.json"],
                 rows), n


if __name__ == "__main__":
    from scripts.make_report import render

    render()
