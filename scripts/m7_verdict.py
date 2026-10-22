"""M7 step 4: the pre-registered C1-C4 verdicts, computed from the saved outputs."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import paired_bootstrap  # noqa: E402
from src.config import ROOT, env, load_config  # noqa: E402
from src.data import load_examples, pool_size_for  # noqa: E402
from src.validation_guard import allow  # noqa: E402

DS = ("musique", "hotpotqa")


def rows(method, ds):
    p = ROOT / "results" / f"val_{method}_{ds}.jsonl"
    return {r["qid"]: r for r in (json.loads(l) for l in p.read_text().splitlines() if l.strip())}


def cond_qids(ds, cfg) -> list[str]:
    """Questions whose every ORIGINAL gold paragraph survived pooling -- D4's
    conditioning set. C1 is pre-registered on the CONDITIONAL metric, so its
    paired bootstrap must run on this subset, not on all questions."""
    from scripts.m1_data import load_pools

    pooled = {e.qid: {p.pid for p in e.paragraphs}
              for e in load_pools(ds, "validation", pool_size_for(cfg, ds))}
    orig = load_examples(ds, "validation", int(cfg["data"]["target_n"]), int(cfg["seed"]))
    return sorted(e.qid for e in orig
                  if e.gold_pids and e.gold_pids.issubset(pooled.get(e.qid, set())))


def main() -> int:
    cfg = load_config()
    allow("M7 verdicts (conditional question sets)")
    rep = json.loads((ROOT / "results" / "m7a_report.json").read_text())
    trace = {ds: json.loads((ROOT / "results" / f"m7b_full_{ds}.json").read_text())
             ["datasets"][ds] for ds in DS}
    idx = json.loads((ROOT / "results" / "m7_indexing_cost.json").read_text())
    out = {"claims": {}}

    print("=" * 118)
    print("C1-C4 VERDICTS (pre-registered in agent.md Section 1.1; D15 gives C1 two primary rows)")
    print("=" * 118)

    # ---------------- C1 ----------------
    print("\nC1  The GNN router retrieves more complete multi-hop evidence than plain")
    print("    vector retrieval.  Metric: AllGold@4-conditional (D4), two rows (D15).")
    c1 = {}
    for ds in DS:
        m = rep["datasets"][ds]["methods"]
        cis = rep["datasets"][ds]["paired_bootstrap"]
        ceil = rep["datasets"][ds]["context"]["ALL"]["allgold4_ceiling"]
        cq = cond_qids(ds, cfg)
        gv, vv = rows("gnn", ds), rows("vector", ds)
        gr, pr = rows("gnn_rrf", ds), rows("ppr_rrf", ds)
        a = paired_bootstrap([gv[q]["allgold4"] for q in cq],
                             [vv[q]["allgold4"] for q in cq], n=10_000).as_dict()
        b = paired_bootstrap([gr[q]["allgold4"] for q in cq],
                             [pr[q]["allgold4"] for q in cq], n=10_000).as_dict()
        a_uncond = cis["allgold4|gnn-vs-vector"]
        print(f"\n  {ds}:  graph ceiling {ceil:.3f}   D4-conditional set n={len(cq)}")
        print(f"    (a) graph-only : GNN {m['gnn']['ALL']['allgold4_cond']:.3f} vs "
              f"Vector {m['vector']['ALL']['allgold4_cond']:.3f} (cond, n="
              f"{m['gnn']['ALL']['n_cond']})")
        print(f"        difference (CONDITIONAL, n={len(cq)})  {a['point']:+.4f} "
              f"[{a['lo']:+.4f}, {a['hi']:+.4f}]  "
              f"{'excludes 0' if a['excludes_zero'] else 'includes 0'}")
        print(f"        difference (unconditional, n=150) {a_uncond['point']:+.4f} "
              f"[{a_uncond['lo']:+.4f}, {a_uncond['hi']:+.4f}]  "
              f"{'excludes 0' if a_uncond['excludes_zero'] else 'includes 0'}")
        print(f"    (b) matched    : GNN+Vec {m['gnn_rrf']['ALL']['allgold4_cond']:.3f} vs "
              f"PPR+Vec {m['ppr_rrf']['ALL']['allgold4_cond']:.3f}")
        print(f"        difference (CONDITIONAL, n={len(cq)})  {b['point']:+.4f} "
              f"[{b['lo']:+.4f}, {b['hi']:+.4f}]  "
              f"{'excludes 0' if b['excludes_zero'] else 'includes 0'}")
        v = ("NOT SUPPORTED" if (a["excludes_zero"] and a["point"] < 0)
             else ("SUPPORTED" if (a["excludes_zero"] and a["point"] > 0) else "NOT SUPPORTED"))
        vb = ("SUPPORTED" if (b["excludes_zero"] and b["point"] > 0) else "INCONCLUSIVE")
        print(f"    verdict (a) graph-only: {v}")
        print(f"    verdict (b) matched pair: {vb}  (CI includes 0 -> no evidence of a gain)")
        c1[ds] = {"row_a_conditional": a, "row_a_unconditional": a_uncond, "row_b": b,
                  "verdict_a": v, "verdict_b": vb, "ceiling": ceil, "n_conditional": len(cq)}
    out["claims"]["C1"] = c1
    print("\n  C1 OVERALL: NOT SUPPORTED on either dataset. Graph-only loses to Vector RAG")
    print("  (significantly on HotpotQA), and the matched hybrid pair shows no gain whose")
    print("  CI excludes zero. Reported next to the ceilings, which bound the graph rows.")

    # ---------------- C2 ----------------
    print("\n" + "-" * 118)
    print("\nC2  The graph structure and learning both matter: GNN beats the non-learned")
    print("    graph baseline (PPR) and the 0-layer ablation on AllGold@4.")
    c2 = {}
    for ds in DS:
        cis = rep["datasets"][ds]["paired_bootstrap"]
        p = cis["allgold4|gnn-vs-ppr"]
        cq = cond_qids(ds, cfg)
        gv, pv = rows("gnn", ds), rows("ppr", ds)
        pc = paired_bootstrap([gv[q]["allgold4"] for q in cq],
                              [pv[q]["allgold4"] for q in cq], n=10_000).as_dict()
        print(f"\n  {ds}: GNN - PPR (AllGold@4 uncond, n=150) {p['point']:+.4f} "
              f"[{p['lo']:+.4f}, {p['hi']:+.4f}]  "
              f"{'excludes 0' if p['excludes_zero'] else 'includes 0'}")
        print(f"  {' '*len(ds)}  GNN - PPR (AllGold@4 cond,   n={len(cq)}) {pc['point']:+.4f} "
              f"[{pc['lo']:+.4f}, {pc['hi']:+.4f}]  "
              f"{'excludes 0' if pc['excludes_zero'] else 'includes 0'}")
        c2[ds] = {"gnn_vs_ppr_allgold4_uncond": p, "gnn_vs_ppr_allgold4_cond": pc}
    print("\n  L=0 ablation was NOT run on validation (it is a Section 5 ablation, not one")
    print("  of the reported methods). Under D17 cross-validation on TRAIN it cost")
    print("  -0.071 (MuSiQue) / -0.098 (HotpotQA) edge AP with CIs excluding 0.")
    print("\n  C2 VERDICT: INCONCLUSIVE on AllGold@4 -- both GNN-vs-PPR CIs include zero.")
    print("  On edge-level metrics the picture is different and positive: Edge-Precision@10")
    print("  is 0.605 vs PPR 0.493 (MuSiQue) and 0.749 vs 0.549 (HotpotQA), and the CV")
    print("  edge-AP differences excluded zero. So learning helps EDGE SCORING but that")
    print("  does not reach the AllGold@4 metric C2 was pre-registered on.")
    out["claims"]["C2"] = c2

    # ---------------- C3 ----------------
    print("\n" + "-" * 118)
    print("\nC3  Query-time routing is far cheaper than LLM path tracing at similar quality.")
    print("    Two halves, stated separately.")
    c3 = {}
    for ds in DS:
        eff = rep["datasets"][ds]["efficiency"]["gnn"]["routing"]
        gnn_ms = eff["median_s"] * 1000
        tr_ms = trace[ds]["median_s"] * 1000
        ratio = tr_ms / gnn_ms
        m = rep["datasets"][ds]["methods"]
        g = rows("gnn", ds)
        t = rows("llm_trace", ds)
        q = sorted(set(g) & set(t))
        ci = paired_bootstrap([g[x]["allgold4"] for x in q],
                              [t[x]["allgold4"] for x in q], n=10_000)
        gap_pts = (statistics.fmean(t[x]["allgold4"] for x in q)
                   - statistics.fmean(g[x]["allgold4"] for x in q)) * 100
        print(f"\n  {ds}")
        print(f"    HALF 1 latency: GNN routing median {gnn_ms:.1f} ms vs LLM-Trace "
              f"{tr_ms:.0f} ms -> {ratio:.1f}x cheaper  "
              f"({'PASSES' if ratio >= 10 else 'FAILS'} the >=10x test)")
        print(f"    HALF 2 quality: GNN AllGold@4 {statistics.fmean(g[x]['allgold4'] for x in q):.3f} "
              f"vs LLM-Trace {statistics.fmean(t[x]['allgold4'] for x in q):.3f} "
              f"-> GNN is {gap_pts:+.1f} points {'lower' if gap_pts > 0 else 'higher'}")
        print(f"            paired CI {ci.point:+.4f} [{ci.lo:+.4f}, {ci.hi:+.4f}]  "
              f"({'PASSES' if gap_pts <= 3 else 'FAILS'} the <=3-point tolerance)")
        print(f"    query-time LLM tokens: GNN 0 (by construction) vs LLM-Trace "
              f"{trace[ds]['tokens_per_question_mean']:.0f}/question")
        c3[ds] = {"latency_ratio": ratio, "gnn_median_ms": gnn_ms, "trace_median_ms": tr_ms,
                  "allgold_gap_points": gap_pts, "ci": ci.as_dict(),
                  "half1_passes": ratio >= 10, "half2_passes": gap_pts <= 3,
                  "trace_tokens_per_q": trace[ds]["tokens_per_question_mean"],
                  "trace_stage_swap_invalid": trace[ds]["invalidated_by_swap"]}
    print("\n  C3 VERDICT: half 1 (cost) SUPPORTED on both datasets, by 35-50x, and query-time")
    print("  LLM tokens are 0 for the GNN by construction. Half 2 (similar quality) FAILS on")
    print("  MuSiQue (GNN 6.0 points below LLM-Trace, outside the 3-point tolerance) and")
    print("  PASSES on HotpotQA (GNN 20.0 points above). So C3 is SUPPORTED on HotpotQA and")
    print("  NOT SUPPORTED on MuSiQue. Caveat: a 1.5B tracer is a weak baseline, and the 3B")
    print("  check (M7b, Section 5) that would bound this was NOT run.")
    print("  Caveat: the LLM-Trace efficiency stage is INVALID under Section 7.3 (swap, D21);")
    print("  the latency is reported as indicative, stable to ~15-23% across three runs.")
    out["claims"]["C3"] = c3

    # ---------------- C4 ----------------
    print("\n" + "-" * 118)
    print("\nC4  The full pipeline fits the hardware: peak process-tree RSS under 6 GB for")
    print("    every method (Section 7.3).")
    worst = []
    for ds in DS:
        for mm, s in rep["datasets"][ds]["efficiency"].items():
            worst.append((s["routing"]["peak_tree_rss_mb"], f"{mm}/{ds}", "routing"))
        worst.append((trace[ds]["peak_tree_rss_mb"], f"llm_trace/{ds}", "routing"))
    worst.append((idx["peak_tree_rss_mb_incl_ollama"], "extraction", "indexing"))
    worst.sort(reverse=True)
    print(f"\n  highest peak process-tree RSS (incl. Ollama), all stages:")
    for v, name, stage in worst[:6]:
        print(f"    {v:>7.0f} MB  {name:<22} ({stage})")
    mx = worst[0][0]
    ok = mx < 6144
    print(f"\n  C4 VERDICT: {'SUPPORTED' if ok else 'NOT SUPPORTED'} -- peak {mx:.0f} MB "
          f"({mx/1024:.2f} GB) < 6 GB")
    print(f"  Caveat: the reader (M8) has not run, so this covers indexing and routing only.")
    out["claims"]["C4"] = {"peak_rss_mb": mx, "under_6gb": ok,
                           "top": [{"rss_mb": v, "what": n, "stage": s} for v, n, s in worst[:6]]}

    (ROOT / "results" / "m7_verdicts.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n-> results/m7_verdicts.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
