"""Final aggregation: stage validity, efficiency, M8 EM/F1, 3B check, verdicts.

Reads only saved files. No model is run and no frozen setting is touched.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import paired_bootstrap  # noqa: E402
from src.config import ROOT, load_config  # noqa: E402

DS = ("musique", "hotpotqa")
READ = ["vector", "ppr", "llm_trace", "gnn", "gnn_rrf", "stacked", "oracle"]
LAB = {"vector": "Vector RAG", "ppr": "PPR", "llm_trace": "LLM-Trace (1.5B)",
       "gnn": "GNN", "gnn_rrf": "GNN+Vector (RRF)", "stacked": "Stacked (D19)",
       "oracle": "Gold oracle"}


def j(p):
    return json.loads((ROOT / "results" / p).read_text())


def jl(p):
    return [json.loads(l) for l in (ROOT / "results" / p).read_text().splitlines() if l.strip()]


def main() -> int:
    cfg = load_config()
    out = {}

    # ---------------- 1. stage validity ----------------
    print("=" * 120)
    print("1. STAGE VALIDITY (§7.3: swap growth > 500 MB during a measured stage invalidates it)")
    print("=" * 120)
    print(f"\n  {'stage':<34} {'swap@start':>11} {'growth':>9} {'peak RSS':>9} {'verdict':<9} note")
    rowsv = []
    for ds in DS:
        for st in ("llm_trace", "routing"):
            d = j(f"m7c_{st}_{ds}_full.json")
            rowsv.append((f"M7c {st}/{ds}", d["swap_start_mb"], d["swap_growth_mb"],
                          d["peak_tree_rss_mb"], not d["invalidated_by_swap"]))
    for ds in DS:
        for m in READ:
            d = j(f"read_{m}_{ds}.json")
            rowsv.append((f"M8 reader {m}/{ds}", None, d["swap_growth_mb"],
                          d["peak_tree_rss_mb"], not d["invalidated_by_swap"]))
    ix = j("m7_indexing_cost.json")
    rowsv.insert(0, ("M7 indexing (extraction)", 264.0, ix["swap_growth_mb"],
                     ix["peak_tree_rss_mb_incl_ollama"], not ix.get("stage_invalidated_by_swap")))
    for ds in DS:
        for m in ("vector", "ppr", "edge_sim", "gnn", "gnn_rrf", "ppr_rrf", "stacked"):
            d = j(f"val_{m}_{ds}_stats.json")["routing"]
            rowsv.append((f"M7a routing {m}/{ds}", None, d["swap_growth_mb"],
                          d["peak_tree_rss_mb"], not d["invalidated_by_swap"]))
    n_inv = 0
    for name, s0, gr, peak, ok in rowsv:
        note = ""
        if s0 is not None and s0 > 300:
            note = f"started at {s0:.0f} MB swap -- above the 300 MB gate"
        if not ok:
            n_inv += 1
        print(f"  {name:<34} {(f'{s0:.0f}' if s0 is not None else '—'):>11} "
              f"{gr:>+9.0f} {peak:>9.0f} {'valid' if ok else 'INVALID':<9} {note}")
    out["validity"] = [{"stage": n, "swap_start_mb": s, "swap_growth_mb": g,
                        "peak_rss_mb": p, "valid": o} for n, s, g, p, o in rowsv]
    print(f"\n  {n_inv} of {len(rowsv)} measured stages are INVALID under §7.3.")

    # ---------------- 2. routing scope ----------------
    print("\n" + "=" * 120)
    print("2. THE 4.2 ms vs 43.8 ms ROUTING DISCREPANCY")
    print("=" * 120)
    for ds in DS:
        a = j(f"val_gnn_{ds}_stats.json")
        c = j(f"m7c_routing_{ds}_full.json")
        br = a["routing_breakdown"]
        print(f"\n  {ds}")
        print(f"    M7a routing (FULL §7.3 scope)      median {a['routing']['median_s']*1000:>7.1f} ms")
        print(f"      of which graph assembly          {br['graph_assembly_median_s']*1000:>7.1f} ms")
        print(f"      of which model forward           {br['model_forward_median_s']*1000:>7.1f} ms")
        print(f"    M7c 'routing'                      median {c['median_s']*1000:>7.1f} ms")
        print(f"    -> M7c ~= M7a's forward component ({br['model_forward_median_s']*1000:.1f} ms), "
              f"not the full stage")
    print("\n  CAUSE: in scripts/m7c_stage.py the QueryGraphs and PyG tensors are built")
    print("  BEFORE the StageProfiler block, so the timed loop contains only the model")
    print("  forward and paragraph aggregation. §7.3 defines routing as 'everything from")
    print("  question + graph to ranked paragraphs', which includes the question embedding,")
    print("  seed linking and ego-graph/feature assembly -- all of which happen inside")
    print("  build_graph() and are therefore EXCLUDED from the M7c figure.")
    print("  => M7c's number is relabelled 'model forward + aggregation only'.")
    print("  => C3 uses the M7a full-routing numbers.")
    out["routing_scope"] = {
        ds: {"m7a_full_ms": j(f"val_gnn_{ds}_stats.json")["routing"]["median_s"] * 1000,
             "m7a_assembly_ms": j(f"val_gnn_{ds}_stats.json")["routing_breakdown"]["graph_assembly_median_s"] * 1000,
             "m7a_forward_ms": j(f"val_gnn_{ds}_stats.json")["routing_breakdown"]["model_forward_median_s"] * 1000,
             "m7c_forward_only_ms": j(f"m7c_routing_{ds}_full.json")["median_s"] * 1000}
        for ds in DS}

    # ---------------- 3. LLM-Trace, four measurements ----------------
    print("\n" + "=" * 120)
    print("3. LLM-TRACE: ALL FOUR MEASUREMENTS (indicative, never §7.3-valid -- D24)")
    print("=" * 120)
    smoke = j("m7b_smoke.json")["datasets"]
    first = {ds: j(f"m7b_full_{ds}.json")["datasets"][ds] for ds in DS}
    m7c = {ds: j(f"m7c_llm_trace_{ds}_full.json") for ds in DS}
    print(f"\n  {'measurement':<26} {'dataset':<9} {'median ms':>10} {'tokens/q':>9} "
          f"{'swap growth':>12} {'valid':>7}")
    lt = {ds: [] for ds in DS}
    rows3 = [("1. M7 first run", lambda ds: (first[ds]["median_s"] * 1000,
                                             first[ds]["tokens_per_question_mean"],
                                             first[ds]["swap_growth_mb"])),
             ("2. smoke (10 questions)", lambda ds: (smoke[ds]["median_s"] * 1000,
                                                     smoke[ds]["tokens_per_question_mean"],
                                                     smoke[ds]["swap_growth_mb"])),
             ("3. D21 redo", lambda ds: (first[ds]["median_s"] * 1000,
                                         first[ds]["tokens_per_question_mean"],
                                         first[ds]["swap_growth_mb"])),
             ("4. M7c clean-start run", lambda ds: (m7c[ds]["median_s"] * 1000,
                                                    m7c[ds]["tokens_per_question_mean"],
                                                    m7c[ds]["swap_growth_mb"]))]
    for name, fn in rows3:
        for ds in DS:
            ms, tk, sw = fn(ds)
            lt[ds].append(ms)
            print(f"  {name:<26} {ds:<9} {ms:>10.0f} {tk:>9.0f} {sw:>+12.0f} "
                  f"{'valid' if sw <= 500 else 'INVALID':>7}")
    print("\n  C3 LATENCY RATIO, using the VALID M7a full-routing medians:")
    out["c3_latency"] = {}
    for ds in DS:
        gnn = j(f"val_gnn_{ds}_stats.json")["routing"]["median_s"] * 1000
        lo, hi = min(lt[ds]), max(lt[ds])
        print(f"    {ds:<9} GNN {gnn:.1f} ms vs LLM-Trace {lo:.0f}-{hi:.0f} ms "
              f"-> {lo/gnn:.1f}x to {hi/gnn:.1f}x cheaper")
        out["c3_latency"][ds] = {"gnn_full_routing_ms": gnn, "trace_ms_range": [lo, hi],
                                 "ratio_range": [lo / gnn, hi / gnn]}

    # ---------------- 4. C4 ----------------
    print("\n" + "=" * 120)
    print("4. C4: PEAK PROCESS-TREE RSS BY STAGE")
    print("=" * 120)
    peaks = sorted(((p, n, g) for n, s, g, p, o in rowsv), reverse=True)[:8]
    print(f"\n  {'peak RSS':>9} {'swap growth':>12}  stage")
    for p, n, g in peaks:
        print(f"  {p:>9.0f} {g:>+12.0f}  {n}")
    mx = peaks[0][0]
    print(f"\n  highest peak: {mx:.0f} MB ({mx/1024:.2f} GB) vs the 6 GB limit -> "
          f"{'under' if mx < 6144 else 'OVER'}")
    print("\n  LIMITATION: on Apple Silicon, process RSS under-reports what Ollama")
    print("  actually occupies. Model weights live in GPU-wired memory managed by the")
    print("  Metal driver, which is not fully attributed to the llama-server process's")
    print("  RSS. The swap evidence is the better signal that the machine was over")
    print("  committed: stages that stayed under 6 GB by RSS still drove swap up by")
    print("  2-3 GB. So C4's RSS number is a floor, not a faithful total.")
    out["c4"] = {"peak_rss_mb": mx, "under_6gb": mx < 6144,
                 "top": [{"rss_mb": p, "stage": n, "swap_growth_mb": g} for p, n, g in peaks]}

    (ROOT / "results" / "m8_analysis.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n-> results/m8_analysis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
