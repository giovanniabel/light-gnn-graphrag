"""M7a aggregation: metric tables, ceilings and paired-bootstrap CIs from the
per-question JSONL files. Recomputes everything from disk, so no method is rerun.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import paired_bootstrap  # noqa: E402
from src.config import ROOT, env, load_config  # noqa: E402
from src.data import load_examples, pool_size_for  # noqa: E402
from src.extractor import TripleCache, effective_prompt_version  # noqa: E402
from src.validation_guard import allow  # noqa: E402
from scripts.m1_data import load_pools  # noqa: E402

METHODS = ["vector", "ppr", "edge_sim", "gnn", "gnn_rrf", "ppr_rrf",
           "gnn_hybrid_tail", "ppr_hybrid_tail", "stacked"]
LABEL = {"vector": "Vector RAG", "ppr": "PPR", "edge_sim": "Edge-Sim",
         "gnn": "GNN (graph-only)", "gnn_rrf": "GNN+Vector (RRF)",
         "ppr_rrf": "PPR+Vector (RRF)", "gnn_hybrid_tail": "GNN+hybrid-tail",
         "ppr_hybrid_tail": "PPR+hybrid-tail", "stacked": "Stacked (D19)"}
PAIRS = [("gnn", "vector"), ("gnn", "ppr"), ("gnn", "edge_sim"),
         ("gnn_rrf", "ppr_rrf"), ("stacked", "vector"), ("stacked", "gnn_rrf")]


def rows_of(method, ds):
    p = ROOT / "results" / f"val_{method}_{ds}.jsonl"
    return {r["qid"]: r for r in (json.loads(l) for l in p.read_text().splitlines() if l.strip())}


def bucket(r, ds):
    return f"{r['n_hops']}hop" if ds == "musique" else (r["hotpot_type"] or "unknown")


def m(v):
    return statistics.fmean(v) if v else None


def main() -> int:
    allow("M7 one-time validation run (aggregation)")
    cfg = load_config()
    tc = TripleCache(cfg.path("triples"), env("SLM_TAG"), effective_prompt_version(cfg))
    report = {"datasets": {}}

    for ds in ("musique", "hotpotqa"):
        ps = pool_size_for(cfg, ds)
        pooled = load_pools(ds, "validation", ps)
        orig = {e.qid: e.gold_pids
                for e in load_examples(ds, "validation", int(cfg["data"]["target_n"]),
                                       int(cfg["seed"]))}
        data = {mm: rows_of(mm, ds) for mm in METHODS}
        qids = sorted(data["vector"])
        by_b = {}
        for q in qids:
            by_b.setdefault(bucket(data["vector"][q], ds), []).append(q)
        buckets = {**by_b, "ALL": qids}

        # ---- context: pool recall, extraction coverage, ceilings ---------
        ctxrows = {}
        for bname, bq in buckets.items():
            gold_tot = gold_pool = 0
            gold_tri = gold_with = 0
            ingraph = cov_all = 0
            scorable = 0
            ag_ceil = 0
            for q in bq:
                ex = next(e for e in pooled if e.qid == q)
                r = data["gnn"][q]
                go = set(orig[q])
                gold_tot += len(go)
                gold_pool += len(go & {p.pid for p in ex.paragraphs})
                for p in ex.paragraphs:
                    if p.is_gold:
                        gold_tri += 1
                        if tc.get(p.text):
                            gold_with += 1
                covered = set(r.get("graph_covered_pids", []))
                cov_all += len(covered) / ps
                ag_ceil += go.issubset(covered)
                if not r["yes_no"]:
                    scorable += 1
                    ingraph += bool(r.get("answer_in_graph"))
            n = len(bq)
            ctxrows[bname] = {
                "n": n,
                "pool_recall": gold_pool / gold_tot if gold_tot else None,
                "extraction_coverage": gold_with / gold_tri if gold_tri else None,
                "graph_pool_coverage": cov_all / n,
                "allgold4_ceiling": ag_ceil / n,
                "answer_in_graph_ceiling": ingraph / scorable if scorable else None,
                "n_scorable_nonyesno": scorable,
            }

        # ---- per-method metrics -----------------------------------------
        mrows = {}
        for mm in METHODS:
            d = data[mm]
            mrows[mm] = {}
            for bname, bq in buckets.items():
                cond_q = [q for q in bq if set(orig[q]).issubset({p for p in d[q]["pool"]})]
                mrows[mm][bname] = {
                    "n": len(bq),
                    "sp_recall4": m([d[q]["sp_recall4"] for q in bq]),
                    "allgold4": m([d[q]["allgold4"] for q in bq]),
                    "allgold4_cond": m([d[q]["allgold4"] for q in cond_q]),
                    "n_cond": len(cond_q),
                    "edge_precision10": m([d[q]["edge_precision10"] for q in bq
                                           if "edge_precision10" in d[q]]),
                    "path_recall10": m([d[q]["path_recall10"] for q in bq
                                        if "path_recall10" in d[q]]),
                }

        # ---- paired bootstrap on ALL -------------------------------------
        cis = {}
        for a, b in PAIRS:
            for metric in ("allgold4", "sp_recall4"):
                va = [data[a][q][metric] for q in qids]
                vb = [data[b][q][metric] for q in qids]
                cis[f"{metric}|{a}-vs-{b}"] = paired_bootstrap(va, vb, n=10_000).as_dict()

        stats = {mm: json.loads((ROOT / "results" / f"val_{mm}_{ds}_stats.json").read_text())
                 for mm in METHODS}
        report["datasets"][ds] = {"context": ctxrows, "methods": mrows,
                                  "paired_bootstrap": cis, "efficiency": stats}

        # ---- print -------------------------------------------------------
        print(f"\n{'='*118}\n{ds.upper()} / VALIDATION  (n={len(qids)})\n{'='*118}")
        c = ctxrows["ALL"]
        print(f"  context: pool recall {c['pool_recall']:.3f} | extraction coverage "
              f"{c['extraction_coverage']:.3f} | graph pool coverage {c['graph_pool_coverage']:.3f}")
        print(f"  CEILINGS: AllGold@4 (graph methods) {c['allgold4_ceiling']:.3f} | "
              f"answer-in-graph {c['answer_in_graph_ceiling']:.3f} "
              f"(n={c['n_scorable_nonyesno']} non-yes/no)")
        print(f"\n  {'method':<20} {'SP@4':>7} {'AllGold@4':>10} {'cond':>7} {'n_cond':>7} "
              f"{'EdgeP@10':>9} {'PathRec':>8}")
        for mm in METHODS:
            r = mrows[mm]["ALL"]
            ep = f"{r['edge_precision10']:.4f}" if r["edge_precision10"] is not None else "—"
            pr = f"{r['path_recall10']:.4f}" if r["path_recall10"] is not None else "—"
            print(f"  {LABEL[mm]:<20} {r['sp_recall4']:>7.4f} {r['allgold4']:>10.3f} "
                  f"{r['allgold4_cond']:>7.3f} {r['n_cond']:>7} {ep:>9} {pr:>8}")

        print(f"\n  per bucket (AllGold@4 unconditional; ceiling in brackets)")
        bnames = [b for b in buckets if b != "ALL"]
        hdr = "".join(f"{b+' (n=' + str(ctxrows[b]['n']) + ')':>18}" for b in bnames)
        print(f"  {'method':<20}{hdr}")
        for mm in METHODS:
            cells = "".join(f"{mrows[mm][b]['allgold4']:>18.3f}" for b in bnames)
            print(f"  {LABEL[mm]:<20}{cells}")
        cl = "".join(f"{'[' + format(ctxrows[b]['allgold4_ceiling'], '.3f') + ']':>18}"
                     for b in bnames)
        print(f"  {'CEILING (graph)':<20}{cl}")

        print(f"\n  paired bootstrap, 10,000 resamples (difference, 95% CI)")
        for a, b in PAIRS:
            for metric in ("allgold4",):
                ci = cis[f"{metric}|{a}-vs-{b}"]
                v = "excludes 0" if ci["excludes_zero"] else "INCLUDES 0 (within noise)"
                print(f"    {LABEL[a]:<20} - {LABEL[b]:<20} {ci['point']:+.4f} "
                      f"[{ci['lo']:+.4f}, {ci['hi']:+.4f}]  {v}")

        print(f"\n  efficiency, routing stage (median / p95 ms; peak tree RSS MB; swap MB)")
        print(f"  {'method':<20} {'median':>8} {'p95':>8} {'assembly':>9} {'forward':>8} "
              f"{'peakRSS':>8} {'dRSS':>7} {'swap':>6} {'tokens':>7}")
        for mm in METHODS:
            s = stats[mm]["routing"]
            b = stats[mm]["routing_breakdown"]
            print(f"  {LABEL[mm]:<20} {s['median_s']*1000:>8.1f} {s['p95_s']*1000:>8.1f} "
                  f"{b['graph_assembly_median_s']*1000:>9.1f} "
                  f"{b['model_forward_median_s']*1000:>8.2f} "
                  f"{s['peak_tree_rss_mb']:>8.0f} {s['delta_rss_mb_over_idle']:>7.0f} "
                  f"{s['swap_growth_mb']:>+6.0f} {s['llm_tokens']:>7}")
        bad = [mm for mm in METHODS if stats[mm]["routing"]["invalidated_by_swap"]]
        print(f"  stages invalidated by swap (>500 MB): {bad or 'none'}")

    (ROOT / "results" / "m7a_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\n-> results/m7a_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
