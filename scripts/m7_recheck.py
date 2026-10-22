"""M7 follow-up: CIs computed from the saved per-question files only.

No model is run. Anything not recoverable from disk is reported as unavailable
rather than regenerated.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import paired_bootstrap  # noqa: E402
from src.config import ROOT, load_config  # noqa: E402
from src.data import load_examples, pool_size_for  # noqa: E402
from src.validation_guard import allow  # noqa: E402
from scripts.m1_data import load_pools  # noqa: E402

DS = ("musique", "hotpotqa")
MARGIN = -0.03          # C3: "AllGold@4 not lower than LLM-Trace by more than 3 points"


def rows(method, ds):
    p = ROOT / "results" / f"val_{method}_{ds}.jsonl"
    return {r["qid"]: r for r in (json.loads(l) for l in p.read_text().splitlines() if l.strip())}


def cond_qids(ds, cfg):
    pooled = {e.qid: {p.pid for p in e.paragraphs}
              for e in load_pools(ds, "validation", pool_size_for(cfg, ds))}
    orig = load_examples(ds, "validation", int(cfg["data"]["target_n"]), int(cfg["seed"]))
    return sorted(e.qid for e in orig
                  if e.gold_pids and e.gold_pids.issubset(pooled.get(e.qid, set())))


def ci_line(a, b, qs, metric, A, B):
    va = [A[q][metric] for q in qs]
    vb = [B[q][metric] for q in qs]
    ci = paired_bootstrap(va, vb, n=10_000)
    return ci, statistics.fmean(va), statistics.fmean(vb)


def main() -> int:
    cfg = load_config()
    allow("M7 follow-up CIs from saved files")
    out = {}

    # ---------------- (a) GNN vs LLM-Trace ----------------
    print("=" * 112)
    print("(a) GNN vs LLM-Trace, from results/val_*.jsonl")
    print("=" * 112)
    for ds in DS:
        G, T = rows("gnn", ds), rows("llm_trace", ds)
        allq = sorted(set(G) & set(T))
        cq = [q for q in cond_qids(ds, cfg) if q in G and q in T]
        print(f"\n  {ds}   unconditional n={len(allq)}   D4-conditional n={len(cq)}")
        out[ds] = {}
        for metric in ("allgold4", "sp_recall4"):
            for name, qs in (("uncond", allq), ("cond", cq)):
                ci, ma, mb = ci_line(None, None, qs, metric, G, T)
                out[ds][f"{metric}_{name}"] = {**ci.as_dict(), "gnn": ma, "trace": mb, "n": len(qs)}
                print(f"    {metric:<11} {name:<7} GNN {ma:.4f} vs Trace {mb:.4f}  "
                      f"diff {ci.point:+.4f} [{ci.lo:+.4f}, {ci.hi:+.4f}]  "
                      f"{'excludes 0' if ci.excludes_zero else 'includes 0'}")

        # C3 quality half, tested against the -3-point margin
        print(f"\n    C3 quality half: is (GNN - Trace) >= {MARGIN:+.2f} on AllGold@4?")
        for name in ("uncond", "cond"):
            d = out[ds][f"allgold4_{name}"]
            lo, hi, pt = d["lo"], d["hi"], d["point"]
            if lo >= MARGIN:
                verdict = "SUPPORTED (whole CI above the margin)"
            elif hi < MARGIN:
                verdict = "NOT SUPPORTED (whole CI below the margin)"
            else:
                verdict = "INCONCLUSIVE (CI straddles the margin)"
            print(f"      {name:<7} point {pt:+.4f}, CI [{lo:+.4f}, {hi:+.4f}] vs margin "
                  f"{MARGIN:+.2f} -> {verdict}")
            out[ds][f"c3_quality_{name}"] = verdict

    # ---------------- (b) GNN vs PPR / Edge-Sim ----------------
    print("\n" + "=" * 112)
    print("(b) GNN vs PPR and Edge-Sim on Edge-Precision@10")
    print("=" * 112)
    print("\n  edge AP is NOT recoverable from the saved files: per-edge scores and labels")
    print("  were not written, and the GNN's would need a model forward. Reported below is")
    print("  Edge-Precision@10, which IS saved per question. The CV edge-AP figures (train,")
    print("  already measured) are quoted for context and are not recomputed here.")
    for ds in DS:
        G, P, E = rows("gnn", ds), rows("ppr", ds), rows("edge_sim", ds)
        qs = sorted(set(G) & set(P) & set(E))
        print(f"\n  {ds}  (n={len(qs)})")
        for label, B in (("PPR", P), ("Edge-Sim", E)):
            ci, ma, mb = ci_line(None, None, qs, "edge_precision10", G, B)
            out[ds][f"edgep10_vs_{label}"] = {**ci.as_dict(), "gnn": ma, "other": mb}
            print(f"    EdgeP@10  GNN {ma:.4f} vs {label:<9} {mb:.4f}  "
                  f"diff {ci.point:+.4f} [{ci.lo:+.4f}, {ci.hi:+.4f}]  "
                  f"{'excludes 0' if ci.excludes_zero else 'includes 0'}")
        for label, B in (("PPR", P), ("Edge-Sim", E)):
            ci, ma, mb = ci_line(None, None, qs, "path_recall10", G, B)
            print(f"    PathRec   GNN {ma:.4f} vs {label:<9} {mb:.4f}  "
                  f"diff {ci.point:+.4f} [{ci.lo:+.4f}, {ci.hi:+.4f}]  "
                  f"{'excludes 0' if ci.excludes_zero else 'includes 0'}")

    (ROOT / "results" / "m7_recheck.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n-> results/m7_recheck.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
