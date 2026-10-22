"""M8 EM/F1 tables, paired CIs, 3B check and the re-verdict. Saved files only."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import paired_bootstrap  # noqa: E402
from src.config import ROOT  # noqa: E402

DS = ("musique", "hotpotqa")
READ = ["vector", "ppr", "llm_trace", "gnn", "gnn_rrf", "stacked", "oracle"]
LAB = {"vector": "Vector RAG", "ppr": "PPR", "llm_trace": "LLM-Trace 1.5B",
       "gnn": "GNN", "gnn_rrf": "GNN+Vector RRF", "stacked": "Stacked D19",
       "oracle": "Gold oracle"}
# pre-registered in agent.md §7.2 / §5: every method against Vector RAG and the oracle
PREREG = {("vector", "oracle")} | {(m, "vector") for m in READ if m != "vector"} \
         | {(m, "oracle") for m in READ if m != "oracle"}


def jl(p):
    return {r["qid"]: r for r in
            (json.loads(l) for l in (ROOT / "results" / p).read_text().splitlines() if l.strip())}


def bucket(r, ds):
    return f"{r['n_hops']}hop" if ds == "musique" else (r["hotpot_type"] or "?")


def main() -> int:
    out = {"comparisons": [], "datasets": {}}
    n_comp = 0

    for ds in DS:
        data = {m: jl(f"read_{m}_{ds}.jsonl") for m in READ}
        qs = sorted(set.intersection(*[set(v) for v in data.values()]))
        print("=" * 118)
        print(f"5. M8 READER EM/F1 -- {ds.upper()} (n={len(qs)})")
        print("=" * 118)
        bs = sorted({bucket(data["vector"][q], ds) for q in qs})
        print(f"\n  {'condition':<18} {'EM':>7} {'F1':>7}   " +
              "".join(f"{b+' EM':>11}{b+' F1':>11}" for b in bs))
        rows = {}
        for m in READ:
            d = data[m]
            em = statistics.fmean(d[q]["em"] for q in qs)
            f1 = statistics.fmean(d[q]["f1"] for q in qs)
            cells = ""
            for b in bs:
                bq = [q for q in qs if bucket(d[q], ds) == b]
                cells += (f"{statistics.fmean(d[q]['em'] for q in bq):>11.3f}"
                          f"{statistics.fmean(d[q]['f1'] for q in bq):>11.3f}")
            rows[m] = {"em": em, "f1": f1}
            print(f"  {LAB[m]:<18} {em:>7.3f} {f1:>7.3f}   {cells}")
        print(f"\n  bucket n: " + ", ".join(
            f"{b}={sum(1 for q in qs if bucket(data['vector'][q], ds)==b)}" for b in bs))

        # oracle sanity check
        orc, vec = rows["oracle"], rows["vector"]
        ok = orc["f1"] >= vec["f1"] and orc["em"] >= vec["em"]
        print(f"\n  ORACLE SANITY CHECK: oracle EM {orc['em']:.3f} / F1 {orc['f1']:.3f} vs "
              f"Vector RAG {vec['em']:.3f} / {vec['f1']:.3f} -> "
              f"{'PASSED (oracle is the ceiling)' if ok else 'FAILED'}")

        pairs = [("gnn", "vector"), ("gnn", "oracle"), ("gnn", "ppr"),
                 ("gnn", "llm_trace"), ("gnn_rrf", "gnn"), ("stacked", "gnn_rrf"),
                 ("vector", "oracle"), ("stacked", "vector")]
        print(f"\n  paired bootstrap, 10,000 resamples (difference, 95% CI)")
        print(f"  {'comparison':<34} {'metric':<4} {'diff':>9} {'95% CI':>22} "
              f"{'verdict':<13} {'registered?'}")
        cis = {}
        for a, b in pairs:
            for metric in ("em", "f1"):
                ci = paired_bootstrap([data[a][q][metric] for q in qs],
                                      [data[b][q][metric] for q in qs], n=10_000)
                n_comp += 1
                reg = "pre-registered" if (a, b) in PREREG or (b, a) in PREREG else "POST-HOC"
                border = ""
                if abs(ci.lo) < 1e-9 or abs(ci.hi) < 1e-9:
                    border = "  <-- BORDERLINE (a bound is exactly 0.000)"
                v = "excludes 0" if ci.excludes_zero else "includes 0"
                cis[f"{metric}|{a}-vs-{b}"] = {**ci.as_dict(), "registered": reg}
                print(f"  {LAB[a]+' - '+LAB[b]:<34} {metric.upper():<4} {ci.point:>+9.4f} "
                      f"[{ci.lo:+.4f}, {ci.hi:+.4f}]".rjust(0) +
                      f"  {v:<13} {reg}{border}")
        out["datasets"][ds] = {"rows": rows, "cis": cis, "n": len(qs),
                               "oracle_check_passed": ok}
        print()

    # ---------------- 6. 3B check ----------------
    print("=" * 118)
    print("6. 3B STRENGTH CHECK (fixed, seeded, stratified 60-question subset)")
    print("=" * 118)
    three = json.loads((ROOT / "results" /
                        "m7b_3b_qwen2.5_3b-instruct-q4_K_M_full.json").read_text())
    subset = json.loads((ROOT / "results" / "m7b_3b_subset.json").read_text())
    out["three_b"] = {}
    for ds in DS:
        sub = set(subset[ds])
        t3 = jl(f"val_llm_trace_3b_{ds}.jsonl")
        t15 = jl(f"val_llm_trace_{ds}.jsonl")
        gnn = jl(f"val_gnn_{ds}.jsonl")
        qs = sorted(sub & set(t3) & set(t15) & set(gnn))
        d = three["datasets"][ds]
        print(f"\n  {ds}  (n={len(qs)} of the {len(sub)}-question subset)")
        print(f"    3B   median {d['median_s']:.2f}s, {d['tokens_per_question_mean']:.0f} tok/q, "
              f"swap {d['swap_growth_mb']:+.0f} MB "
              f"({'valid' if not d['invalidated_by_swap'] else 'INVALID'})")
        res = {}
        for metric in ("allgold4", "sp_recall4"):
            for a, b, nm in ((t3, t15, "3B - 1.5B"), (gnn, t3, "GNN - 3B"),
                             (gnn, t15, "GNN - 1.5B")):
                ci = paired_bootstrap([a[q][metric] for q in qs],
                                      [b[q][metric] for q in qs], n=10_000)
                n_comp += 1
                res[f"{metric}|{nm}"] = ci.as_dict()
                print(f"    {metric:<11} {nm:<11} {ci.point:+.4f} "
                      f"[{ci.lo:+.4f}, {ci.hi:+.4f}]  "
                      f"{'excludes 0' if ci.excludes_zero else 'includes 0'}  POST-HOC")
        out["three_b"][ds] = {"n": len(qs), "summary": d, "cis": res}
    print(f"\n  n=30 per dataset. With AllGold@4 taking 31 values on 30 questions, the")
    print(f"  CIs here are wide by construction; treat these as directional only.")

    out["n_comparisons"] = n_comp
    print(f"\n  TOTAL paired comparisons run in this analysis: {n_comp}")
    print(f"  NONE are corrected for multiple comparisons. At 95% confidence, ~{n_comp*0.05:.0f}")
    print(f"  false positives are expected by chance alone, so single CIs that only just")
    print(f"  exclude zero should not be read as findings.")
    (ROOT / "results" / "m8_emf1.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n-> results/m8_emf1.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
