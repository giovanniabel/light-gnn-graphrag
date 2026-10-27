"""M11 step 5: the C1-C4 verdicts on three question sets, side by side (D27).

Primary = the 350 NEW questions alone (confirmatory). Secondary = the pooled 500.
The original 150 is recomputed here only so the three columns are produced by one
code path; its published verdicts are not revised. Where a verdict differs
between columns the disagreement is printed, not smoothed.

Decision rules are the corrected ones actually used in the report:
  * C1(a)  GNN - Vector on AllGold@4, D4-conditional set. SUPPORTED if the CI
           excludes 0 and the point is positive; NOT SUPPORTED if it excludes 0
           and is negative; INCONCLUSIVE otherwise.
  * C1(b)  the D15 matched hybrid pair, GNN+Vec - PPR+Vec, same rule.
  * C2     GNN - PPR on AllGold@4 (the metric C2 was registered on), same rule.
  * C3q    GNN - LLM-Trace on AllGold@4 against the -3-point margin (D23):
           the whole CI is compared to the margin, not the point estimate.
  * C3c    LLM-Trace median latency / GNN routing median latency, >= 10x.
  * C4     peak process-tree RSS (incl. Ollama) over every measured stage < 6 GB.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import paired_bootstrap  # noqa: E402
from src.config import ROOT, load_config  # noqa: E402
from src.data import pool_size_for  # noqa: E402
from src.validation_guard import allow  # noqa: E402

DS = ("musique", "hotpotqa")
MARGIN = -0.03          # C3's pre-registered 3-point quality tolerance
N_BOOT = 10_000
SETS = ("orig150", "new350", "pooled500")


def unpaired_bootstrap(a: list[float], b: list[float], n: int = N_BOOT,
                       seed: int = 1234) -> dict:
    """Difference of means between two INDEPENDENT question sets.

    The original 150 and the new 350 are disjoint samples, so the paired
    bootstrap used for every method contrast does not apply here -- there is no
    per-question pairing to preserve. Each set is resampled on its own.
    The seed mirrors evaluation.metrics.paired_bootstrap's, so every interval in
    the project comes from the same fixed stream.
    """
    import random

    rng = random.Random(seed)
    la, lb = len(a), len(b)
    point = statistics.fmean(a) - statistics.fmean(b)
    ds = []
    for _ in range(n):
        ra = statistics.fmean(a[rng.randrange(la)] for _ in range(la))
        rb = statistics.fmean(b[rng.randrange(lb)] for _ in range(lb))
        ds.append(ra - rb)
    ds.sort()
    lo, hi = ds[int(0.025 * n)], ds[int(0.975 * n) - 1]
    return {"point": point, "lo": lo, "hi": hi, "n_a": la, "n_b": lb,
            "mean_a": statistics.fmean(a), "mean_b": statistics.fmean(b),
            "excludes_zero": (lo > 0) == (hi > 0)}


def ceiling_series(ds, cfg, split, prefix):
    """Per-question (graph reaches all gold, graph pool coverage).

    Read off the GNN rows because every graph method shares one ego-graph, so
    the ceiling is a property of the graph, not of the router.
    """
    from scripts.m7a_report import originals_for

    orig = originals_for(ds, split, cfg)
    ps = pool_size_for(cfg, ds)
    ceil, cov = [], []
    for q, r in sorted(rows("gnn", ds, prefix).items()):
        covered = set(r.get("graph_covered_pids", []))
        ceil.append(float(set(orig[q]).issubset(covered)))
        cov.append(len(covered) / ps)
    return ceil, cov



def rows(method, ds, prefix):
    p = ROOT / "results" / f"{prefix}_{method}_{ds}.jsonl"
    if not p.exists():
        return {}
    return {r["qid"]: r for r in
            (json.loads(l) for l in p.read_text().splitlines() if l.strip())}


def cond_qids(ds, cfg, split, prefix):
    """D4's conditioning set: questions whose every ORIGINAL gold paragraph
    survived pool selection. Needs the unpooled gold, so it goes through
    m7a_report.originals_for rather than the pooled examples."""
    from scripts.m1_data import load_pools
    from scripts.m7a_report import originals_for

    pooled = {e.qid: {p.pid for p in e.paragraphs}
              for e in load_pools(ds, split, pool_size_for(cfg, ds))}
    orig = originals_for(ds, split, cfg)
    return sorted(q for q, gold in orig.items()
                  if gold and gold.issubset(pooled.get(q, set())))


def verdict_zero(ci):
    if ci["excludes_zero"]:
        return "SUPPORTED" if ci["point"] > 0 else "NOT SUPPORTED"
    return "INCONCLUSIVE"


def verdict_margin(ci):
    if ci["lo"] >= MARGIN:
        return "SUPPORTED"
    if ci["hi"] < MARGIN:
        return "NOT SUPPORTED"
    return "INCONCLUSIVE"


def diff(a_rows, b_rows, qids, metric):
    qs = [q for q in qids if q in a_rows and q in b_rows]
    if not qs:
        return None
    ci = paired_bootstrap([a_rows[q][metric] for q in qs],
                          [b_rows[q][metric] for q in qs], n=N_BOOT).as_dict()
    ci["n"] = len(qs)
    ci["mean_a"] = statistics.fmean(a_rows[q][metric] for q in qs)
    ci["mean_b"] = statistics.fmean(b_rows[q][metric] for q in qs)
    return ci


def main() -> int:
    cfg = load_config()
    allow("M11 step 5: verdicts on the extension (D27)")

    # ---- per-dataset row stores, per source split ------------------------
    METHODS = ["vector", "ppr", "gnn", "gnn_rrf", "ppr_rrf", "stacked", "llm_trace"]
    store, cond, qset = {}, {}, {}
    for ds in DS:
        store[ds] = {}
        for mm in METHODS:
            o = rows(mm, ds, "val")
            n = rows(mm, ds, "ext_val")
            store[ds][mm] = {**o, **n}      # qids are disjoint by construction
            assert not (set(o) & set(n)), f"{mm}/{ds}: orig and ext qids overlap"
        co = cond_qids(ds, cfg, "validation", "val")
        cn = cond_qids(ds, cfg, "validation_ext", "ext_val")
        cond[ds] = {"orig150": co, "new350": cn, "pooled500": sorted(set(co) | set(cn))}
        qo, qn = sorted(rows("vector", ds, "val")), sorted(rows("vector", ds, "ext_val"))
        qset[ds] = {"orig150": qo, "new350": qn, "pooled500": sorted(set(qo) | set(qn))}

    out = {"margin": MARGIN, "n_bootstrap": N_BOOT, "sets": {}, "claims": {}}
    for ds in DS:
        out["sets"][ds] = {s: {"n": len(qset[ds][s]), "n_conditional": len(cond[ds][s])}
                           for s in SETS}

    # ---- C1, C2, C3 quality ---------------------------------------------
    SPECS = [
        ("C1a", "GNN - Vector, AllGold@4 (D4-conditional)", "gnn", "vector",
         "allgold4", True, verdict_zero),
        ("C1a_uncond", "GNN - Vector, AllGold@4 (unconditional)", "gnn", "vector",
         "allgold4", False, verdict_zero),
        ("C1b", "GNN+Vec - PPR+Vec, AllGold@4 (D15 matched, conditional)",
         "gnn_rrf", "ppr_rrf", "allgold4", True, verdict_zero),
        ("C2", "GNN - PPR, AllGold@4 (unconditional)", "gnn", "ppr",
         "allgold4", False, verdict_zero),
        ("C2_cond", "GNN - PPR, AllGold@4 (conditional)", "gnn", "ppr",
         "allgold4", True, verdict_zero),
        ("C3q", "GNN - LLM-Trace, AllGold@4 vs -3pt margin", "gnn", "llm_trace",
         "allgold4", False, verdict_margin),
    ]
    for key, label, a, b, metric, use_cond, rule in SPECS:
        out["claims"][key] = {"label": label, "metric": metric,
                              "conditional": use_cond, "datasets": {}}
        for ds in DS:
            per = {}
            for s in SETS:
                qs = cond[ds][s] if use_cond else qset[ds][s]
                ci = diff(store[ds][a], store[ds][b], qs, metric)
                per[s] = None if ci is None else {**ci, "verdict": rule(ci)}
            vs = {s: per[s]["verdict"] for s in SETS if per[s]}
            per["agrees"] = len(set(vs.values())) <= 1
            per["verdicts"] = vs
            out["claims"][key]["datasets"][ds] = per

    # ---- C3 cost half ----------------------------------------------------
    def eff(prefix, mm, ds):
        p = ROOT / "results" / f"{prefix}_{mm}_{ds}_stats.json"
        return json.loads(p.read_text()) if p.exists() else None

    c3c = {}
    for ds in DS:
        g_o, g_n = eff("val", "gnn", ds), eff("ext_val", "gnn", ds)
        # The original M7b ran one dataset per invocation (m7b_full_<ds>.json);
        # the extension ran both in one (m7b_ext_val_full.json). Both keep their
        # per-dataset payload under ["datasets"][ds].
        tr_o = json.loads(
            (ROOT / "results" / f"m7b_full_{ds}.json").read_text())["datasets"][ds]
        tp = ROOT / "results" / "m7b_ext_val_full.json"
        tr_n = (json.loads(tp.read_text())["datasets"].get(ds)
                if tp.exists() else None)
        e = {}
        for name, g, t in (("orig150", g_o, tr_o), ("new350", g_n, tr_n)):
            if not g or not t:
                e[name] = None
                continue
            gm = g["routing"]["median_s"] * 1000
            tm = t["median_s"] * 1000
            e[name] = {"gnn_median_ms": gm, "trace_median_ms": tm,
                       "ratio": tm / gm, "verdict": "SUPPORTED" if tm / gm >= 10
                       else "NOT SUPPORTED",
                       "gnn_stage_swap_invalid": g["routing"]["invalidated_by_swap"],
                       "trace_stage_swap_invalid": t["invalidated_by_swap"]}
        c3c[ds] = e
    out["claims"]["C3c"] = {"label": "LLM-Trace / GNN routing median latency, >=10x",
                            "datasets": c3c}

    # ---- comparability: is the new draw as easy for graph methods? -------
    # AllGold@4 is bounded above by whether the ego-graph reaches every gold
    # paragraph. If that bound differs between the two question sets, absolute
    # AllGold@4 is not comparable across them and only the within-set paired
    # contrasts above carry over. Measured, not assumed.
    comp = {}
    for ds in DS:
        co, cvo = ceiling_series(ds, cfg, "validation", "val")
        cn, cvn = ceiling_series(ds, cfg, "validation_ext", "ext_val")
        comp[ds] = {
            "allgold4_ceiling": unpaired_bootstrap(cn, co),
            "graph_pool_coverage": unpaired_bootstrap(cvn, cvo),
        }
        for k in comp[ds]:
            d = comp[ds][k]
            d["harder_on_new350"] = d["excludes_zero"] and d["point"] < 0
    out["comparability"] = comp

    # ---- reader (M8): EM/F1 on the new 350 -------------------------------
    RMETHODS = ["vector", "ppr", "llm_trace", "gnn", "gnn_rrf", "stacked", "oracle"]
    RPAIRS = [("gnn", "vector"), ("gnn", "ppr"), ("gnn", "llm_trace"),
              ("gnn_rrf", "vector"), ("stacked", "vector"), ("stacked", "gnn_rrf")]
    reader = {}
    for ds in DS:
        rd = {"orig150": {mm: rows(mm, ds, "read") for mm in RMETHODS},
              "new350": {mm: rows(mm, ds, "ext_read") for mm in RMETHODS}}
        tbl, cis = {}, {}
        for s_ in ("orig150", "new350"):
            tbl[s_] = {mm: ({"n": len(rd[s_][mm]),
                             "em": statistics.fmean(r["em"] for r in rd[s_][mm].values()),
                             "f1": statistics.fmean(r["f1"] for r in rd[s_][mm].values())}
                            if rd[s_][mm] else None)
                       for mm in RMETHODS}
            cis[s_] = {}
            for a, b in RPAIRS:
                for metric in ("em", "f1"):
                    ci = diff(rd[s_][a], rd[s_][b], sorted(rd[s_][a]), metric)
                    if ci:
                        ci["verdict"] = verdict_zero(ci)
                    cis[s_][f"{metric}|{a}-vs-{b}"] = ci
        reader[ds] = {"table": tbl, "paired_bootstrap": cis}
    out["reader"] = reader

    # ---- C4 --------------------------------------------------------------
    peaks = []
    for ds in DS:
        for mm in ["vector", "ppr", "edge_sim", "gnn", "gnn_rrf", "ppr_rrf",
                   "gnn_hybrid_tail", "ppr_hybrid_tail", "stacked"]:
            s = eff("ext_val", mm, ds)
            if s:
                peaks.append((s["routing"]["peak_tree_rss_mb"], f"{mm}/{ds}", "routing"))
        tp = ROOT / "results" / "m7b_ext_val_full.json"
        if tp.exists():
            d = json.loads(tp.read_text())["datasets"].get(ds)
            if d:
                peaks.append((d["peak_tree_rss_mb"], f"llm_trace/{ds}", "routing"))
    ext = json.loads((ROOT / "results" / "ext_extraction.json").read_text())
    peaks += [(c["peak_tree_rss_mb"], f"extraction chunk {c['chunk']}", "indexing")
              for c in ext["chunks"] if c.get("peak_tree_rss_mb")]
    peaks.sort(reverse=True)
    mx = peaks[0][0] if peaks else None
    out["claims"]["C4"] = {
        "label": "peak process-tree RSS (incl. Ollama) < 6 GB, extension stages",
        "peak_rss_mb": mx, "under_6gb": (mx is not None and mx < 6144),
        "verdict": "SUPPORTED" if (mx is not None and mx < 6144) else "NOT SUPPORTED",
        "top": [{"rss_mb": v, "what": n, "stage": s} for v, n, s in peaks[:8]]}

    # ---- print -----------------------------------------------------------
    print("=" * 112)
    print("M11 VERDICT COMPARISON -- primary column is new350 (confirmatory, D27)")
    print("=" * 112)
    for ds in DS:
        n = out["sets"][ds]
        print(f"\n{ds}:  " + "   ".join(
            f"{s} n={n[s]['n']} (cond {n[s]['n_conditional']})" for s in SETS))
    for key, label, *_ in SPECS:
        print(f"\n{'-'*112}\n{key}  {label}")
        for ds in DS:
            per = out["claims"][key]["datasets"][ds]
            print(f"  {ds}")
            for s in SETS:
                d = per[s]
                if not d:
                    print(f"    {s:<11} (no data)")
                    continue
                print(f"    {s:<11} n={d['n']:<4} {d['mean_a']:.3f} vs {d['mean_b']:.3f}  "
                      f"diff {d['point']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]  "
                      f"-> {d['verdict']}")
            print(f"    {'AGREE' if per['agrees'] else '*** DISAGREEMENT ***'}: {per['verdicts']}")
    print(f"\n{'-'*112}\nC3c  cost half")
    for ds in DS:
        for s, d in c3c[ds].items():
            if d:
                print(f"  {ds:<9} {s:<11} GNN {d['gnn_median_ms']:.1f} ms vs Trace "
                      f"{d['trace_median_ms']:.0f} ms = {d['ratio']:.1f}x -> {d['verdict']}"
                      + ("  [stage swap-INVALID]" if d["trace_stage_swap_invalid"]
                         or d["gnn_stage_swap_invalid"] else ""))
            else:
                print(f"  {ds:<9} {s:<11} (no data)")
    print(f"\n{'-'*112}\nCOMPARABILITY of the two question sets (unpaired bootstrap, new350 - orig150)")
    for ds in DS:
        for k, d in out["comparability"][ds].items():
            flag = "  *** NEW SET IS HARDER ***" if d["harder_on_new350"] else ""
            print(f"  {ds:<9} {k:<20} {d['mean_a']:.3f} vs {d['mean_b']:.3f}  "
                  f"diff {d['point']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]{flag}")

    print(f"\n{'-'*112}\nREADER EM/F1 (M8, answer quality)")
    for ds in DS:
        print(f"  {ds}")
        print(f"    {'method':<14}" + "".join(f"{s_+' EM':>12}{s_+' F1':>12}" for s_ in ("orig150", "new350")))
        for mm in RMETHODS:
            cells = ""
            for s_ in ("orig150", "new350"):
                t = reader[ds]["table"][s_][mm]
                cells += (f"{t['em']:>12.4f}{t['f1']:>12.4f}" if t else f"{'—':>12}{'—':>12}")
            print(f"    {mm:<14}{cells}")
        for a, b in RPAIRS:
            ci = reader[ds]["paired_bootstrap"]["new350"][f"f1|{a}-vs-{b}"]
            if ci:
                print(f"    F1  {a:<10} - {b:<10} {ci['point']:+.4f} "
                      f"[{ci['lo']:+.4f}, {ci['hi']:+.4f}]  -> {ci['verdict']}")

    print(f"\n{'-'*112}\nC4  peak RSS over the extension's stages")
    for d in out["claims"]["C4"]["top"]:
        # iterating the dicts directly would unpack their KEYS (three of them,
        # so the unpacking silently succeeds and prints the field names)
        print(f"    {d['rss_mb']:>7.0f} MB  {d['what']:<26} ({d['stage']})")
    print(f"  -> {out['claims']['C4']['verdict']} "
          f"(peak {mx:.0f} MB = {mx/1024:.2f} GB)" if mx else "  -> no data")

    (ROOT / "results" / "ext_pooled_report.json").write_text(json.dumps(out, indent=2,
                                                                        default=str))
    print(f"\n-> results/ext_pooled_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
