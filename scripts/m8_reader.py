"""M8: one reader condition on one dataset, in its own process. Resumable.

Contexts come from the SAVED M7 routings (results/val_<method>_<dataset>.jsonl), so
no router is re-run; `oracle` uses the gold paragraphs as the reader's ceiling.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import em_f1, paired_bootstrap  # noqa: E402
from src.config import ROOT, env, load_config  # noqa: E402
from src.data import pool_size_for  # noqa: E402
from src.generator import MAX_TOKENS, answer  # noqa: E402
from src.ollama_client import OllamaClient, unload_all  # noqa: E402
from src.profiling import StageProfiler, swap_used_mb  # noqa: E402
from scripts.m1_data import load_pools  # noqa: E402

N_TEST = 10
METHODS = ["vector", "ppr", "llm_trace", "gnn", "gnn_rrf", "stacked", "oracle"]
DS = ["musique", "hotpotqa"]


def out_path(m, ds, prefix="read"):
    return ROOT / "results" / f"{prefix}_{m}_{ds}.jsonl"


def summarize_estimate():
    rows = []
    for ds in DS:
        for m in METHODS:
            p = ROOT / "results" / f"read_{m}_{ds}_test.json"
            if p.exists():
                rows.append(json.loads(p.read_text()))
    if not rows:
        print("  no test files found")
        return
    total = sum(r["median_s"] * r["n_validation_total"] for r in rows)
    print(f"\n=== M8 ESTIMATED TOTAL from the 10-question tests ===")
    print(f"  {'condition':<14} {'dataset':<9} {'median_s':>9} {'x n':>5} {'minutes':>9}")
    for r in rows:
        print(f"  {r['method']:<14} {r['dataset']:<9} {r['median_s']:>9.2f} "
              f"{r['n_validation_total']:>5} {r['median_s']*r['n_validation_total']/60:>9.1f}")
    print(f"  {'TOTAL':<24} {'':>9} {'':>5} {total/60:>9.1f} min")
    if total / 60 > 120:
        print(f"\n  !! {total/60:.0f} min exceeds the 2-hour budget. Reductions, in the order")
        print(f"     I would prefer them:")
        print(f"       1. drop `gnn_rrf` and `stacked` (their routings differ from `gnn` on a")
        print(f"          minority of questions, and EM/F1 from a 1.5B reader is the least")
        print(f"          discriminative metric in the project) -> saves ~{2*sum(r['median_s']*r['n_validation_total'] for r in rows if r['method'] in ('gnn_rrf','stacked'))/60/2:.0f} min")
        print(f"       2. subsample to 100 questions per dataset -> saves ~1/3")
        print(f"       3. keep all conditions but MuSiQue only, since that is where the")
        print(f"          retrieval differences are largest")
    else:
        print(f"  within the 2-hour budget.")


def report():
    print("\n=== M8 EM/F1 ===")
    for ds in DS:
        base = {}
        print(f"\n  {ds}")
        print(f"    {'condition':<14} {'n':>5} {'EM':>8} {'F1':>8}")
        per_q = {}
        for m in METHODS:
            p = out_path(m, ds)
            if not p.exists():
                continue
            rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
            per_q[m] = {r["qid"]: r for r in rows}
            print(f"    {m:<14} {len(rows):>5} {statistics.fmean(r['em'] for r in rows):>8.4f} "
                  f"{statistics.fmean(r['f1'] for r in rows):>8.4f}")
        for a in per_q:
            for b in ("vector", "oracle"):
                if a == b or b not in per_q:
                    continue
                qs = sorted(set(per_q[a]) & set(per_q[b]))
                for metric in ("em", "f1"):
                    ci = paired_bootstrap([per_q[a][q][metric] for q in qs],
                                          [per_q[b][q][metric] for q in qs], n=10_000)
                    print(f"      {a:<14} - {b:<8} {metric.upper():<3} {ci.point:+.4f} "
                          f"[{ci.lo:+.4f}, {ci.hi:+.4f}]  "
                          f"{'excludes 0' if ci.excludes_zero else 'includes 0'}")


def main(method: str, ds: str, mode: str, split: str = "validation",
         prefix: str = "read", src_prefix: str = "val") -> int:
    """`split`/`prefix`/`src_prefix` default to the M8 one-time run; the M11
    extension (D27) passes "validation_ext"/"ext_read"/"ext_val"."""
    cfg = load_config()
    k = int(cfg["routing"]["k"])
    exs = {e.qid: e for e in load_pools(ds, split, pool_size_for(cfg, ds))}
    n_total = len(exs)

    if method == "oracle":
        contexts = {q: sorted(e.gold_pids) for q, e in exs.items()}
    else:
        src = ROOT / "results" / f"{src_prefix}_{method}_{ds}.jsonl"
        contexts = {r["qid"]: r["ranked"][:k]
                    for r in (json.loads(l) for l in src.read_text().splitlines() if l.strip())}

    path = out_path(method, ds, prefix)
    done = set()
    if mode == "full" and path.exists():
        done = {json.loads(l)["qid"] for l in path.read_text().splitlines() if l.strip()}
        print(f"  resuming: {len(done)} already done")

    todo = [q for q in sorted(contexts) if q not in done]
    if mode == "test":
        todo = todo[:N_TEST]

    cli = OllamaClient(env("SLM_TAG"), num_ctx=int(cfg["extraction"]["num_ctx"]),
                       temperature=0.0)
    cli.assert_model_present()
    cli.generate("ready?", num_predict=1)

    lat, toks, rows = [], [], []
    with StageProfiler("reading", interval_s=float(cfg["profiling"]["rss_interval_s"]),
                       swap_invalidate_mb=float(cfg["profiling"]["swap_invalidate_mb"])) as prof:
        for q in todo:
            ex = exs[q]
            paras = [p for pid in contexts[q] for p in ex.paragraphs if p.pid == pid]
            with prof.item():
                pred, usage = answer(ex.question, paras, client=cli, max_tokens=MAX_TOKENS)
            lat.append(prof.per_item_s[-1])
            toks.append(usage.total)
            golds = [ex.answer] + list(ex.meta.get("answer_aliases") or [])
            em, f1 = em_f1(pred, golds)
            rows.append({"qid": q, "dataset": ds, "method": method, "pred": pred,
                         "gold": ex.answer, "em": em, "f1": f1,
                         "n_paragraphs": len(paras), "latency_s": lat[-1],
                         "llm_tokens": usage.total,
                         "n_hops": ex.n_hops, "hotpot_type": ex.meta.get("hotpot_type")})
    # A fully-resumed run measures nothing, so every statistic below would be
    # computed over an empty list -- which used to surface as a bare
    # StatisticsError from statistics.median, i.e. the same failure shape as
    # D28: the run looks finished, exits non-zero somewhere a grep can miss, and
    # writes no report. Say what happened and what to do instead. Rebuilding the
    # stats from the existing rows is deliberately NOT done: their latencies come
    # from a different process on a differently loaded machine, and mixing
    # timings across runs is exactly what the Section 7.3 rules exist to prevent.
    if not rows:
        raise SystemExit(
            f"{method}/{ds}/{split}: all {len(done)} questions were already in "
            f"{path.name}, so nothing was measured and no report can be written. "
            "Delete that file (and its _stats.json) to re-read and re-measure, "
            "or leave them as they are if the existing run still stands."
        )

    s = prof.stats
    if mode == "full":
        with open(path, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    res = {"method": method, "dataset": ds, "split": split, "mode": mode,
           "n": len(rows),
           "n_validation_total": n_total,
           "median_s": statistics.median(lat), "mean_s": statistics.fmean(lat),
           "p95_s": sorted(lat)[min(len(lat)-1, int(round(0.95*(len(lat)-1))))],
           "tokens_per_question_mean": statistics.fmean(toks),
           "em": statistics.fmean(r["em"] for r in rows),
           "f1": statistics.fmean(r["f1"] for r in rows),
           "peak_tree_rss_mb": s.peak_rss_mb, "delta_rss_mb": s.delta_rss_mb,
           "swap_growth_mb": s.swap_delta_mb, "invalidated_by_swap": prof.invalidated}
    tag = f"_{mode}" if mode == "test" else ""
    (ROOT / "results" / f"{prefix}_{method}_{ds}{tag}.json").write_text(
        json.dumps(res, indent=2))
    print(f"    n={res['n']}  EM {res['em']:.4f}  F1 {res['f1']:.4f}  "
          f"median {res['median_s']:.2f}s  {res['tokens_per_question_mean']:.0f} tok/q  "
          f"peakRSS {res['peak_tree_rss_mb']:.0f}MB  swap {res['swap_growth_mb']:+.0f}MB"
          + ("  !! SWAP-INVALID" if prof.invalidated else ""))
    if mode == "test":
        print(f"    projected for {n_total} questions: "
              f"{res['median_s']*n_total/60:.1f} min")
    return 0


if __name__ == "__main__":
    if sys.argv[1] == "--summarize-estimate":
        summarize_estimate()
    elif sys.argv[1] == "--report":
        report()
    else:
        raise SystemExit(main(sys.argv[1], sys.argv[2],
                              sys.argv[3] if len(sys.argv) > 3 else "test",
                              *sys.argv[4:7]))
