"""M7b 3B strength check: LLM-Trace with the 3B model on a fixed 60-question subset.

The subset is seeded and stratified (MuSiQue by hop count, HotpotQA by question
type), 30 per dataset, and is written to disk so the 1.5B and 3B runs are compared
on exactly the same questions.

Does NOT pull any model. If the tag is absent it says so and exits.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.baselines import llm_trace  # noqa: E402
from evaluation.metrics import all_gold, paired_bootstrap, sp_recall  # noqa: E402
from src.config import ROOT, env, load_config  # noqa: E402
from src.data import pool_size_for  # noqa: E402
from src.ollama_client import OllamaClient, OllamaError, unload_all  # noqa: E402
from src.profiling import StageProfiler, swap_used_mb  # noqa: E402
from scripts.m1_data import load_pools  # noqa: E402
from scripts.m7b_trace import build_qgs  # noqa: E402

PER_DATASET = 30
SUBSET = ROOT / "results" / "m7b_3b_subset.json"


def pick_subset(cfg) -> dict:
    """Fixed, seeded, stratified. Written once and reused by both runs."""
    if SUBSET.exists():
        return json.loads(SUBSET.read_text())
    out = {}
    for ds in ("musique", "hotpotqa"):
        exs = load_pools(ds, "validation", pool_size_for(cfg, ds))
        strata: dict[str, list[str]] = {}
        for e in exs:
            key = f"{e.n_hops}hop" if ds == "musique" else str(e.meta.get("hotpot_type"))
            strata.setdefault(key, []).append(e.qid)
        rng = random.Random(int(cfg["seed"]))
        per = max(1, PER_DATASET // len(strata))
        chosen = []
        for key in sorted(strata):
            qs = sorted(strata[key])
            rng.shuffle(qs)
            chosen += qs[:per]
        # top up deterministically if integer division left a shortfall
        if len(chosen) < PER_DATASET:
            rest = [q for key in sorted(strata) for q in sorted(strata[key])
                    if q not in chosen]
            rng.shuffle(rest)
            chosen += rest[:PER_DATASET - len(chosen)]
        out[ds] = sorted(chosen[:PER_DATASET])
    SUBSET.write_text(json.dumps(out, indent=2))
    return out


def main(tag: str, mode: str) -> int:
    cfg = load_config()
    from src.validation_guard import allow

    allow(f"M7b 3B strength check ({mode})")
    subset = pick_subset(cfg)
    print(f"  subset: " + ", ".join(f"{ds} {len(v)}" for ds, v in subset.items())
          + f"  (seeded {cfg['seed']}, stratified, fixed in "
          f"{SUBSET.relative_to(ROOT)})")

    unload_all()
    cli = OllamaClient(tag, num_ctx=int(cfg["extraction"]["num_ctx"]), temperature=0.0)
    try:
        cli.assert_model_present()
    except OllamaError as e:
        print(f"\n  MODEL NOT AVAILABLE: {e}")
        print(f"  Nothing was pulled. To run this check you would need:")
        print(f"      ollama pull {tag}")
        print(f"  Exiting without measuring.")
        return 3
    cli.generate("ready?", num_predict=1)

    out = {"tag": tag, "mode": mode, "datasets": {}}
    for ds, qids in subset.items():
        exs = [e for e in load_pools(ds, "validation", pool_size_for(cfg, ds))
               if e.qid in set(qids)]
        if mode == "test":
            exs = exs[:10]
        qgs = build_qgs(cfg, ds, exs)
        k = int(cfg["routing"]["k"])
        t = cfg["llm_trace"]
        lat, toks, rows = [], [], []
        with StageProfiler("routing_llm_trace_3b", interval_s=0.1,
                           swap_invalidate_mb=float(cfg["profiling"]["swap_invalidate_mb"])) as prof:
            for ex in exs:
                with prof.item():
                    r = llm_trace(qgs[ex.qid], ex, tag, k=k, steps=int(t["steps"]),
                                  max_candidates=int(t["max_candidate_edges"]),
                                  num_predict=int(t["num_predict"]), client=cli)
                lat.append(prof.per_item_s[-1])
                toks.append(r.stats.llm_tokens)
                rows.append({"qid": ex.qid, "dataset": ds, "method": f"llm_trace_3b",
                             "tag": tag, "ranked": r.paragraph_ids,
                             "sp_recall4": sp_recall(r, ex, k=k),
                             "allgold4": all_gold(r, ex, k=k),
                             "latency_s": lat[-1], "llm_tokens": r.stats.llm_tokens})
        d = {"n": len(rows), "median_s": statistics.median(lat),
             "tokens_per_question_mean": statistics.fmean(toks),
             "sp_recall4": statistics.fmean(r["sp_recall4"] for r in rows),
             "allgold4": statistics.fmean(r["allgold4"] for r in rows),
             "peak_tree_rss_mb": prof.stats.peak_rss_mb,
             "swap_growth_mb": prof.stats.swap_delta_mb,
             "invalidated_by_swap": prof.invalidated,
             "n_subset_total": len(qids)}
        out["datasets"][ds] = d
        if mode == "full":
            with open(ROOT / "results" / f"val_llm_trace_3b_{ds}.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            # paired against the 1.5B run on the same questions
            p15 = ROOT / "results" / f"val_llm_trace_{ds}.jsonl"
            if p15.exists():
                base = {json.loads(l)["qid"]: json.loads(l)
                        for l in p15.read_text().splitlines() if l.strip()}
                qs = [r["qid"] for r in rows if r["qid"] in base]
                for metric in ("allgold4", "sp_recall4"):
                    ci = paired_bootstrap([next(r[metric] for r in rows if r["qid"] == q)
                                           for q in qs],
                                          [base[q][metric] for q in qs], n=10_000)
                    d[f"vs_1_5b_{metric}"] = ci.as_dict()
                    print(f"    3B - 1.5B {metric:<11} {ci.point:+.4f} "
                          f"[{ci.lo:+.4f}, {ci.hi:+.4f}]  "
                          f"{'excludes 0' if ci.excludes_zero else 'includes 0'}")
        print(f"    {ds}: n={d['n']}  median {d['median_s']:.2f}s  "
              f"{d['tokens_per_question_mean']:.0f} tok/q  "
              f"SP@4 {d['sp_recall4']:.4f}  AllGold@4 {d['allgold4']:.3f}  "
              f"swap {d['swap_growth_mb']:+.0f} MB")
        if mode == "test":
            print(f"    projected for {len(qids)} subset questions: "
                  f"{d['median_s']*len(qids)/60:.1f} min")
    unload_all()
    tagsafe = tag.replace(":", "_").replace("/", "_")
    (ROOT / "results" / f"m7b_3b_{tagsafe}_{mode}.json").write_text(json.dumps(out, indent=2))
    print(f"\n-> results/m7b_3b_{tagsafe}_{mode}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else env("STRENGTH_TAG"),
                          sys.argv[2] if len(sys.argv) > 2 else "test"))
