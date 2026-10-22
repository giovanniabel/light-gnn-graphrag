"""M7b: LLM-Trace on validation, Ollama ON. Resumable, per-question outputs.

Usage: m7b_trace.py smoke | full
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.baselines import llm_trace, trace_candidates  # noqa: E402
from evaluation.metrics import all_gold, sp_recall  # noqa: E402
from src.config import ROOT, env, load_config  # noqa: E402
from src.data import pool_size_for  # noqa: E402
from src.embedder import EmbeddingCache, Embedder  # noqa: E402
from src.extractor import TripleCache, effective_prompt_version  # noqa: E402
from src.graph_builder import KIND_TRIPLE, build_graph, is_yes_no  # noqa: E402
from src.ollama_client import OllamaClient, unload_all  # noqa: E402
from src.profiling import StageProfiler, swap_used_mb, tree_rss_mb  # noqa: E402
from src.validation_guard import allow  # noqa: E402
from scripts.m1_data import load_pools  # noqa: E402

DATASETS = ("musique", "hotpotqa")


def build_qgs(cfg, ds, exs):
    g = cfg["graph"]
    tc = TripleCache(cfg.path("triples"), env("SLM_TAG"), effective_prompt_version(cfg))
    ec = EmbeddingCache(cfg.path("embeddings"), env("EMBED_MODEL"))
    emb = Embedder(env("EMBED_MODEL"), device=cfg.get("device") or "cpu", cache=ec)
    out = {}
    for ex in exs:
        tri = []
        for p in ex.paragraphs:
            tri.extend(tc.get(p.text) or [])
        out[ex.qid] = build_graph(
            ex, tri, emb, hops=int(g["hops"]), max_nodes=int(g["max_nodes"]),
            max_edges=int(g["max_edges"]),
            merge_near_duplicates=bool(g["merge_near_duplicates"]),
            merge_cosine=float(g["merge_cosine"]), max_seeds=int(g["max_seeds"]),
            seed_fallback_topk=int(g["seed_fallback_topk"]),
            connectivity=dict(g.get("connectivity") or {}))
    emb.free()
    return out


def verify_filter(qgs) -> dict:
    """Prove on real validation graphs that no scaffolding edge is ever offered."""
    offered = scaffold = 0
    for qg in list(qgs.values())[:30]:
        cand = trace_candidates(qg, set(qg.seeds), set(), 20)
        offered += len(cand)
        kinds = qg.edge_kind or []
        scaffold += sum(1 for i in cand if kinds[i] != KIND_TRIPLE)
    return {"candidate_edges_offered": offered, "scaffolding_offered": scaffold}


def run_one(ds, exs, qgs, cfg, cli, limit=None, done=None):
    lat, toks, calls, rows = [], [], [], []
    k = int(cfg["routing"]["k"])
    tcfg = cfg["llm_trace"]
    todo = [e for e in exs if not done or e.qid not in done]
    if limit:
        todo = todo[:limit]
    with StageProfiler("routing_llm_trace",
                       interval_s=float(cfg["profiling"]["rss_interval_s"]),
                       swap_invalidate_mb=float(cfg["profiling"]["swap_invalidate_mb"])) as prof:
        for i, ex in enumerate(todo, 1):
            qg = qgs[ex.qid]
            with prof.item():
                r = llm_trace(qg, ex, env("SLM_TAG"), k=k, steps=int(tcfg["steps"]),
                              max_candidates=int(tcfg["max_candidate_edges"]),
                              num_predict=int(tcfg["num_predict"]), client=cli)
            lat.append(prof.per_item_s[-1])
            toks.append(r.stats.llm_tokens)
            calls.append(getattr(r, "n_calls", 0))
            rows.append({
                "qid": ex.qid, "dataset": ds, "method": "llm_trace",
                "n_hops": ex.n_hops, "hotpot_type": ex.meta.get("hotpot_type"),
                "yes_no": is_yes_no(ex.answer),
                "pool": [p.pid for p in ex.paragraphs], "gold": sorted(ex.gold_pids),
                "ranked": r.paragraph_ids, "top_k": r.paragraph_ids[:k],
                "sp_recall4": sp_recall(r, ex, k=k), "allgold4": all_gold(r, ex, k=k),
                "latency_routing_s": lat[-1], "llm_tokens": r.stats.llm_tokens,
                "prompt_tokens": getattr(r, "prompt_tokens", 0),
                "eval_tokens": getattr(r, "eval_tokens", 0),
                "n_calls": getattr(r, "n_calls", 0),
            })
            if i % 25 == 0:
                print(f"    {ds} {i}/{len(todo)}  {statistics.fmean(lat):.2f}s/q  "
                      f"{statistics.fmean(toks):.0f} tok/q", flush=True)
    return rows, lat, toks, calls, prof


def main(mode: str, only_ds: str | None = None) -> int:
    import gc

    cfg = load_config()
    allow(f"M7b LLM-Trace ({mode})")
    unload_all()
    print(f"--- M7b: LLM-Trace, Ollama ON ({mode}){f' [{only_ds}]' if only_ds else ''} ---")
    print(f"  swap at start: {swap_used_mb():.0f} MB")
    print(f"  schema output, num_predict={cfg['llm_trace']['num_predict']}, "
          f"steps<={cfg['llm_trace']['steps']}, candidates<={cfg['llm_trace']['max_candidate_edges']}")

    out = {"mode": mode, "datasets": {}}
    for ds in ([only_ds] if only_ds else list(DATASETS)):
        exs = load_pools(ds, "validation", pool_size_for(cfg, ds))
        # Build graphs FIRST, then release the encoder and cache, THEN load Ollama.
        # The first attempt held 300 graphs + bge + cache + Ollama at once and drove
        # the machine 3-4 GB into swap, voiding the stage under Section 7.3.
        qgs = build_qgs(cfg, ds, exs)
        gc.collect()
        print(f"  graphs built and encoder released; swap {swap_used_mb():.0f} MB, "
              f"tree RSS {tree_rss_mb():.0f} MB")
        cli = OllamaClient(env("SLM_TAG"), num_ctx=int(cfg["extraction"]["num_ctx"]),
                           temperature=0.0)
        cli.assert_model_present()
        cli.generate("ready?", num_predict=1)
        ps = subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout
        print(f"  ollama ps: {[l.split()[0] for l in ps.strip().splitlines()[1:] if l.strip()]}"
              f"  swap {swap_used_mb():.0f} MB")
        vf = verify_filter(qgs)
        print(f"\n  {ds}: filter check -> {vf['candidate_edges_offered']} candidate edges "
              f"offered across 30 graphs, {vf['scaffolding_offered']} of them scaffolding "
              f"({'triple-only CONFIRMED' if vf['scaffolding_offered'] == 0 else 'LEAK'})")

        path = ROOT / "results" / f"val_llm_trace_{ds}.jsonl"
        done = set()
        if mode == "full" and path.exists():
            done = {json.loads(l)["qid"] for l in path.read_text().splitlines() if l.strip()}
            print(f"    resuming: {len(done)} already done")

        rows, lat, toks, calls, prof = run_one(
            ds, exs, qgs, cfg, cli, limit=10 if mode == "smoke" else None, done=done)

        if mode == "full":
            with open(path, "a") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")

        s = prof.stats
        d = {
            "n": len(rows), "median_s": statistics.median(lat), "mean_s": statistics.fmean(lat),
            "p95_s": sorted(lat)[min(len(lat) - 1, int(round(0.95 * (len(lat) - 1))))],
            "tokens_per_question_mean": statistics.fmean(toks),
            "llm_calls_per_question_mean": statistics.fmean(calls),
            "peak_tree_rss_mb": s.peak_rss_mb, "idle_baseline_rss_mb": s.baseline_rss_mb,
            "delta_rss_mb": s.delta_rss_mb, "swap_growth_mb": s.swap_delta_mb,
            "invalidated_by_swap": prof.invalidated,
            "filter_check": vf,
            "sp_recall4": statistics.fmean(r["sp_recall4"] for r in rows),
            "allgold4": statistics.fmean(r["allgold4"] for r in rows),
            "n_validation_total": len(exs),
        }
        out["datasets"][ds] = d
        print(f"    n={d['n']}  median {d['median_s']:.2f}s  p95 {d['p95_s']:.2f}s  "
              f"{d['tokens_per_question_mean']:.0f} tok/q  "
              f"{d['llm_calls_per_question_mean']:.2f} calls/q")
        print(f"    SP@4 {d['sp_recall4']:.4f}  AllGold@4 {d['allgold4']:.3f}")
        print(f"    peak tree RSS {d['peak_tree_rss_mb']:.0f} MB (delta {d['delta_rss_mb']:+.0f}), "
              f"swap {d['swap_growth_mb']:+.0f} MB"
              + ("  !! SWAP-INVALID" if prof.invalidated else ""))

    if mode == "smoke":
        tot = sum(out["datasets"][d]["median_s"] * out["datasets"][d]["n_validation_total"]
                  for d in out["datasets"])
        print(f"\n  EXTRAPOLATION to all validation questions:")
        for d in out["datasets"]:
            x = out["datasets"][d]
            print(f"    {d}: {x['n_validation_total']} questions x {x['median_s']:.2f}s "
                  f"= {x['median_s']*x['n_validation_total']/60:.1f} min")
        print(f"    TOTAL ~{tot/60:.1f} min   (gate: stop and ask if > 60 min)")
        out["projected_total_min"] = tot / 60
        out["gate_exceeds_60min"] = tot / 60 > 60
    unload_all()
    print(f"\n  model unloaded")
    tag = f"{mode}_{only_ds}" if only_ds else mode
    (ROOT / "results" / f"m7b_{tag}.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"-> results/m7b_{tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "smoke",
                          sys.argv[2] if len(sys.argv) > 2 else None))
