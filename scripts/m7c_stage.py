"""One M7c efficiency stage in its own process (driven by run_m7c_efficiency.sh)."""

from __future__ import annotations

import gc
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from evaluation.baselines import llm_trace, vector_rag  # noqa: E402
from src.config import ROOT, env, load_config  # noqa: E402
from src.data import pool_size_for  # noqa: E402
from src.embedder import EmbeddingCache, Embedder  # noqa: E402
from src.extractor import TripleCache, effective_prompt_version  # noqa: E402
from src.graph_builder import build_graph, to_pyg  # noqa: E402
from src.ollama_client import OllamaClient, unload_all  # noqa: E402
from src.profiling import StageProfiler, swap_used_mb, tree_rss_mb  # noqa: E402
from src.routing import rank_paragraphs  # noqa: E402
from scripts.m1_data import load_pools  # noqa: E402

N_TEST = 10


def q95(v):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(0.95 * (len(v) - 1))))] if v else None


def main(stage: str, ds: str, mode: str) -> int:
    cfg = load_config()
    g = cfg["graph"]
    hops = int(g["hops"])
    k = int(cfg["routing"]["k"])
    exs = load_pools(ds, "validation", pool_size_for(cfg, ds))
    if mode == "test":
        exs = exs[:N_TEST]
    n_total = len(load_pools(ds, "validation", pool_size_for(cfg, ds)))

    tc = TripleCache(cfg.path("triples"), env("SLM_TAG"), effective_prompt_version(cfg))
    ec = EmbeddingCache(cfg.path("embeddings"), env("EMBED_MODEL"))
    emb = Embedder(env("EMBED_MODEL"), device=cfg.get("device") or "cpu", cache=ec)

    print(f"  swap at start {swap_used_mb():.0f} MB, tree RSS {tree_rss_mb():.0f} MB")
    qgs, pygs = {}, {}
    for ex in exs:
        tri = []
        for p in ex.paragraphs:
            tri.extend(tc.get(p.text) or [])
        qg = build_graph(ex, tri, emb, hops=hops, max_nodes=int(g["max_nodes"]),
                         max_edges=int(g["max_edges"]),
                         merge_near_duplicates=bool(g["merge_near_duplicates"]),
                         merge_cosine=float(g["merge_cosine"]),
                         max_seeds=int(g["max_seeds"]),
                         seed_fallback_topk=int(g["seed_fallback_topk"]),
                         connectivity=dict(g.get("connectivity") or {}))
        qgs[ex.qid] = qg
        if stage == "routing":
            pygs[ex.qid] = to_pyg(qg, ec, hops)

    models = []
    if stage == "routing":
        from src.gnn_router import load_router

        models = [load_router(ROOT / "results" / "checkpoints_final" /
                              f"router_{ds}_final_L2_seed{s}.pt", cfg)
                  for s in [int(x) for x in cfg["seeds"]]]
        cos = {ex.qid: vector_rag(ex, emb, k=k) for ex in exs}   # warm vector path

    # release the encoder and cache BEFORE Ollama loads (the D21 lesson)
    emb.free()
    del emb, ec, tc
    gc.collect()
    print(f"  encoder and cache released; swap {swap_used_mb():.0f} MB, "
          f"tree RSS {tree_rss_mb():.0f} MB")

    cli = None
    if stage == "llm_trace":
        cli = OllamaClient(env("SLM_TAG"), num_ctx=int(cfg["extraction"]["num_ctx"]),
                           temperature=0.0)
        cli.assert_model_present()
        cli.generate("ready?", num_predict=1)
        ps = subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout
        print(f"  ollama ps: {[l.split()[0] for l in ps.strip().splitlines()[1:] if l.strip()]}"
              f"  swap {swap_used_mb():.0f} MB")

    lat, toks = [], []
    swap0 = swap_used_mb()
    with StageProfiler(stage, interval_s=float(cfg["profiling"]["rss_interval_s"]),
                       swap_invalidate_mb=float(cfg["profiling"]["swap_invalidate_mb"])) as prof:
        for ex in exs:
            qg = qgs[ex.qid]
            with prof.item():
                if stage == "llm_trace":
                    r = llm_trace(qg, ex, env("SLM_TAG"), k=k,
                                  steps=int(cfg["llm_trace"]["steps"]),
                                  max_candidates=int(cfg["llm_trace"]["max_candidate_edges"]),
                                  num_predict=int(cfg["llm_trace"]["num_predict"]), client=cli)
                    toks.append(r.stats.llm_tokens)
                else:
                    with torch.no_grad():
                        per = [m(pygs[ex.qid]).cpu().tolist() for m in models]
                    sc = [statistics.fmean(v) for v in zip(*per)]
                    rank_paragraphs(qg, sc, [p.pid for p in ex.paragraphs], k=k,
                                    mode=str(cfg["routing"]["paragraph_score"]),
                                    top2=int(cfg["routing"]["top2"]))
                    toks.append(0)
            lat.append(prof.per_item_s[-1])
    s = prof.stats
    if cli:
        unload_all()

    out = {
        "stage": stage, "dataset": ds, "mode": mode, "n": len(lat),
        "n_validation_total": n_total,
        "median_s": statistics.median(lat), "p95_s": q95(lat),
        "mean_s": statistics.fmean(lat),
        "tokens_per_question_mean": statistics.fmean(toks),
        "peak_tree_rss_mb": s.peak_rss_mb, "idle_baseline_rss_mb": s.baseline_rss_mb,
        "delta_rss_mb": s.delta_rss_mb,
        "swap_start_mb": swap0, "swap_end_mb": swap_used_mb(),
        "swap_growth_mb": s.swap_delta_mb,
        "invalidated_by_swap": prof.invalidated,
        "projected_full_min": statistics.median(lat) * n_total / 60,
    }
    (ROOT / "results" / f"m7c_{stage}_{ds}_{mode}.json").write_text(json.dumps(out, indent=2))
    print(f"    n={out['n']}  median {out['median_s']*1000:.1f} ms  p95 {out['p95_s']*1000:.1f} ms"
          f"  {out['tokens_per_question_mean']:.0f} tok/q")
    print(f"    peak tree RSS {out['peak_tree_rss_mb']:.0f} MB (delta {out['delta_rss_mb']:+.0f}), "
          f"swap growth {out['swap_growth_mb']:+.0f} MB -> "
          f"{'INVALID under §7.3' if prof.invalidated else 'valid'}")
    if mode == "test":
        print(f"    projected for all {n_total} questions: "
              f"{out['projected_full_min']:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "test"))
