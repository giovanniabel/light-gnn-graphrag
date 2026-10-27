"""M7a: score ONE method on ONE validation dataset, in its own process.

Invoked by scripts/m7a_run.py as a subprocess so each method gets a fresh
interpreter and its own idle RSS baseline, as the efficiency protocol requires.

Writes results/val_<method>_<dataset>.jsonl (one row per question, so every metric
is recomputable without rerunning) plus a _stats.json with the stage timings.

Reading the validation split requires ALLOW_VALIDATION; the parent sets it.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from evaluation.baselines import ppr_rag, vector_cosines, vector_rag  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    all_gold,
    edge_precision,
    path_recall,
    sp_recall,
)
from src.config import ROOT, env, load_config  # noqa: E402
from src.data import pool_size_for  # noqa: E402
from src.embedder import EmbeddingCache, Embedder  # noqa: E402
from src.extractor import TripleCache, effective_prompt_version  # noqa: E402
from src.graph_builder import build_graph, is_yes_no, to_pyg  # noqa: E402
from src.profiling import StageProfiler, swap_used_mb, tree_rss_mb  # noqa: E402
from src.routing import fuse_with_vector, rank_paragraphs  # noqa: E402
from scripts.m1_data import load_pools  # noqa: E402

GRAPH_METHODS = {"ppr", "edge_sim", "gnn", "gnn_rrf", "ppr_rrf",
                 "gnn_hybrid_tail", "ppr_hybrid_tail", "stacked"}
NEEDS_GNN = {"gnn", "gnn_rrf", "gnn_hybrid_tail", "stacked"}


def load_final_models(cfg, ds):
    from src.gnn_router import load_router

    out = []
    for seed in [int(s) for s in cfg["seeds"]]:
        p = ROOT / "results" / "checkpoints_final" / f"router_{ds}_final_L2_seed{seed}.pt"
        out.append(load_router(p, cfg, cfg.get("device") or "cpu"))
    return out


def main(method: str, ds: str, split: str = "validation",
         prefix: str = "val") -> int:
    """`split`/`prefix` default to the M7a one-time run. The M11 extension
    (D27) passes split="validation_ext", prefix="ext_val" so the extension is
    measured by this same code -- same metrics, same timed region, same
    Edge-Sim preload -- rather than by a parallel copy that could drift."""
    cfg = load_config()
    k = int(cfg["routing"]["k"])
    hops = int(cfg["graph"]["hops"])
    g = cfg["graph"]
    idle_rss = tree_rss_mb()          # this process's own idle baseline
    swap0 = swap_used_mb()

    exs = load_pools(ds, split, pool_size_for(cfg, ds))
    tc = TripleCache(cfg.path("triples"), env("SLM_TAG"), effective_prompt_version(cfg))
    ec = EmbeddingCache(cfg.path("embeddings"), env("EMBED_MODEL"))
    emb = Embedder(env("EMBED_MODEL"), device=cfg.get("device") or "cpu", cache=ec)

    models = load_final_models(cfg, ds) if method in NEEDS_GNN else []
    stacker = None
    if method == "stacked":
        import pickle

        with open(ROOT / "results" / "checkpoints_final" / f"stacker_{ds}.pkl", "rb") as f:
            stacker = pickle.load(f)["model"]

    # Edge-Sim reads ~400 edge-text embeddings per question. Served from the
    # sharded on-disk cache inside the timed loop, the first touch of each shard
    # pays a torch.load, so the measured latency was dominated by cache I/O rather
    # than by the method. Preload every edge text ONCE, before the timed stage, so
    # the loop hits resident memory and the number reflects Edge-Sim itself.
    if method == "edge_sim":
        texts = set()
        for ex in exs:
            tri = []
            for p in ex.paragraphs:
                tri.extend(tc.get(p.text) or [])
            qgp = build_graph(
                ex, tri, emb, hops=hops, max_nodes=int(g["max_nodes"]),
                max_edges=int(g["max_edges"]),
                merge_near_duplicates=bool(g["merge_near_duplicates"]),
                merge_cosine=float(g["merge_cosine"]), max_seeds=int(g["max_seeds"]),
                seed_fallback_topk=int(g["seed_fallback_topk"]),
                connectivity=dict(g.get("connectivity") or {}))
            for (u, v), rel in zip(qgp.edges, qgp.edge_rel):
                texts.add(f"{qgp.nodes[u]} {rel} {qgp.nodes[v]}")
        emb.encode(sorted(texts), kind="edges")
        print(f"  [edge_sim] preloaded {len(texts)} edge-text embeddings before timing")

    rows = []
    t_assembly, t_forward, t_route = [], [], []

    with StageProfiler("routing", interval_s=float(cfg["profiling"]["rss_interval_s"]),
                       swap_invalidate_mb=float(cfg["profiling"]["swap_invalidate_mb"])) as prof:
        for ex in exs:
            with prof.item():
                # ---- graph assembly (graph methods only) -----------------
                qg = pyg = None
                ta = 0.0
                if method in GRAPH_METHODS:
                    t0 = time.perf_counter()
                    tri = []
                    for p in ex.paragraphs:
                        tri.extend(tc.get(p.text) or [])
                    qg = build_graph(
                        ex, tri, emb, hops=hops, max_nodes=int(g["max_nodes"]),
                        max_edges=int(g["max_edges"]),
                        merge_near_duplicates=bool(g["merge_near_duplicates"]),
                        merge_cosine=float(g["merge_cosine"]),
                        max_seeds=int(g["max_seeds"]),
                        seed_fallback_topk=int(g["seed_fallback_topk"]),
                        connectivity=dict(g.get("connectivity") or {}),
                    )
                    if method in NEEDS_GNN:
                        pyg = to_pyg(qg, ec, hops)
                    ta = time.perf_counter() - t0

                # ---- model forward / scoring -----------------------------
                t1 = time.perf_counter()
                pool = [p.pid for p in ex.paragraphs]
                edge_scores = None

                if method == "vector":
                    ranked = vector_rag(ex, emb, k=k).paragraph_ids
                elif method in ("ppr", "ppr_rrf", "ppr_hybrid_tail"):
                    r = ppr_rag(qg, ex, k=k)
                    edge_scores = [0.0] * len(qg.edge_pid)
                    for i, v in r.edges:
                        edge_scores[i] = v
                    base = r.paragraph_ids
                    if method == "ppr_rrf":
                        ranked = fuse_with_vector(base, vector_rag(ex, emb, k=k).paragraph_ids, pool)
                    elif method == "ppr_hybrid_tail":
                        ranked = rank_paragraphs(qg, edge_scores, pool, k=k, variant="hybrid",
                                                 cosines=vector_cosines(ex, emb))
                    else:
                        ranked = base
                elif method == "edge_sim":
                    from evaluation.baselines import edge_sim

                    r = edge_sim(qg, emb, ex, k=k)
                    edge_scores = [0.0] * len(qg.edge_pid)
                    for i, v in r.edges:
                        edge_scores[i] = v
                    ranked = r.paragraph_ids
                else:  # the GNN family
                    with torch.no_grad():
                        per_seed = [m(pyg).cpu().tolist() for m in models]
                    edge_scores = [statistics.fmean(v) for v in zip(*per_seed)]
                    gnn_ranked = rank_paragraphs(
                        qg, edge_scores, pool, k=k,
                        mode=str(cfg["routing"]["paragraph_score"]),
                        top2=int(cfg["routing"]["top2"]))
                    if method == "gnn":
                        ranked = gnn_ranked
                    elif method == "gnn_rrf":
                        ranked = fuse_with_vector(
                            gnn_ranked, vector_rag(ex, emb, k=k).paragraph_ids, pool)
                    elif method == "gnn_hybrid_tail":
                        ranked = rank_paragraphs(qg, edge_scores, pool, k=k, variant="hybrid",
                                                 cosines=vector_cosines(ex, emb))
                    else:  # stacked
                        from scripts.m6_stack import _norm
                        from src.routing import paragraph_scores
                        import numpy as np

                        vec = _norm(vector_cosines(ex, emb), pool)
                        gnn = _norm(paragraph_scores(
                            qg, edge_scores, mode=str(cfg["routing"]["paragraph_score"]),
                            top2=int(cfg["routing"]["top2"])), pool)
                        pr = ppr_rag(qg, ex, k=k)
                        pr_sc = [0.0] * len(qg.edge_pid)
                        for i, v in pr.edges:
                            pr_sc[i] = v
                        ppr_n = _norm(paragraph_scores(qg, pr_sc), pool)
                        X = np.array([[vec[p], gnn[p], ppr_n[p]] for p in pool])
                        sc = stacker.predict_proba(X)[:, 1]
                        ranked = [pool[i] for i in sorted(range(len(pool)), key=lambda i: -sc[i])]
                tf = time.perf_counter() - t1

            t_assembly.append(ta)
            t_forward.append(tf)
            t_route.append(prof.per_item_s[-1])

            row = {
                "qid": ex.qid, "dataset": ds, "method": method, "split": split,
                "n_hops": ex.n_hops, "hotpot_type": ex.meta.get("hotpot_type"),
                "answer": ex.answer, "yes_no": is_yes_no(ex.answer),
                "pool": pool, "gold": sorted(ex.gold_pids),
                "ranked": ranked, "top_k": ranked[:k],
                "sp_recall4": sp_recall(ranked, ex, k=k),
                "allgold4": all_gold(ranked, ex, k=k),
                "latency_routing_s": t_route[-1],
                "latency_graph_assembly_s": ta,
                "latency_model_forward_s": tf,
                "llm_tokens": 0,
            }
            if qg is not None and edge_scores is not None:
                class _R:
                    edges = sorted(enumerate(edge_scores), key=lambda t: -t[1])
                    paragraph_ids = ranked
                row["edge_precision10"] = edge_precision(_R(), qg, b=int(cfg["routing"]["edge_b"]))
                row["path_recall10"] = path_recall(_R(), qg, b=int(cfg["routing"]["edge_b"]))
                row["answer_in_graph"] = qg.answer_in_graph(True)
                row["graph_covered_pids"] = sorted(qg.covered_pids())
                row["n_nodes"] = qg.n_nodes
                row["n_edges"] = qg.n_edges
            rows.append(row)

    s = prof.stats
    out = ROOT / "results" / f"{prefix}_{method}_{ds}.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    def q(v, p):
        v = sorted(v)
        return v[min(len(v) - 1, int(round(p * (len(v) - 1))))] if v else None

    stats = {
        "method": method, "dataset": ds, "split": split, "n_questions": len(rows),
        "idle_baseline_rss_mb": idle_rss,
        "routing": {
            "median_s": statistics.median(t_route), "p95_s": q(t_route, 0.95),
            "mean_s": statistics.fmean(t_route),
            "llm_tokens": 0,
            "peak_tree_rss_mb": s.peak_rss_mb,
            "delta_rss_mb_over_idle": s.peak_rss_mb - idle_rss,
            "swap_growth_mb": s.swap_delta_mb,
            "invalidated_by_swap": prof.invalidated,
        },
        "routing_breakdown": {
            "graph_assembly_median_s": statistics.median(t_assembly) if any(t_assembly) else 0.0,
            "graph_assembly_p95_s": q(t_assembly, 0.95) if any(t_assembly) else 0.0,
            "model_forward_median_s": statistics.median(t_forward),
            "model_forward_p95_s": q(t_forward, 0.95),
        },
        "swap_start_mb": swap0, "swap_end_mb": swap_used_mb(),
    }
    (ROOT / "results" / f"{prefix}_{method}_{ds}_stats.json").write_text(
        json.dumps(stats, indent=2))
    print(f"  {method:<18} {ds:<9} n={len(rows)}  "
          f"SP@4 {statistics.fmean(r['sp_recall4'] for r in rows):.4f}  "
          f"AllGold@4 {statistics.fmean(r['allgold4'] for r in rows):.3f}  "
          f"route med {stats['routing']['median_s']*1000:.1f}ms  "
          f"peakRSS {s.peak_rss_mb:.0f}MB  swap {s.swap_delta_mb:+.0f}MB"
          + ("  !! SWAP-INVALID" if prof.invalidated else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:5]))
