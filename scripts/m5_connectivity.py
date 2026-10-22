"""Connectivity sweep on TRAIN/DEV only (agent.md Section 4.1.1, deviation D12).

Tries the mechanisms in the order the spec fixes -- (a) title hubs, (b) same_as
edges, (c) paragraph nodes -- and stops at the first configuration meeting the
target: >= 90% of pool paragraphs with an ego-graph edge, cap-binding <= 10%.

VALIDATION IS NEVER LOADED HERE. The chosen configuration is written to
configs/default.yaml and `python main.py m5` then reports validation once.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT, env, load_config  # noqa: E402
from src.data import carve_dev, pool_size_for  # noqa: E402
from src.embedder import EmbeddingCache, Embedder  # noqa: E402
from src.extractor import TripleCache, effective_prompt_version  # noqa: E402
from src.graph_builder import SYNTHETIC_RELATIONS, build_graph  # noqa: E402
from src.graph_stats import link_quality_diagnostic, measure, meets_target  # noqa: E402
from src.normalize import normalize_entity  # noqa: E402
from src.ollama_client import unload_all  # noqa: E402
from scripts.m1_data import load_pools  # noqa: E402

MIN_COVER = 0.90
MAX_CAP_RATE = 0.10


def ensure_embeddings(cfg, pools, emb_cache) -> int:
    """Cache normalized titles and the synthetic relation phrases."""
    need_e, need_r = set(), set(SYNTHETIC_RELATIONS)
    for exs in pools.values():
        for ex in exs:
            for p in ex.paragraphs:
                t = normalize_entity(p.title)
                if t:
                    need_e.add(t)
    todo_e = [t for t in sorted(need_e) if emb_cache.get(t, "entities") is None]
    todo_r = [t for t in sorted(need_r) if emb_cache.get(t, "relations") is None]
    if not todo_e and not todo_r:
        print("  embeddings: all titles and synthetic relations already cached")
        return 0
    emb = Embedder(env("EMBED_MODEL"), device=cfg.get("device") or "cpu",
                   cache=emb_cache, batch_size=128)
    if todo_e:
        emb.encode(todo_e, kind="entities")
    if todo_r:
        emb.encode(todo_r, kind="relations")
    emb_cache.flush()
    emb.free()
    print(f"  embeddings: encoded {len(todo_e)} titles, {len(todo_r)} synthetic relations")
    return len(todo_e) + len(todo_r)


def build_set(exs, tcache, warm, cfg, conn) -> list:
    g = cfg["graph"]
    out = []
    for ex in exs:
        triples = []
        for p in ex.paragraphs:
            triples.extend(tcache.get(p.text) or [])
        out.append(build_graph(
            ex, triples, warm, hops=int(g["hops"]),
            max_nodes=int(g["max_nodes"]), max_edges=int(g["max_edges"]),
            merge_near_duplicates=bool(g["merge_near_duplicates"]),
            merge_cosine=float(g["merge_cosine"]),
            max_seeds=int(g["max_seeds"]), seed_fallback_topk=int(g["seed_fallback_topk"]),
            connectivity=conn,
        ))
    return out


def row(label: str, st: dict) -> str:
    a = st["ALL"]
    cov = a["pool_covered_share"]
    return (f"  {label:<34} {a['components_per_question']:>6.1f} {a['bfs_node_share']:>8.1%} "
            f"{cov:>9.1%} {a['allgold_ceiling']:>9.1%} {a['nodes_mean']:>7.1f} "
            f"{a['edges_mean']:>7.1f} {a['cap_bound_rate']:>7.1%} "
            f"{a['positive_edge_rate']:>7.1%} {'MEETS' if cov >= MIN_COVER else '':>6}")


def header() -> str:
    return (f"  {'config':<34} {'comps':>6} {'bfs%':>8} {'pool_cov':>9} {'allgold':>9} "
            f"{'nodes':>7} {'edges':>7} {'cap%':>7} {'pos%':>7} {'':>6}")


def run(cfg, args) -> int:
    unload_all()
    print("--- connectivity sweep (TRAIN/DEV only -- validation is not loaded) ---")
    print(f"  target: pool coverage >= {MIN_COVER:.0%}, cap-binding <= {MAX_CAP_RATE:.0%}\n")

    datasets = list(cfg["data"]["datasets"])
    tcache = TripleCache(cfg.path("triples"), env("SLM_TAG"), effective_prompt_version(cfg))
    emb_cache = EmbeddingCache(cfg.path("embeddings"), env("EMBED_MODEL"))

    pools, devs = {}, {}
    for ds in datasets:
        ps = pool_size_for(cfg, ds)
        tr = load_pools(ds, "train", ps)
        tr_only, dev = carve_dev(tr, float(cfg["data"]["dev_frac"]), int(cfg["seed"]))
        pools[ds] = tr_only
        devs[ds] = dev
        print(f"  {ds}: {len(tr_only)} train + {len(dev)} dev (pool {ps})")

    ensure_embeddings(cfg, {**{(d, "t"): v for d, v in pools.items()},
                            **{(d, "d"): v for d, v in devs.items()}}, emb_cache)
    warm = Embedder(env("EMBED_MODEL"), device=cfg.get("device") or "cpu", cache=emb_cache)

    # candidate configurations, in the order Section 4.1.1 fixes
    candidates: list[tuple[str, dict]] = [
        ("baseline (no connectivity)", {}),
        ("(a) title hubs, exact only", {"title_hubs": True, "title_fuzzy": False}),
        ("(a) title hubs + fuzzy j=0.5", {"title_hubs": True, "title_fuzzy": True,
                                          "title_token_jaccard": 0.5}),
    ]
    for thr in (0.90, 0.88, 0.85, 0.82):
        candidates.append((
            f"(a)+(b) same_as cos>={thr:.2f}",
            {"title_hubs": True, "title_fuzzy": True, "title_token_jaccard": 0.5,
             "same_as_edges": True, "same_as_cosine": thr, "same_as_token_match": True},
        ))
    candidates.append((
        "(a)+(b)+(c) paragraph nodes",
        {"title_hubs": True, "title_fuzzy": True, "title_token_jaccard": 0.5,
         "same_as_edges": True, "same_as_cosine": 0.85, "same_as_token_match": True,
         "paragraph_nodes": True},
    ))

    results: dict = {"target": {"min_cover": MIN_COVER, "max_cap_rate": MAX_CAP_RATE},
                     "datasets": {}, "order": []}
    chosen: tuple[str, dict] | None = None

    for ds in datasets:
        ps = pool_size_for(cfg, ds)
        print(f"\n{'='*118}\n{ds.upper()} / TRAIN  (n={len(pools[ds])})\n{'='*118}")
        print(header())
        results["datasets"][ds] = {}
        for label, conn in candidates:
            t0 = time.perf_counter()
            gs = build_set(pools[ds], tcache, warm, cfg, conn)
            st = measure(gs, ps)
            results["datasets"][ds][label] = {"train": st, "seconds": time.perf_counter() - t0,
                                              "connectivity": conn}
            print(row(label, st), flush=True)

    # pick the first candidate meeting the target on train for BOTH datasets
    for label, conn in candidates:
        if all(
            meets_target(results["datasets"][ds][label]["train"]["ALL"], MIN_COVER, MAX_CAP_RATE)
            for ds in datasets
        ):
            chosen = (label, conn)
            break

    if chosen is None:
        # Rank the near-misses by BOTH target criteria, not by coverage alone: a
        # config that reaches high coverage only by saturating the node/edge caps
        # has not met the target, it has hidden the failure inside the caps.
        def _score(lc):
            per = [results["datasets"][ds][lc[0]]["train"]["ALL"] for ds in datasets]
            return (
                min(a["pool_covered_share"] for a in per),
                -max((a["cap_bound_rate"] or 0.0) for a in per),
            )

        best = max(candidates, key=_score)
        print(f"\n  !! NO configuration met the target "
              f"(coverage >= {MIN_COVER:.0%} AND cap-binding <= {MAX_CAP_RATE:.0%}) "
              f"on both datasets.")
        print(f"     nothing has been frozen and validation was NOT run.")
        for label, _c in candidates:
            per = {ds: results["datasets"][ds][label]["train"]["ALL"] for ds in datasets}
            why = ", ".join(
                f"{ds}: cover {a['pool_covered_share']:.1%}"
                + (f" / cap {a['cap_bound_rate']:.1%} OVER" if (a["cap_bound_rate"] or 0) > MAX_CAP_RATE else "")
                for ds, a in per.items()
            )
            print(f"       {label:<34} {why}")
        print(f"\n     closest on both criteria: {best[0]}")
        results["chosen"] = None
        results["closest"] = {"label": best[0], "connectivity": best[1]}
        out = ROOT / "results" / "m5_connectivity_sweep.json"
        out.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nsweep -> {out.relative_to(ROOT)}")
        print("\nNo config frozen; a decision is required before validation is measured.")
        return 2

    print(f"\n  FIRST configuration meeting the target on both datasets: {chosen[0]}")

    label, conn = chosen
    results["chosen"] = {"label": label, "connectivity": conn}

    # ---- confirm on DEV (held out from the sweep's selection) -------------
    print(f"\n{'='*118}\nDEV CONFIRMATION of {label!r}\n{'='*118}")
    print(header())
    for ds in datasets:
        ps = pool_size_for(cfg, ds)
        gs = build_set(devs[ds], tcache, warm, cfg, conn)
        st = measure(gs, ps)
        results["datasets"][ds].setdefault("_dev", {})[label] = st
        print(row(f"{ds} dev", st))

    # ---- TRAIN-ONLY link quality diagnostic -------------------------------
    print(f"\n{'='*118}\nLINK QUALITY DIAGNOSTIC (TRAIN ONLY -- descriptive, never a feature)\n{'='*118}")
    results["link_quality_train"] = {}
    for ds in datasets:
        gs = build_set(pools[ds], tcache, warm, cfg, conn)
        diag = link_quality_diagnostic(gs)
        results["link_quality_train"][ds] = diag
        sh = diag["shares"]
        print(f"\n  {ds}: {diag['cross_paragraph_pairs']:,} cross-paragraph paragraph-pairs added")
        for k in ("gold_gold", "gold_distractor", "distractor_distractor"):
            v = sh.get(k)
            print(f"      {k:<24} {v:>7.1%}" if v is not None else f"      {k:<24}     n/a")
        for kind, c in sorted(diag["by_kind"].items()):
            s2 = c["shares"]
            print(f"      by kind {kind:<10} ({c['pairs']:>6,} pairs): "
                  f"g-g {s2['gold_gold']:.1%}, g-d {s2['gold_distractor']:.1%}, "
                  f"d-d {s2['distractor_distractor']:.1%}")

    out = ROOT / "results" / "m5_connectivity_sweep.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nsweep -> {out.relative_to(ROOT)}")
    print(f"\nCHOSEN (frozen): {label}")
    print(f"  {json.dumps(conn)}")
    print("\nWrite it into configs/default.yaml, then run `python main.py m5` once for validation.")
    return 0


if __name__ == "__main__":
    from src.config import load_config as _lc

    class _A:
        n = None
        dataset = None
        resume = True
    raise SystemExit(run(_lc(), _A()))
