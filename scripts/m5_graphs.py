"""M5: graph building, labels and ceilings (agent.md Section 11).

Acceptance: graph ceiling, seed-hit rate, # dropped train questions logged; PyG
Data shapes validated on 10 graphs.

Ceiling conventions (requested, recorded as D12):
  * yes/no answers are excluded from the answer-in-graph and Path-Recall
    ceiling denominators -- 'yes' is never an entity node, so scoring it as a
    miss understates the ceiling. The excluded count is reported.
  * ceilings are reported both with and without MuSiQue answer_aliases.

No training and no Ollama: this phase loads bge to cache entity and relation
embeddings, then frees it (Section 6.1).
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src.config import ROOT, env, load_config  # noqa: E402
from src.data import load_examples, pool_size_for  # noqa: E402
from src.embedder import EmbeddingCache, Embedder  # noqa: E402
from src.extractor import TripleCache, effective_prompt_version  # noqa: E402
from src.graph_stats import link_quality_diagnostic, measure  # noqa: E402
from src.graph_builder import (  # noqa: E402
    EDGE_FEAT_DIM,
    NODE_FEAT_DIM,
    SEED_FALLBACK,
    SEED_NONE,
    SEED_SUBSTRING,
    build_graph,
    is_yes_no,
    to_pyg,
)
from src.normalize import normalize_entity  # noqa: E402
from src.ollama_client import unload_all  # noqa: E402
from src.profiling import own_tree_rss_mb  # noqa: E402
from scripts.m1_data import load_pools  # noqa: E402


def _bucket(qg) -> str:
    return f"{qg.n_hops}hop" if qg.dataset == "musique" else "2hop"


def _frac(a: int, b: int):
    return (a / b) if b else None


def validate_shapes(graphs: list, emb_cache, hops: int, k: int = 10) -> dict:
    """Section 11: validate PyG Data shapes on 10 graphs, and print them."""
    print(f"\n--- PyG Data shape validation on {k} graphs ---")
    print(f"  expected: x[N, {NODE_FEAT_DIM}]  edge_index[2, E]  edge_attr[E, {EDGE_FEAT_DIM}]  "
          f"y_edge[E]  q[1, 384]  seed_mask[N]  evidence_mask[E]")
    rows, ok = [], True
    for qg in graphs[:k]:
        d = to_pyg(qg, emb_cache, hops)
        N, E = d.x.shape[0], d.edge_index.shape[1]
        checks = {
            "x dim": d.x.shape == (N, NODE_FEAT_DIM),
            "edge_index": tuple(d.edge_index.shape) == (2, E),
            "edge_attr": tuple(d.edge_attr.shape) == (E, EDGE_FEAT_DIM),
            "y_edge": tuple(d.y_edge.shape) == (E,),
            "q": tuple(d.q.shape) == (1, 384),
            "seed_mask": tuple(d.seed_mask.shape) == (N,),
            "edge_pid len": len(d.edge_pid) == E,
            "edge_kind len": len(d.edge_kind) == E,
            "evidence_mask": tuple(d.evidence_mask.shape) == (E,),
            "no scaffolding positive": (E == 0) or bool(
                (d.y_edge[~d.evidence_mask] == 0).all()
            ),
            "N matches nodes": N == qg.n_nodes,
            "E matches edges": E == qg.n_edges,
            "index in range": (E == 0) or (int(d.edge_index.max()) < N),
            "y binary": (E == 0) or bool(((d.y_edge == 0) | (d.y_edge == 1)).all()),
            "x finite": bool(torch.isfinite(d.x).all()),
            "edge_attr finite": bool(torch.isfinite(d.edge_attr).all()),
        }
        bad = [k2 for k2, v in checks.items() if not v]
        ok = ok and not bad
        rows.append({"qid": qg.qid, "N": N, "E": E, "pos": int(d.y_edge.sum()),
                     "seeds": int(d.seed_mask.sum()), "failed": bad})
        print(f"  {qg.qid:<28} x{tuple(d.x.shape)!s:<12} ei{tuple(d.edge_index.shape)!s:<10} "
              f"ea{tuple(d.edge_attr.shape)!s:<12} y{tuple(d.y_edge.shape)!s:<8} "
              f"pos={int(d.y_edge.sum()):<4} seeds={int(d.seed_mask.sum()):<3} "
              f"{'OK' if not bad else 'FAIL ' + ','.join(bad)}")
    print(f"  all {min(k, len(graphs))} graphs valid: {ok}")
    return {"all_valid": ok, "graphs": rows}


def run(cfg, args) -> int:
    unloaded = unload_all()
    print(f"--- M5: graphs, labels, ceilings ---")
    print(f"  ollama unloaded : {unloaded or 'nothing was loaded'}  (M5 never calls it)")
    g_cfg = cfg["graph"]
    hops = int(g_cfg["hops"])
    datasets = [args.dataset] if args.dataset else list(cfg["data"]["datasets"])
    tcache = TripleCache(cfg.path("triples"), env("SLM_TAG"), effective_prompt_version(cfg))
    print(f"  triple cache    : {len(tcache)} paragraphs")
    print(f"  H={hops}  caps: {g_cfg['max_nodes']} nodes / {g_cfg['max_edges']} edges  "
          f"merge_near_dupes={g_cfg['merge_near_duplicates']} @ cos>={g_cfg['merge_cosine']}")

    pools = {}
    for ds in datasets:
        ps = pool_size_for(cfg, ds)
        for split in ("train", "validation"):
            pools[(ds, split)] = load_pools(ds, split, ps)

    # ---- pre-cache every entity name and relation phrase, then free bge ----
    ent, rel = set(), set()
    for exs in pools.values():
        for ex in exs:
            for p in ex.paragraphs:
                for t in tcache.get(p.text) or []:
                    h, ta = normalize_entity(t.head), normalize_entity(t.tail)
                    if h:
                        ent.add(h)
                    if ta:
                        ent.add(ta)
                    r = " ".join(str(t.rel).split())
                    if r:
                        rel.add(r)
    print(f"\n  unique normalized entities : {len(ent):,}")
    print(f"  unique relation phrases    : {len(rel):,}")

    emb_cache = EmbeddingCache(cfg.path("embeddings"), env("EMBED_MODEL"))
    device = cfg.get("device") or "cpu"
    t0 = time.perf_counter()
    rss_before = own_tree_rss_mb()
    emb = Embedder(env("EMBED_MODEL"), device=device, cache=emb_cache, batch_size=128)
    todo_e = [t for t in sorted(ent) if emb_cache.get(t, "entities") is None]
    todo_r = [t for t in sorted(rel) if emb_cache.get(t, "relations") is None]
    print(f"  to encode: {len(todo_e):,} entities, {len(todo_r):,} relations", flush=True)
    for label, todo, kind in (("entities", todo_e, "entities"), ("relations", todo_r, "relations")):
        for i in range(0, len(todo), 2048):
            emb.encode(todo[i:i + 2048], kind=kind)
            print(f"    {label}: {min(i + 2048, len(todo)):,}/{len(todo):,}", flush=True)
    emb_cache.flush()
    model_loaded = emb._model is not None
    emb.free()
    import gc

    gc.collect()
    print(f"  embedding pass: {time.perf_counter()-t0:.1f}s  "
          f"(encoded {emb.n_encoded:,}, hits {emb.n_cache_hits:,})")
    print(f"  bge freed: model_was_loaded={model_loaded}, own RSS "
          f"{rss_before:.0f} -> {own_tree_rss_mb():.0f} MB")

    # ---- build graphs (serves embeddings from the warm cache, no model) ----
    warm = Embedder(env("EMBED_MODEL"), device=device, cache=emb_cache)
    report = {"hops": hops, "caps": {"max_nodes": int(g_cfg["max_nodes"]),
                                     "max_edges": int(g_cfg["max_edges"])},
              "entities_cached": len(ent), "relations_cached": len(rel), "datasets": {}}
    graph_dir = cfg.path("graphs")
    graph_dir.mkdir(parents=True, exist_ok=True)
    all_graphs: dict[tuple[str, str], list] = {}

    for (ds, split), exs in pools.items():
        t1 = time.perf_counter()
        gs = []
        for ex in exs:
            triples = []
            for p in ex.paragraphs:
                triples.extend(tcache.get(p.text) or [])
            gs.append(build_graph(
                ex, triples, warm, hops=hops,
                max_nodes=int(g_cfg["max_nodes"]), max_edges=int(g_cfg["max_edges"]),
                merge_near_duplicates=bool(g_cfg["merge_near_duplicates"]),
                merge_cosine=float(g_cfg["merge_cosine"]),
                max_seeds=int(g_cfg["max_seeds"]),
                seed_fallback_topk=int(g_cfg["seed_fallback_topk"]),
                connectivity=dict(g_cfg.get("connectivity") or {}),
            ))
        all_graphs[(ds, split)] = gs
        print(f"  [{ds}/{split}] {len(gs)} graphs in {time.perf_counter()-t1:.1f}s", flush=True)
        # pre-pooling gold sets, so the ceiling can be reported unconditionally
        orig = {e.qid: e.gold_pids for e in load_examples(
            ds, split, int(cfg["data"]["target_n"]), int(cfg["seed"]))}
        report["datasets"].setdefault(ds, {})[split] = measure(
            gs, pool_size_for(cfg, ds), orig_gold=orig)

    if warm._model is not None:
        warm.free()

    # ---- serialize PyG graphs -------------------------------------------
    t2 = time.perf_counter()
    n_saved = 0
    for (ds, split), gs in all_graphs.items():
        datas = [to_pyg(qg, emb_cache, hops) for qg in gs]
        torch.save(datas, graph_dir / f"{ds}_{split}_H{hops}.pt")
        n_saved += len(datas)
    print(f"\n  serialized {n_saved} PyG graphs -> {graph_dir.relative_to(ROOT)} "
          f"in {time.perf_counter()-t2:.1f}s")

    # ---- dropped train questions (Section 4.2) ---------------------------
    dropped = {}
    for ds in datasets:
        gs = all_graphs[(ds, "train")]
        drop = [g.qid for g in gs if g.positive_edges == 0]
        dropped[ds] = {"n_train": len(gs), "dropped": len(drop),
                       "rate": _frac(len(drop), len(gs)), "examples": drop[:5]}
    report["dropped_train_questions"] = dropped

    # ---- printed report --------------------------------------------------
    for ds in datasets:
        for split in ("train", "validation"):
            st = report["datasets"][ds][split]
            print(f"\n{'='*118}\n{ds.upper()} / {split}   (frozen config, Section 4.1.1)\n{'='*118}")
            print(f"  {'bucket':<7} {'n':>4} {'comps':>6} {'bfs%':>7} {'cov':>7} "
                  f"{'allgold':>9} {'allgold':>9} {'in-pool':>8} {'nodes':>7} {'edges':>7} "
                  f"{'cap%':>6} {'pos%':>6} {'posw':>6}")
            print(f"  {'':<7} {'':>4} {'':>6} {'':>7} {'':>7} {'(uncond)':>9} {'(cond)':>9} "
                  f"{'':>8} {'':>7} {'':>7} {'':>6} {'':>6} {'':>6}")
            for b in sorted(st, key=lambda x: (x == "ALL", x)):
                v = st[b]
                f = lambda x: f"{x:.1%}" if x is not None else "n/a"
                pw = f"{v['pos_weight']:.2f}" if v["pos_weight"] else "n/a"
                print(f"  {b:<7} {v['n_questions']:>4} {v['components_per_question']:>6.1f} "
                      f"{v['bfs_node_share']:>6.1%} {v['pool_covered_share']:>6.1%} "
                      f"{f(v['allgold_ceiling']):>9} {f(v['allgold_ceiling_conditional_on_pooling']):>9} "
                      f"{f(v['all_gold_in_pool']):>8} {v['nodes_mean']:>7.1f} {v['edges_mean']:>7.1f} "
                      f"{v['cap_bound_rate']:>5.1%} {v['positive_edge_rate']:>5.1%} {pw:>6}")
            a = st["ALL"]
            print(f"  empty graphs {a['empty_graphs']}, questions with no positive edge "
                  f"{a['questions_without_positive_edge']}/{a['n_questions']}; "
                  f"added links/question {a['added_links_mean']}")

    print(f"\n--- dropped TRAIN questions (Section 4.2: training only, kept in eval) ---")
    for ds, d in dropped.items():
        print(f"  {ds:<9} {d['dropped']}/{d['n_train']} ({d['rate']:.1%})")

    # ---- shape validation ------------------------------------------------
    sample = all_graphs[(datasets[0], "train")]
    report["shape_validation"] = validate_shapes(sample, emb_cache, hops, k=10)

    checks = {
        "graph ceiling reported both conditional and unconditional": all(
            report["datasets"][ds][sp]["ALL"]["allgold_ceiling"] is not None
            and report["datasets"][ds][sp]["ALL"]["allgold_ceiling_conditional_on_pooling"] is not None
            for ds in datasets for sp in ("train", "validation")
        ),
        "unconditional ceiling <= all-gold-in-pool (identity check)": all(
            report["datasets"][ds][sp]["ALL"]["allgold_ceiling"]
            <= report["datasets"][ds][sp]["ALL"]["all_gold_in_pool"] + 1e-9
            for ds in datasets for sp in ("train", "validation")
        ),
        "dropped train questions logged": all(d["rate"] is not None for d in dropped.values()),
        "positive-edge rate available for pos_weight": all(
            report["datasets"][ds][sp]["ALL"]["pos_weight"] is not None
            for ds in datasets for sp in ("train", "validation")
        ),
        "PyG Data shapes valid on 10 graphs": report["shape_validation"]["all_valid"],
        "cap-binding stays low (D14 target <= 10%)": all(
            (report["datasets"][ds][sp]["ALL"]["cap_bound_rate"] or 0.0) <= 0.10
            for ds in datasets for sp in ("train", "validation")
        ),
        "no scaffolding edge is a positive": all(
            r["failed"] == [] for r in report["shape_validation"]["graphs"]
        ),
    }
    print("\n=== M5 ACCEPTANCE ===")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    report["acceptance"] = checks
    report["all_passed"] = all(checks.values())

    out = ROOT / "results" / "m5_report.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nreport -> {out.relative_to(ROOT)}")
    print(f"M5: {'PASS' if report['all_passed'] else 'FAIL'}")
    return 0 if report["all_passed"] else 1
