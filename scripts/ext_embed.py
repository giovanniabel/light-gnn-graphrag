"""M11: the M3 embedding pass for the extension pools.

M3 pre-caches every pool question and pool paragraph embedding, which is a
precondition of the frozen pipeline: Vector RAG reads them at query time and is
timed on the assumption that they are resident. M3 was never re-run for the
extension, so the first scoring sweep had HotpotQA's Vector RAG *encoding* 10
paragraphs per question inside the timed loop (291 ms median forward, against
0.51 ms on the original 150). MuSiQue was unaffected only by accident: its
top-14-of-20 pool selection had already needed those embeddings, while
HotpotQA's pool is all 10 paragraphs and needs no selection.

Questions and paragraphs are not enough (D30). Graph assembly also needs the
*entity* and *relation* texts cached: build_graph encodes entity names itself, so
those land in the in-memory cache before to_pyg reads them, but nothing encodes
relation texts, and to_pyg used to zero-fill the misses -- 10% of the extension's
edges got r_e = 0 and cos(r_e,q) = 0. The only way to cache them is to build the
graphs once, which is what the second pass below does. Entity names are warmed in
the same pass so that routing latency is not measured with bge-small running
inside the timed region.

This is not a new optimisation and changes no setting -- it restores the same
cache state the original run was measured under, so the two are comparable
(published train and validation splits are 0% miss on both kinds).
Nothing here reads or writes a model, a label or a hyperparameter.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT, env, load_config  # noqa: E402
from src.data import pool_size_for  # noqa: E402
from src.embedder import EmbeddingCache, Embedder  # noqa: E402
from src.extractor import TripleCache, effective_prompt_version  # noqa: E402
from src.graph_builder import build_graph  # noqa: E402
from src.validation_guard import allow  # noqa: E402
from scripts.m1_data import load_pools  # noqa: E402


def main() -> int:
    cfg = load_config()
    allow("M11: embedding pass for the extension pools (D27)")
    cache = EmbeddingCache(cfg.path("embeddings"), env("EMBED_MODEL"))
    emb = Embedder(env("EMBED_MODEL"), device=cfg.get("device") or "cpu", cache=cache)
    out = {"splits": {}}
    t_all = time.perf_counter()

    for ds in cfg["data"]["datasets"]:
        ps = pool_size_for(cfg, ds)
        pooled = load_pools(ds, "validation_ext", ps)
        t0 = time.perf_counter()
        before_enc, before_hit = emb.n_encoded, emb.n_cache_hits
        emb.encode_query([ex.question for ex in pooled])
        emb.encode([f"{p.title}. {p.text}" for ex in pooled for p in ex.paragraphs],
                   kind="paragraphs")
        out["splits"][ds] = {
            "n_questions": len(pooled),
            "paragraphs": sum(len(e.paragraphs) for e in pooled),
            "encoded": emb.n_encoded - before_enc,
            "cache_hits": emb.n_cache_hits - before_hit,
            "seconds": time.perf_counter() - t0,
        }
        d = out["splits"][ds]
        print(f"  [{ds}/validation_ext] {d['n_questions']} questions, "
              f"{d['paragraphs']} paragraphs: encoded {d['encoded']}, "
              f"cache hits {d['cache_hits']}, {d['seconds']:.1f}s", flush=True)

    # ---- second pass: entity and relation texts (D30) --------------------
    # Needs the graphs, so it goes through build_graph with the frozen Section 4.1
    # settings read straight from the config -- same call as the scoring runners.
    g = cfg["graph"]
    tc = TripleCache(cfg.path("triples"), env("SLM_TAG"), effective_prompt_version(cfg))
    for ds in cfg["data"]["datasets"]:
        pooled = load_pools(ds, "validation_ext", pool_size_for(cfg, ds))
        t0 = time.perf_counter()
        before_enc, before_hit = emb.n_encoded, emb.n_cache_hits
        rels, ents = set(), set()
        for ex in pooled:
            tri = []
            for para in ex.paragraphs:
                tri.extend(tc.get(para.text) or [])
            qg = build_graph(
                ex, tri, emb, hops=int(g["hops"]), max_nodes=int(g["max_nodes"]),
                max_edges=int(g["max_edges"]),
                merge_near_duplicates=bool(g["merge_near_duplicates"]),
                merge_cosine=float(g["merge_cosine"]), max_seeds=int(g["max_seeds"]),
                seed_fallback_topk=int(g["seed_fallback_topk"]),
                connectivity=dict(g.get("connectivity") or {}),
            )
            rels.update(qg.edge_rel)
            ents.update(qg.node_text or qg.nodes)
        emb.encode(sorted(rels), kind="relations")
        emb.encode(sorted(ents), kind="entities")
        out["splits"][ds]["graph_pass"] = {
            "relation_texts": len(rels),
            "entity_texts": len(ents),
            "encoded": emb.n_encoded - before_enc,
            "cache_hits": emb.n_cache_hits - before_hit,
            "seconds": time.perf_counter() - t0,
        }
        d = out["splits"][ds]["graph_pass"]
        print(f"  [{ds}/validation_ext] graph pass: {d['relation_texts']} relation "
              f"and {d['entity_texts']} entity texts: encoded {d['encoded']}, "
              f"cache hits {d['cache_hits']}, {d['seconds']:.1f}s", flush=True)

    n_written = cache.flush()
    out["shards_written"] = n_written
    out["total_seconds"] = time.perf_counter() - t_all
    print(f"\n  flushed {n_written} shard(s) in {out['total_seconds']:.1f}s total")
    (ROOT / "results" / "ext_embed.json").write_text(json.dumps(out, indent=2))
    print("-> results/ext_embed.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
