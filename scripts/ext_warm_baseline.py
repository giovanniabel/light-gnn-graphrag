"""M11 D30: cache the graph texts of the PUBLISHED validation split.

Purpose is a like-for-like latency baseline, nothing else. After D30 the
extension's graph assembly does zero bge-small encodes inside the timed routing
region; the published 150 still does 8.2 (MuSiQue) and 6.5 (HotpotQA) per graph,
because the M3/M5 passes never persisted every text `build_graph` produces --
notably the ones that only exist after near-duplicate merging. Comparing the
extension's clean 26.5 ms against Section 7's 43.8 ms would therefore compare two
different measurements, and in the direction that flatters the extension.

This writes embedding-cache entries only. It touches no config, no checkpoint, no
label and no published results file, and it cannot change a metric: the vectors
are the same ones `build_graph` was already computing on the fly, just persisted.
Same fault class as D23's Edge-Sim artefact (583.5 -> 10.6 ms), handled the same
way: fix the measurement, leave the metrics alone, write down both numbers.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import env, load_config  # noqa: E402
from src.data import pool_size_for  # noqa: E402
from src.embedder import EmbeddingCache, Embedder  # noqa: E402
from src.extractor import TripleCache, effective_prompt_version  # noqa: E402
from src.graph_builder import build_graph  # noqa: E402
from src.validation_guard import allow  # noqa: E402
from scripts.m1_data import load_pools  # noqa: E402


def main() -> int:
    cfg = load_config()
    allow("M11 D30: warm the published validation split's graph texts (latency baseline)")
    g = cfg["graph"]
    cache = EmbeddingCache(cfg.path("embeddings"), env("EMBED_MODEL"))
    tc = TripleCache(cfg.path("triples"), env("SLM_TAG"), effective_prompt_version(cfg))

    with Embedder(env("EMBED_MODEL"), device=cfg.get("device") or "cpu",
                  cache=cache) as emb:
        for ds in cfg["data"]["datasets"]:
            pooled = load_pools(ds, "validation", pool_size_for(cfg, ds))
            t0 = time.perf_counter()
            before = emb.n_encoded
            rels, ents = set(), set()
            for ex in pooled:
                tri = []
                for para in ex.paragraphs:
                    tri.extend(tc.get(para.text) or [])
                qg = build_graph(
                    ex, tri, emb, hops=int(g["hops"]), max_nodes=int(g["max_nodes"]),
                    max_edges=int(g["max_edges"]),
                    merge_near_duplicates=bool(g["merge_near_duplicates"]),
                    merge_cosine=float(g["merge_cosine"]),
                    max_seeds=int(g["max_seeds"]),
                    seed_fallback_topk=int(g["seed_fallback_topk"]),
                    connectivity=dict(g.get("connectivity") or {}),
                )
                rels.update(qg.edge_rel)
                ents.update(qg.node_text or qg.nodes)
            emb.encode(sorted(rels), kind="relations")
            emb.encode(sorted(ents), kind="entities")
            print(f"  [{ds}/validation] {len(rels)} relation + {len(ents)} entity "
                  f"texts, encoded {emb.n_encoded - before}, "
                  f"{time.perf_counter() - t0:.1f}s", flush=True)

    print(f"  flushed {cache.flush()} shard(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
