"""Persist the three connectivity diagnostics quoted in report.md.

Cache-only: counts paragraphs with usable triples and evidence-only ego coverage.
No model inference -- the embedder serves from the warm disk cache.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT, env, load_config  # noqa: E402
from src.data import carve_dev, pool_size_for  # noqa: E402
from src.embedder import EmbeddingCache, Embedder  # noqa: E402
from src.extractor import TripleCache, effective_prompt_version  # noqa: E402
from src.graph_builder import build_graph  # noqa: E402
from src.normalize import normalize_entity  # noqa: E402
from scripts.m1_data import load_pools  # noqa: E402


def run(cfg) -> dict:
    g = cfg["graph"]
    tc = TripleCache(cfg.path("triples"), env("SLM_TAG"), effective_prompt_version(cfg))
    ec = EmbeddingCache(cfg.path("embeddings"), env("EMBED_MODEL"))
    warm = Embedder(env("EMBED_MODEL"), device=cfg.get("device") or "cpu", cache=ec)
    out = {}
    for ds in cfg["data"]["datasets"]:
        ps = pool_size_for(cfg, ds)
        pooled = load_pools(ds, "train", ps)
        tr, _dev = carve_dev(pooled, float(cfg["data"]["dev_frac"]), int(cfg["seed"]))

        slots = usable = 0
        for ex in pooled:
            for p in ex.paragraphs:
                slots += 1
                ts = tc.get(p.text) or []
                if any(normalize_entity(t.head) and normalize_entity(t.tail)
                       and normalize_entity(t.head) != normalize_entity(t.tail)
                       and " ".join(str(t.rel).split()) for t in ts):
                    usable += 1

        cov_ev, cov_any = [], []
        for ex in tr:
            tri = []
            for p in ex.paragraphs:
                tri.extend(tc.get(p.text) or [])
            qg = build_graph(ex, tri, warm, hops=int(g["hops"]),
                             max_nodes=int(g["max_nodes"]), max_edges=int(g["max_edges"]),
                             merge_near_duplicates=bool(g["merge_near_duplicates"]),
                             merge_cosine=float(g["merge_cosine"]),
                             max_seeds=int(g["max_seeds"]),
                             seed_fallback_topk=int(g["seed_fallback_topk"]),
                             connectivity=dict(g.get("connectivity") or {}))
            cov_ev.append(len(qg.covered_pids()) / ps)
            cov_any.append(len(qg.covered_pids_any_edge()) / ps)

        out[ds] = {
            "n_train_subset": len(tr), "pool_size": ps,
            "paragraph_slots": slots, "slots_with_usable_triple": usable,
            "usable_triple_rate": usable / slots,
            "no_usable_triple_rate": 1 - usable / slots,
            "hard_coverage_ceiling": usable / slots,
            "ego_coverage_evidence_only": statistics.fmean(cov_ev),
            "ego_coverage_any_edge": statistics.fmean(cov_any),
        }
        d = out[ds]
        print(f"  {ds}: {d['slots_with_usable_triple']}/{d['paragraph_slots']} slots have a "
              f"usable triple ({d['usable_triple_rate']:.1%}); "
              f"{d['no_usable_triple_rate']:.1%} do not")
        print(f"      ego coverage: evidence-only {d['ego_coverage_evidence_only']:.1%}, "
              f"any-edge {d['ego_coverage_any_edge']:.1%}  (train subset n={d['n_train_subset']})")
    warm.free()
    (ROOT / "results" / "m5_diagnostics.json").write_text(json.dumps(out, indent=2))
    print("\n-> results/m5_diagnostics.json")
    return out


if __name__ == "__main__":
    run(load_config())
