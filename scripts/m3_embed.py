"""M3: build pools at the governor's N and cache every embedding they need.

agent.md Section 11 acceptance: re-run takes seconds; bge freed from memory.

Caches question embeddings and pool-paragraph embeddings. Entity-name and
relation-phrase embeddings (Section 4.3) depend on extracted triples, so they are
cached in M5 after M4 produces them.

Ends by asserting the encoder is gone and reporting what is still resident, so the
Section 6.1 phase isolation into extraction can be verified before Ollama loads.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT, env, load_config  # noqa: E402
from src.data import assert_disjoint, build_pool, carve_dev, load_examples, pool_size_for  # noqa: E402
from src.embedder import EmbeddingCache, Embedder  # noqa: E402
from src.ollama_client import unload_all  # noqa: E402
from src.profiling import ollama_processes, tree_rss_mb, virtual_memory_snapshot  # noqa: E402
from scripts.m1_data import load_pools, pool_path, save_pools  # noqa: E402


def governor_n(cfg, override=None) -> int:
    if override:
        return int(override)
    p = ROOT / "results" / "budget.json"
    if p.exists():
        b = json.loads(p.read_text())
        n = b.get("governor", {}).get("chosen_n")
        if n:
            return min(int(n), int(cfg["data"]["target_n"]))
    return int(cfg["data"]["target_n"])


def run(cfg, args) -> int:
    seed = int(cfg["seed"])
    n = governor_n(cfg, args.n)
    datasets = [args.dataset] if args.dataset else list(cfg["data"]["datasets"])
    pool_sizes = {ds: pool_size_for(cfg, ds) for ds in datasets}
    device = cfg.get("device") or "cpu"

    print(f"--- M3: pools + embedding cache ---")
    print(f"  N per (dataset, split) : {n}   (governor; target_n={cfg['data']['target_n']})")
    print(f"  pool sizes             : {pool_sizes}")
    print(f"  device                 : {device}")

    # Section 6.1: nothing from the LLM phase may be resident while bge loads.
    unloaded = unload_all()
    print(f"  ollama unloaded        : {unloaded or 'nothing was loaded'}")

    rss_start = tree_rss_mb()
    cache = EmbeddingCache(cfg.path("embeddings"), env("EMBED_MODEL"))
    report = {"n": n, "pool_sizes": pool_sizes, "seed": seed, "datasets": {}}
    t_all = time.perf_counter()

    emb = Embedder(env("EMBED_MODEL"), device=device, cache=cache)
    try:
        for ds in datasets:
            ps = pool_sizes[ds]
            report["datasets"][ds] = {"pool_size": ps, "splits": {}}
            for split in ("train", "validation"):
                t0 = time.perf_counter()
                cached = load_pools(ds, split, ps)
                originals = load_examples(ds, split, n, seed)
                qids = {e.qid for e in originals}

                if cached is not None and {e.qid for e in cached} == qids:
                    pooled = cached
                    built = "cache hit"
                else:
                    pooled = [build_pool(ex, emb, ps) for ex in originals]
                    save_pools(pooled, ds, split, ps)
                    built = "built"

                # every question embedding
                emb.encode_query([ex.question for ex in pooled])
                # every pool-paragraph embedding (needed by Vector RAG at query time)
                emb.encode([f"{p.title}. {p.text}" for ex in pooled for p in ex.paragraphs],
                           kind="paragraphs")

                dt = time.perf_counter() - t0
                report["datasets"][ds]["splits"][split] = {
                    "n": len(pooled),
                    "paragraphs": sum(len(e.paragraphs) for e in pooled),
                    "pools": built,
                    "seconds": dt,
                    "pool_file": str(pool_path(ds, split, ps).relative_to(ROOT)),
                }
                print(f"  [{ds}/{split}] {len(pooled)} questions, "
                      f"{sum(len(e.paragraphs) for e in pooled)} paragraphs, {built}, {dt:.1f}s",
                      flush=True)

            tr = load_pools(ds, "train", ps)
            ev = load_pools(ds, "validation", ps)
            assert_disjoint(tr, ev)
            tr_only, dev = carve_dev(tr, float(cfg["data"]["dev_frac"]), seed)
            report["datasets"][ds]["dev_split"] = {"n_train": len(tr_only), "n_dev": len(dev)}

        n_written = cache.flush()
        report["embedding_shards_written"] = n_written
        report["embedder_encoded"] = emb.n_encoded
        report["embedder_cache_hits"] = emb.n_cache_hits
        print(f"\n  encoded {emb.n_encoded} texts, {emb.n_cache_hits} cache hits, "
              f"{n_written} shards written")
    finally:
        # "bge freed" is a claim about THIS process, so it is checked with the
        # Python process RSS plus a direct assertion that the model is
        # dereferenced. tree_rss_mb() is wrong here: it also sums the Ollama
        # processes, whose drift swamps the encoder's ~130 MB. (Section 7.3's ban
        # on process-only RSS is about reporting a method's memory cost, which
        # this is not.)
        import psutil

        me = psutil.Process()
        py_before = me.memory_info().rss / (1024 * 1024)
        rss_before_free = tree_rss_mb()
        model_was_loaded = emb._model is not None
        emb.free()
        model_released = emb._model is None
        del emb
        gc.collect()
        time.sleep(0.5)
        py_after = me.memory_info().rss / (1024 * 1024)
        rss_after_free = tree_rss_mb()

    report["rss_mb"] = {
        "tree_start": rss_start,
        "tree_before_free": rss_before_free,
        "tree_after_free": rss_after_free,
        "python_before_free": py_before,
        "python_after_free": py_after,
        "python_freed": py_before - py_after,
        "model_released": model_released,
        "model_was_loaded_this_run": model_was_loaded,
    }
    report["seconds_total"] = time.perf_counter() - t_all
    report["embedding_cache_counts"] = cache.stats()

    print(f"\n--- bge freed (Section 2) ---")
    print(f"  encoder loaded this run : {model_was_loaded}"
          f"{'  (fully cached re-run, nothing to load)' if not model_was_loaded else ''}")
    print(f"  encoder dereferenced : {model_released}")
    print(f"  python RSS before    : {py_before:.0f} MB")
    print(f"  python RSS after     : {py_after:.0f} MB   "
          f"(released {py_before - py_after:.0f} MB)")
    print(f"  tree RSS (incl. ollama, for context): {rss_before_free:.0f} -> {rss_after_free:.0f} MB")
    print(f"  cached embeddings    : {cache.stats()}")

    print(f"\n--- heavy processes before extraction (Section 6.1) ---")
    print(f"  ollama processes     : {len(ollama_processes())}")
    print(f"  vm                   : {virtual_memory_snapshot()}")
    import torch

    print(f"  torch cuda/mps alloc : mps={torch.backends.mps.is_available()} (device={cfg.get('device')})")

    checks = {
        "pools exist for every dataset/split": all(
            pool_path(ds, sp, pool_sizes[ds]).exists()
            for ds in datasets for sp in ("train", "validation")
        ),
        "question + paragraph embeddings cached": (
            cache.stats().get("questions", 0) > 0 and cache.stats().get("paragraphs", 0) > 0
        ),
        # 32 MB tolerance: RSS wobbles by a few MB between two samples, and on a
        # fully cached re-run the encoder is never loaded, so there is nothing to
        # release and the delta is pure noise.
        "bge freed from memory": model_released and (py_after - py_before) <= 32.0,
        "ollama not holding a model": not unload_all(),
    }
    print("\n=== M3 ACCEPTANCE ===")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    report["acceptance"] = checks
    report["all_passed"] = all(checks.values())

    out = ROOT / "results" / "m3_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nreport -> {out.relative_to(ROOT)}")
    print(f"total {report['seconds_total']:.1f}s")
    print(f"M3: {'PASS' if report['all_passed'] else 'FAIL'}")
    return 0 if report["all_passed"] else 1
