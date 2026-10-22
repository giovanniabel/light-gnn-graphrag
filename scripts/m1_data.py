"""M1 acceptance check (agent.md Section 11):
  * train/eval id sets disjoint (assert)
  * pool recall ceiling printed per dataset
  * build_pool never reads is_gold (unit test -- tests/test_data.py)

Pools are cached to data/cache/pools/ so M2's pilot and M3's embedding phase
reuse them instead of re-sampling (Section 8.1).
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT, env, load_config, save_config  # noqa: E402
from src.data import (  # noqa: E402
    assert_disjoint,
    pool_size_for,
    build_pool,
    carve_dev,
    ensure_raw,
    hop_distribution,
    level_distribution,
    load_examples,
    pool_recall,
    type_distribution,
)
from src.embedder import EmbeddingCache, Embedder  # noqa: E402
from src.schemas import Example, Paragraph  # noqa: E402

POOL_DIR = ROOT / "data" / "cache" / "pools"


def pool_path(dataset: str, split: str, pool_size: int) -> Path:
    return POOL_DIR / f"{dataset}_{split}_p{pool_size}.json"


def save_pools(examples: list[Example], dataset: str, split: str, pool_size: int) -> Path:
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    p = pool_path(dataset, split, pool_size)
    with open(p, "w") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex)) + "\n")
    return p


def load_pools(dataset: str, split: str, pool_size: int) -> list[Example] | None:
    if split == "validation":
        from src.validation_guard import require

        require(f"load_pools({dataset!r}, 'validation')")
    p = pool_path(dataset, split, pool_size)
    if not p.exists():
        return None
    out = []
    with open(p) as f:
        for line in f:
            d = json.loads(line)
            d["paragraphs"] = [Paragraph(**x) for x in d["paragraphs"]]
            out.append(Example(**d))
    return out


def _build_split(dataset, split, n, seed, embedder, pool_size, resume) -> tuple[list, list, dict]:
    cached = load_pools(dataset, split, pool_size) if resume else None
    t0 = time.perf_counter()
    print(f"\n  [{dataset}/{split}] loading n={n} ...", flush=True)
    originals = load_examples(dataset, split, n, seed)
    print(f"    sampled {len(originals)} questions in {time.perf_counter()-t0:.1f}s")

    if cached is not None and {e.qid for e in cached} == {e.qid for e in originals}:
        print(f"    pools: cache hit ({len(cached)}) -> {pool_path(dataset, split, pool_size).name}")
        pooled = cached
    else:
        t1 = time.perf_counter()
        pooled = [build_pool(ex, embedder, pool_size) for ex in originals]
        print(f"    built pools in {time.perf_counter()-t1:.1f}s "
              f"(encoded {embedder.n_encoded}, cache hits {embedder.n_cache_hits})")
        save_pools(pooled, dataset, split, pool_size)

    rec = pool_recall(pooled, originals)
    info = {
        "n_requested": n,
        "n_loaded": len(originals),
        "paragraphs_per_question": sorted({len(e.paragraphs) for e in pooled}),
        "hop_distribution": hop_distribution(pooled),
        "pool_file": str(pool_path(dataset, split, pool_size).relative_to(ROOT)),
        **rec,
    }
    if dataset == "hotpotqa":
        info["type_distribution"] = type_distribution(pooled)
        info["level_distribution"] = level_distribution(pooled)
    return originals, pooled, info


def run(cfg, args) -> int:
    d = cfg["data"]
    seed = int(cfg["seed"])
    n = args.n or int(d["target_n"])
    datasets = [args.dataset] if args.dataset else list(d["datasets"])
    pool_sizes = {ds: pool_size_for(cfg, ds) for ds in datasets}
    report = {"pool_sizes": pool_sizes, "n_per_split": n, "seed": seed, "datasets": {}}

    print("--- raw files ---")
    t0 = time.perf_counter()
    files = ensure_raw(cfg)
    for k, v in files.items():
        print(f"  {k:<20} {v.stat().st_size/1e6:>8.1f} MB  {v.name}")
    print(f"  ready in {time.perf_counter()-t0:.1f}s")

    cache = EmbeddingCache(cfg.path("embeddings"), env("EMBED_MODEL"))
    device = cfg.get("device") or "cpu"
    print(f"\n--- pool building (pool sizes {pool_sizes}, device={device}) ---")

    with Embedder(env("EMBED_MODEL"), device=device, cache=cache) as emb:
        for ds in datasets:
            pool_size = pool_sizes[ds]
            tr_orig, tr_pool, tr_info = _build_split(ds, "train", n, seed, emb, pool_size, args.resume)
            ev_orig, ev_pool, ev_info = _build_split(ds, "validation", n, seed, emb, pool_size, args.resume)

            overlap = assert_disjoint(tr_pool, ev_pool)
            tr_only, dev = carve_dev(tr_pool, float(d["dev_frac"]), seed)
            dev_overlap = assert_disjoint(tr_only, dev) | assert_disjoint(dev, ev_pool)

            report["datasets"][ds] = {
                "pool_size": pool_size,
                "train": tr_info,
                "validation": ev_info,
                "qid_overlap_train_eval": len(overlap),
                "dev_split": {"n_train": len(tr_only), "n_dev": len(dev),
                              "overlap": len(dev_overlap)},
            }

    print("\n=== POOL RECALL CEILING (per dataset, never pooled together) ===")
    for ds, r in report["datasets"].items():
        print(f"\n  {ds}  (pool_size={r['pool_size']})")
        for split in ("train", "validation"):
            s = r[split]
            print(f"    {split:<11} n={s['n_loaded']:<4} "
                  f"pool_recall={s['pool_recall']:.4f} "
                  f"({s['gold_paragraphs_in_pool']}/{s['gold_paragraphs_total']} gold paragraphs)  "
                  f"all-gold-in-pool={s['all_gold_rate']:.4f}")
            print(f"    {'':<11} paragraphs/question={s['paragraphs_per_question']}  hops={s['hop_distribution']}"
                  + (f"  types={s['type_distribution']}" if "type_distribution" in s else ""))
            if "level_distribution" in s:
                print(f"    {'':<11} levels={s['level_distribution']}")
        print(f"    train/eval qid overlap = {r['qid_overlap_train_eval']}  (assert: 0)")
        print(f"    dev carve: {r['dev_split']['n_train']} train / {r['dev_split']['n_dev']} dev, "
              f"overlap {r['dev_split']['overlap']}")

    # Section 3.1 gate: MuSiQue pool recall < 85% -> raise POOL_SIZE to 12
    gate_ok = True
    musique = report["datasets"].get("musique")
    if musique:
        worst = min(musique["train"]["pool_recall"], musique["validation"]["pool_recall"])
        report["musique_pool_recall_min"] = worst
        if worst < 0.85:
            gate_ok = False
            print(f"\n  !! MuSiQue pool recall {worst:.4f} < 0.85 (Section 3.1 gate)")
            report["pool_size_action"] = "raise MuSiQue pool size (Section 3.1 / D1)"
        else:
            print(f"\n  MuSiQue pool recall {worst:.4f} >= 0.85 "
                  f"-> pool size {pool_sizes.get('musique')} clears the Section 3.1 gate")

    checks = {
        "train/eval qid sets disjoint": all(
            r["qid_overlap_train_eval"] == 0 for r in report["datasets"].values()
        ),
        "dev split disjoint from train and eval": all(
            r["dev_split"]["overlap"] == 0 for r in report["datasets"].values()
        ),
        "pool recall ceiling printed per dataset": all(
            "pool_recall" in r["train"] for r in report["datasets"].values()
        ),
        "every question has exactly its dataset's pool size": all(
            r[s]["paragraphs_per_question"] == [r["pool_size"]]
            for r in report["datasets"].values() for s in ("train", "validation")
        ),
        "MuSiQue pool recall >= 85% (Section 3.1 gate)": gate_ok,
    }
    print("\n=== M1 ACCEPTANCE ===")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    report["acceptance"] = checks
    report["all_passed"] = all(checks.values())

    out = ROOT / "results" / "m1_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nreport -> {out.relative_to(ROOT)}")
    print(f"M1: {'PASS' if report['all_passed'] else 'FAIL'}")
    return 0 if report["all_passed"] else 1
