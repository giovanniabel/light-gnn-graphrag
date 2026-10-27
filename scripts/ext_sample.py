"""M11 step 1: draw the 350 new validation questions per dataset (deviation D27).

Sampling rule, seed and stratification are fixed by D27, which was committed
before this script was run. Pool construction is the frozen, gold-blind one.

Asserts zero overlap with the original 150 and with train, and writes the chosen
qids with the seed and a sha256 of the list so the sample is verifiable.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT, env, load_config  # noqa: E402
from src.data import (  # noqa: E402
    _hotpotqa_example,
    _musique_example,
    _musique_hops,
    _stream_hotpotqa,
    _stream_musique,
    build_pool,
    ensure_raw,
    pool_size_for,
)
from src.embedder import EmbeddingCache, Embedder  # noqa: E402
from src.validation_guard import allow, require  # noqa: E402
from scripts.m1_data import POOL_DIR, save_pools  # noqa: E402

SEED = 20261026                     # fixed by D27
N_NEW = 350
MUSIQUE_PER_HOP = {2: 117, 3: 117, 4: 116}


def original_qids(ds: str) -> set[str]:
    """The original 150, read from the existing val files and used ONLY to exclude."""
    p = ROOT / "results" / f"val_vector_{ds}.jsonl"
    return {json.loads(l)["qid"] for l in p.read_text().splitlines() if l.strip()}


def train_qids(ds: str, cfg) -> set[str]:
    from scripts.m1_data import load_pools

    return {e.qid for e in load_pools(ds, "train", pool_size_for(cfg, ds))}


def eligible(ds: str, cfg, files) -> list:
    """Every validation question that meets the frozen eligibility filters."""
    ps = pool_size_for(cfg, ds)
    out = []
    if ds == "musique":
        for r in _stream_musique(files["musique_eval"]):
            if not r.get("answerable", True):
                continue
            if len(r["paragraphs"]) < ps:
                continue
            out.append(_musique_example(r))
    else:
        for r in _stream_hotpotqa([files["hotpotqa_eval"]]):
            if len(r["context"]["title"]) < ps:
                continue
            if r.get("level") != "hard":          # D27: draw from the hard level
                continue
            out.append(_hotpotqa_example(r))
    return out


def main() -> int:
    cfg = load_config()
    allow("M11 extension: sampling the 350 new validation questions (D27)")
    require("ext_sample: reading the validation split to draw new questions")
    files = ensure_raw(cfg)
    emb_cache = EmbeddingCache(cfg.path("embeddings"), env("EMBED_MODEL"))
    emb = Embedder(env("EMBED_MODEL"), device=cfg.get("device") or "cpu", cache=emb_cache)
    report = {}

    for ds in cfg["data"]["datasets"]:
        ps = pool_size_for(cfg, ds)
        orig = original_qids(ds)
        trn = train_qids(ds, cfg)
        pool_all = eligible(ds, cfg, files)
        avail = [e for e in pool_all if e.qid not in orig and e.qid not in trn]

        print(f"\n  {ds}: {len(pool_all)} eligible validation questions, "
              f"{len(avail)} after excluding the original {len(orig)}")

        rng = random.Random(SEED)
        if ds == "musique":
            by_hop: dict[int, list] = {}
            for e in avail:
                by_hop.setdefault(e.n_hops, []).append(e)
            print(f"    available per hop: "
                  f"{ {h: len(v) for h, v in sorted(by_hop.items())} }")
            short = {h: n for h, n in MUSIQUE_PER_HOP.items()
                     if len(by_hop.get(h, [])) < n}
            if short:
                print(f"    !! INSUFFICIENT for the pre-registered design: need "
                      f"{MUSIQUE_PER_HOP}, short on {short}")
                return 2
            chosen = []
            for h in sorted(MUSIQUE_PER_HOP):
                qs = sorted(by_hop[h], key=lambda e: e.qid)
                rng.shuffle(qs)
                chosen += qs[:MUSIQUE_PER_HOP[h]]
        else:
            if len(avail) < N_NEW:
                print(f"    !! INSUFFICIENT: need {N_NEW}, have {len(avail)}")
                return 2
            qs = sorted(avail, key=lambda e: e.qid)
            rng.shuffle(qs)
            chosen = qs[:N_NEW]

        chosen = sorted(chosen, key=lambda e: e.qid)
        qids = [e.qid for e in chosen]
        assert len(qids) == N_NEW, f"{ds}: drew {len(qids)}, expected {N_NEW}"
        assert not (set(qids) & orig), f"{ds}: overlap with the original 150"
        assert not (set(qids) & trn), f"{ds}: overlap with train"
        assert len(set(qids)) == len(qids), f"{ds}: duplicate qids"

        pooled = [build_pool(ex, emb, ps) for ex in chosen]
        sizes = sorted({len(e.paragraphs) for e in pooled})
        assert sizes == [ps], f"{ds}: pool sizes {sizes}, expected [{ps}]"
        save_pools(pooled, ds, "validation_ext", ps)

        sha = hashlib.sha256("\n".join(qids).encode()).hexdigest()
        info = {
            "dataset": ds, "n": len(qids), "seed": SEED,
            "sha256_of_qid_list": sha,
            "pool_size": ps,
            "excluded_original_150": len(orig),
            "excluded_train": len(trn),
            "eligible_total": len(pool_all),
            "available_after_exclusion": len(avail),
            "hop_distribution": {str(k): v for k, v in
                                 sorted(Counter(e.n_hops for e in pooled).items())},
            "type_distribution": {str(k): v for k, v in
                                  sorted(Counter(e.meta.get("hotpot_type")
                                                 for e in pooled).items(),
                                         key=lambda kv: str(kv[0]))},
            "level_distribution": {str(k): v for k, v in
                                   sorted(Counter(e.meta.get("level")
                                                  for e in pooled).items(),
                                          key=lambda kv: str(kv[0]))},
            "pool_file": str((POOL_DIR / f"{ds}_validation_ext_p{ps}.json").relative_to(ROOT)),
            "qids": qids,
        }
        (ROOT / "results" / f"ext_sample_{ds}.json").write_text(json.dumps(info, indent=2))
        report[ds] = {k: v for k, v in info.items() if k != "qids"}
        print(f"    drew {len(qids)} | hops {info['hop_distribution']}"
              + (f" | types {info['type_distribution']}" if ds == "hotpotqa" else ""))
        print(f"    overlap with original 150: 0 | with train: 0 | pool size {sizes[0]} for all")
        print(f"    sha256 {sha[:16]}…  -> results/ext_sample_{ds}.json")

    emb_cache.flush()
    emb.free()
    nparas = sum(report[d]["n"] * report[d]["pool_size"] for d in report)
    print(f"\n  total new pool paragraph slots: {nparas:,}")
    (ROOT / "results" / "ext_sample_summary.json").write_text(json.dumps(report, indent=2))
    print(f"  -> results/ext_sample_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
