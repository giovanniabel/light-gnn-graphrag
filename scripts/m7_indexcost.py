"""M7 step 1: re-measure indexing cost on a clean machine (replaces the void D11).

D11 voided the M4 timings because that run executed under up to +8 GB of system
swap, far past §7.3's 500 MB invalidation rule. This measures the same work with
swap at the gated baseline, writing to its OWN cache file so data/cache/triples.jsonl
is never touched.

Paragraphs are drawn from TRAIN questions outside the sampled 150, and filtered to
keys absent from the real cache, so every one is a genuine cold extraction.
Validation is never read.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT, env, load_config  # noqa: E402
from src.data import load_examples, pool_size_for  # noqa: E402
from src.embedder import EmbeddingCache, Embedder  # noqa: E402
from src.extractor import (  # noqa: E402
    TripleCache,
    effective_prompt_version,
    extract_paragraph,
)
from src.normalize import normalize_entity  # noqa: E402
from src.ollama_client import OllamaClient, unload_all  # noqa: E402
from src.profiling import StageProfiler, swap_used_mb, virtual_memory_snapshot  # noqa: E402

N_PARAGRAPHS = 100


def run(cfg) -> dict:
    ex_cfg = cfg["extraction"]
    tag = env("SLM_TAG")
    pv = effective_prompt_version(cfg)
    real = TripleCache(cfg.path("triples"), tag, pv)
    print(f"--- M7 step 1: indexing cost, clean machine ---")
    print(f"  real cache holds {len(real)} paragraphs; those keys are EXCLUDED")
    print(f"  writing to a separate cache: data/cache/triples_m7_indexcost.jsonl")
    print(f"  swap at start: {swap_used_mb():.0f} MB")

    # draw TRAIN questions outside the sampled 150 (different seed), keep uncached keys
    want, seen_q = [], 0
    for seed in (999, 1000, 1001, 1002):
        for ds in cfg["data"]["datasets"]:
            exs = load_examples(ds, "train", 60, seed)
            for e in exs:
                seen_q += 1
                for p in e.paragraphs:
                    if real.key_for(p.text) not in real._index:
                        want.append((e.qid, p))
            if len(want) >= N_PARAGRAPHS * 4:
                break
        if len(want) >= N_PARAGRAPHS * 4:
            break
    rng = random.Random(int(cfg["seed"]))
    rng.shuffle(want)
    # de-dup by key so all 100 are distinct cold extractions
    picked, keys = [], set()
    for qid, p in want:
        k = real.key_for(p.text)
        if k in keys:
            continue
        keys.add(k)
        picked.append((qid, p))
        if len(picked) >= N_PARAGRAPHS:
            break
    print(f"  drew {len(picked)} uncached paragraphs from {len({q for q,_ in picked})} "
          f"train questions (scanned {seen_q})")

    cache_path = ROOT / "data" / "cache" / "triples_m7_indexcost.jsonl"
    cache_path.unlink(missing_ok=True)
    cache = TripleCache(cache_path, tag, pv + "-m7cost")
    cli = OllamaClient(tag, num_ctx=int(ex_cfg["num_ctx"]),
                       temperature=float(ex_cfg["temperature"]))
    cli.assert_model_present()

    # warm the model so the one-time load is not charged to per-paragraph latency
    cli.generate("ready?", num_predict=1)
    print(f"  model loaded and warm; swap now {swap_used_mb():.0f} MB")

    per_s, toks, prompt_toks, eval_toks, ntri, fails = [], [], [], [], [], 0
    swap0 = swap_used_mb()
    with StageProfiler("indexing", interval_s=float(cfg["profiling"]["rss_interval_s"]),
                       swap_invalidate_mb=float(cfg["profiling"]["swap_invalidate_mb"])) as prof:
        for i, (qid, p) in enumerate(picked, 1):
            with prof.item():
                tr = extract_paragraph(p, cache, cli,
                                       num_predict=int(ex_cfg["num_predict"]),
                                       max_words=int(ex_cfg["max_words_per_paragraph"]),
                                       retries=int(ex_cfg["retries"]),
                                       max_triples=int(ex_cfg["max_triples"]))
            m = cache._meta[cache.key_for(p.text)]
            per_s.append(prof.per_item_s[-1])
            toks.append(int(m.get("llm_tokens", 0)))
            eval_toks.append(int(m.get("eval_count", 0)))
            ntri.append(len(tr))
            fails += bool(m.get("json_failed"))
            if i % 25 == 0:
                print(f"    {i}/{len(picked)}  t_p={statistics.fmean(per_s):.2f}s  "
                      f"swap {swap_used_mb()-swap0:+.0f} MB", flush=True)
    s = prof.stats
    lat = prof.latency_summary()
    cache.close()

    idx = {
        "n_paragraphs": len(picked),
        "n_questions_drawn_from": len({q for q, _ in picked}),
        "t_p_mean_s": statistics.fmean(per_s),
        "t_p_median_s": lat["median_s"], "t_p_p95_s": lat["p95_s"],
        "llm_tokens_per_paragraph_mean": statistics.fmean(toks),
        "llm_tokens_total": sum(toks),
        "eval_tokens_per_paragraph_mean": statistics.fmean(eval_toks),
        "triples_per_paragraph_mean": statistics.fmean(ntri),
        "json_failures": fails,
        "peak_tree_rss_mb_incl_ollama": s.peak_rss_mb,
        "idle_baseline_rss_mb": s.baseline_rss_mb,
        "delta_rss_mb": s.delta_rss_mb,
        "swap_growth_mb": s.swap_delta_mb,
        "stage_invalidated_by_swap": prof.invalidated,
        "wall_s": s.latency_s,
    }
    print(f"\n  EXTRACTION (per paragraph)")
    print(f"    seconds           : mean {idx['t_p_mean_s']:.3f}  median {idx['t_p_median_s']:.3f}  "
          f"p95 {idx['t_p_p95_s']:.3f}")
    print(f"    LLM tokens        : {idx['llm_tokens_per_paragraph_mean']:.0f} "
          f"(of which {idx['eval_tokens_per_paragraph_mean']:.0f} generated)")
    print(f"    triples           : {idx['triples_per_paragraph_mean']:.2f}")
    print(f"    JSON failures     : {fails}/{len(picked)}")
    print(f"    peak tree RSS     : {idx['peak_tree_rss_mb_incl_ollama']:.0f} MB "
          f"(idle baseline {idx['idle_baseline_rss_mb']:.0f}, delta {idx['delta_rss_mb']:+.0f})")
    print(f"    swap growth       : {idx['swap_growth_mb']:+.0f} MB "
          f"-> stage {'INVALID' if prof.invalidated else 'valid'} under §7.3")

    print(f"\n  unloading the model before the embedding measurement (§6.1)")
    unload_all()
    time.sleep(1.0)

    # ---- entity/relation embedding, per question -------------------------
    ents, rels = set(), set()
    for _qid, p in picked:
        for t in cache.get(p.text) or []:
            for n in (normalize_entity(t.head), normalize_entity(t.tail)):
                if n:
                    ents.add(n)
            r = " ".join(str(t.rel).split())
            if r:
                rels.add(r)
    nq = len({q for q, _ in picked})
    ec = EmbeddingCache(ROOT / "data" / "cache" / "embeddings_m7cost", env("EMBED_MODEL"))
    emb = Embedder(env("EMBED_MODEL"), device=cfg.get("device") or "cpu", cache=ec,
                   batch_size=128)
    emb.encode(["warm"], kind="misc")      # exclude model load from the timing
    with StageProfiler("indexing_embed", interval_s=0.1) as ep:
        t0 = time.perf_counter()
        emb.encode(sorted(ents), kind="entities")
        emb.encode(sorted(rels), kind="relations")
        dt = time.perf_counter() - t0
    emb.free()
    idx["embedding"] = {
        "unique_entities": len(ents), "unique_relations": len(rels),
        "n_questions": nq, "seconds_total": dt,
        "seconds_per_question": dt / nq,
        "entities_per_question": len(ents) / nq,
        "relations_per_question": len(rels) / nq,
        "peak_tree_rss_mb": ep.stats.peak_rss_mb,
        "delta_rss_mb": ep.stats.delta_rss_mb,
        "swap_growth_mb": ep.stats.swap_delta_mb,
    }
    e = idx["embedding"]
    print(f"\n  ENTITY/RELATION EMBEDDING")
    print(f"    {e['unique_entities']} entities + {e['unique_relations']} relations "
          f"from {nq} questions")
    print(f"    {e['seconds_total']:.2f}s total -> {e['seconds_per_question']*1000:.1f} ms/question")
    print(f"    peak tree RSS {e['peak_tree_rss_mb']:.0f} MB (delta {e['delta_rss_mb']:+.0f}), "
          f"swap {e['swap_growth_mb']:+.0f} MB")

    idx["vm_snapshot_end"] = virtual_memory_snapshot()
    (ROOT / "results" / "m7_indexing_cost.json").write_text(json.dumps(idx, indent=2))
    print(f"\n-> results/m7_indexing_cost.json")
    print(f"  (replaces the D11-voided M4 timings)")
    return idx


if __name__ == "__main__":
    run(load_config())
