"""M11 step 2: dedupe the new pools against the cache, then time 100 cold paragraphs.

Uses the frozen extractor exactly as configured: same model tag, prompt version
v2-mt6, greedy, JSON schema with 6 triples, 120-word truncation.
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
from src.data import pool_size_for  # noqa: E402
from src.extractor import TripleCache, effective_prompt_version, extract_paragraph  # noqa: E402
from src.ollama_client import OllamaClient, unload_all  # noqa: E402
from src.profiling import StageProfiler, swap_used_mb, virtual_memory_snapshot  # noqa: E402
from src.validation_guard import allow  # noqa: E402
from scripts.m1_data import load_pools  # noqa: E402

N_TIME = 100
MAX_HOURS_GATE = 12.0


def new_paragraphs(cfg, cache):
    """Every (qid, paragraph) in the new pools, split by cache status."""
    cached, todo, seen = [], [], set()
    for ds in cfg["data"]["datasets"]:
        ps = pool_size_for(cfg, ds)
        exs = load_pools(ds, "validation_ext", ps)
        for ex in exs:
            for p in ex.paragraphs:
                k = cache.key_for(p.text)
                if k in cache._index:
                    cached.append((ds, p))
                elif k not in seen:
                    seen.add(k)
                    todo.append((ds, p))
    return cached, todo


def main() -> int:
    cfg = load_config()
    allow("M11 extension: timing check (D27)")
    ex_cfg = cfg["extraction"]
    tag = env("SLM_TAG")
    pv = effective_prompt_version(cfg)
    cache = TripleCache(cfg.path("triples"), tag, pv)

    print(f"--- M11 step 2: dedupe + timing (frozen extractor) ---")
    print(f"  model {tag}, prompt_version {pv}, max_triples {ex_cfg['max_triples']}, "
          f"{ex_cfg['max_words_per_paragraph']}-word truncation, greedy")
    print(f"  existing cache: {len(cache)} paragraphs")
    cached, todo = new_paragraphs(cfg, cache)
    total_slots = len(cached) + len(todo)
    print(f"\n  new pool paragraph slots : {total_slots:,}")
    print(f"    already in the cache    : {len(cached):,} "
          f"({len(cached)/total_slots:.1%}) -- reused, no extraction")
    print(f"    to extract (unique)     : {len(todo):,}")

    print(f"\n  environment: {virtual_memory_snapshot()}")
    print(f"  swap at start: {swap_used_mb():.0f} MB")

    cli = OllamaClient(tag, num_ctx=int(ex_cfg["num_ctx"]),
                       temperature=float(ex_cfg["temperature"]))
    cli.assert_model_present()
    cli.generate("ready?", num_predict=1)       # warm, so load is not charged

    rng = random.Random(20261026)
    sample = rng.sample(todo, min(N_TIME, len(todo)))
    tmp_cache = TripleCache(ROOT / "data" / "cache" / "triples_ext_timing.jsonl",
                            tag, pv + "-exttiming")
    per_s, toks, ntri, fails = [], [], [], 0
    with StageProfiler("indexing_ext_timing",
                       interval_s=float(cfg["profiling"]["rss_interval_s"]),
                       swap_invalidate_mb=float(cfg["profiling"]["swap_invalidate_mb"])) as prof:
        for i, (_ds, p) in enumerate(sample, 1):
            with prof.item():
                tr = extract_paragraph(p, tmp_cache, cli,
                                       num_predict=int(ex_cfg["num_predict"]),
                                       max_words=int(ex_cfg["max_words_per_paragraph"]),
                                       retries=int(ex_cfg["retries"]),
                                       max_triples=int(ex_cfg["max_triples"]))
            m = tmp_cache._meta[tmp_cache.key_for(p.text)]
            per_s.append(prof.per_item_s[-1])
            toks.append(int(m.get("llm_tokens", 0)))
            ntri.append(len(tr))
            fails += bool(m.get("json_failed"))
            if i % 25 == 0:
                print(f"    {i}/{len(sample)}  t_p={statistics.fmean(per_s):.2f}s", flush=True)
    tmp_cache.close()
    unload_all()

    t_p = statistics.fmean(per_s)
    s = prof.stats
    extract_h = len(todo) * t_p / 3600
    # the rest of the pipeline, from the measured M7 figures
    embed_h = (700 * 23.3e-3) / 3600                      # entity/relation embedding
    trace_h = 700 * 1.6 / 3600                            # LLM-Trace, ~1.6 s/question
    reader_h = 700 * 7 * 0.6 / 3600                       # 7 conditions x ~0.6 s
    other_h = 0.2                                          # graphs, scoring, bootstraps
    total_h = extract_h + embed_h + trace_h + reader_h + other_h

    out = {
        "n_timed": len(sample), "t_p_mean_s": t_p,
        "t_p_median_s": statistics.median(per_s),
        "t_p_p95_s": sorted(per_s)[min(len(per_s)-1, int(round(0.95*(len(per_s)-1))))],
        "llm_tokens_per_paragraph": statistics.fmean(toks),
        "triples_per_paragraph": statistics.fmean(ntri),
        "json_failures": fails, "json_failure_rate": fails / len(sample),
        "peak_tree_rss_mb": s.peak_rss_mb, "swap_growth_mb": s.swap_delta_mb,
        "stage_valid": not prof.invalidated,
        "slots_total": total_slots, "slots_cached": len(cached),
        "paragraphs_to_extract": len(todo),
        "projected": {
            "extraction_h": extract_h, "embedding_h": embed_h,
            "llm_trace_h": trace_h, "reader_h": reader_h, "other_h": other_h,
            "total_h": total_h, "gate_h": MAX_HOURS_GATE,
            "within_gate": total_h <= MAX_HOURS_GATE,
        },
    }
    print(f"\n  MEASURED on {len(sample)} cold paragraphs")
    print(f"    t_p            : {t_p:.3f} s (median {out['t_p_median_s']:.3f}, "
          f"p95 {out['t_p_p95_s']:.3f})   [original run: 1.603 s]")
    print(f"    LLM tokens     : {out['llm_tokens_per_paragraph']:.0f}/paragraph  "
          f"[original: 408]")
    print(f"    triples        : {out['triples_per_paragraph']:.2f}/paragraph  "
          f"[original: 5.48]")
    print(f"    JSON failures  : {fails}/{len(sample)} = {out['json_failure_rate']:.2%}  "
          f"[original: 0/100]")
    print(f"    peak tree RSS  : {s.peak_rss_mb:.0f} MB, swap {s.swap_delta_mb:+.0f} MB "
          f"-> {'valid' if not prof.invalidated else 'INVALID'}")
    print(f"\n  PROJECTED TOTAL")
    for k, lab in (("extraction_h", f"extraction of {len(todo):,} paragraphs"),
                   ("embedding_h", "entity/relation embedding"),
                   ("llm_trace_h", "LLM-Trace on 700 questions"),
                   ("reader_h", "reader, 7 conditions x 700"),
                   ("other_h", "graphs, scoring, bootstraps")):
        print(f"    {lab:<38} {out['projected'][k]:>6.2f} h")
    print(f"    {'TOTAL':<38} {total_h:>6.2f} h   (gate {MAX_HOURS_GATE} h -> "
          f"{'WITHIN' if total_h <= MAX_HOURS_GATE else 'EXCEEDS -- STOP AND ASK'})")

    (ROOT / "results" / "ext_timing.json").write_text(json.dumps(out, indent=2))
    print(f"\n-> results/ext_timing.json")
    return 0 if total_h <= MAX_HOURS_GATE else 3


if __name__ == "__main__":
    raise SystemExit(main())
