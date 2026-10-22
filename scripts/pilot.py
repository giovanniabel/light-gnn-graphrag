"""M2: throughput pilot + budget governor (agent.md Section 8.2).

1. Pilot extraction on 50 random pool paragraphs; measure t_p and the JSON
   failure rate.
2. N = clamp(floor(EXTRACT_BUDGET_S / (4 * POOL_SIZE * t_p)), 60, 300).
3. If t_p > 6 s or JSON failure > 5% after one prompt fix: stop and report.
4. Write the chosen N values and measured throughput to results/budget.json.

The NUM_PARALLEL=2 option (Section 2) is probed by issuing two concurrent
requests and comparing throughput, rather than restarting the user's Ollama
server; the result is recorded either way.
"""

from __future__ import annotations

import json
import math
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT, env, load_config  # noqa: E402
from src.extractor import (  # noqa: E402
    PROMPT_PREFIX,
    PROMPT_VERSION,
    SYSTEM,
    TripleCache,
    extract_paragraph,
    parse_triples,
    truncate_words,
)
from src.ollama_client import OllamaClient  # noqa: E402
from src.profiling import StageProfiler  # noqa: E402

POOL_CANDIDATES = (10, 12, 14, 16, 20)


def sample_pilot_paragraphs(cfg, n: int, seed: int) -> list:
    """n random paragraphs drawn from the M1 pools, across both datasets."""
    from scripts.m1_data import load_pools
    from src.data import pool_size_for

    paras = []
    for ds in cfg["data"]["datasets"]:
        pool_size = pool_size_for(cfg, ds)
        for split in ("train", "validation"):
            exs = load_pools(ds, split, pool_size)
            if not exs:
                raise SystemExit(
                    f"no cached pool for {ds}/{split} (pool_size={pool_size}); run M1 first"
                )
            for ex in exs:
                paras.extend(ex.paragraphs)
    rng = random.Random(seed)
    rng.shuffle(paras)
    # unique paragraph texts only -- extraction dedupes by text (Section 6.2)
    seen, out = set(), []
    for p in paras:
        k = truncate_words(p.text)
        if k in seen:
            continue
        seen.add(k)
        out.append(p)
        if len(out) >= n:
            break
    return out


# Measured paragraph-dedupe fractions (Section 6.2 lever), from the n=150 pools.
DEDUPE = {"musique": 0.152, "hotpotqa": 0.010}


def governor_n(
    t_p: float,
    pool_sizes: dict[str, int],
    budget_s: float,
    lo: int,
    hi: int,
    max_hours: float | None = None,
) -> dict:
    """Budget governor, generalized to per-dataset pool sizes (deviations D1/D3).

    agent.md Section 8.2 assumed one POOL_SIZE shared by both datasets:
        N = clamp(floor(EXTRACT_BUDGET_S / (4 * POOL_SIZE * t_p)), 60, 300)
    where 4 = 2 datasets x 2 splits. With a per-dataset pool size the paragraph
    count per question differs by dataset, so the 4*POOL_SIZE term becomes a sum
    over datasets, times the 2 splits:
        N = clamp(floor(EXTRACT_BUDGET_S / (2 * sum_d(pool_size_d) * t_p)), 60, 300)
    This reduces to the original when both pool sizes are equal.
    """
    per_question = sum(pool_sizes.values())          # paragraphs per question, summed over datasets
    denom = 2.0 * per_question * t_p                 # 2 splits
    raw = budget_s / denom if t_p > 0 else float("inf")
    n = int(math.floor(raw))
    clamped = max(lo, min(hi, n))

    total = 2 * per_question * clamped
    # dedupe-adjusted: the same paragraph text is extracted once (Section 6.2)
    unique = sum(
        2 * ps * clamped * (1.0 - DEDUPE.get(ds, 0.0)) for ds, ps in pool_sizes.items()
    )
    est_s = total * t_p
    est_h = est_s / 3600.0
    est_h_dedup = unique * t_p / 3600.0
    return {
        "pool_sizes": dict(pool_sizes),
        "paragraphs_per_question_summed": per_question,
        "n_raw": n,
        "n": clamped,
        "clamped_at": "min" if n < lo else ("max" if n > hi else None),
        "total_paragraphs": total,
        "unique_paragraphs_est": round(unique),
        "estimated_extraction_s": est_s,
        "estimated_extraction_h": est_h,
        "estimated_extraction_h_deduped": est_h_dedup,
        "within_budget": est_s <= budget_s,
        "within_hard_cap": (est_h <= max_hours) if max_hours else None,
        "per_dataset_paragraphs": {
            ds: 2 * ps * clamped for ds, ps in pool_sizes.items()
        },
    }


def test_parallel_2(client: OllamaClient, paragraphs: list, cfg) -> dict:
    """Probe whether 2 concurrent requests beat sequential (Section 2 'pilot may test 2').

    The server's OLLAMA_NUM_PARALLEL is set at launch, so this measures the
    throughput actually available without restarting the user's Ollama server.
    """
    ex_cfg = cfg["extraction"]
    prompts = [
        PROMPT_PREFIX + f"{p.title}. {truncate_words(p.text, int(ex_cfg['max_words_per_paragraph']))}" + "\nJSON:"
        for p in paragraphs
    ]

    def one(pr):
        t = time.perf_counter()
        out, usage, _ = client.generate(pr, system=SYSTEM, fmt="json",
                                        num_predict=int(ex_cfg["num_predict"]))
        return time.perf_counter() - t, usage.total, out

    t0 = time.perf_counter()
    seq = [one(pr) for pr in prompts]
    seq_wall = time.perf_counter() - t0

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        par = list(pool.map(one, prompts))
    par_wall = time.perf_counter() - t0

    n = len(prompts)
    return {
        "n_paragraphs": n,
        "sequential_wall_s": seq_wall,
        "sequential_t_p": seq_wall / n,
        "parallel2_wall_s": par_wall,
        "parallel2_t_p": par_wall / n,
        "speedup": seq_wall / par_wall if par_wall else None,
        "parallel2_faster": par_wall < seq_wall * 0.95,
        "note": (
            "server-side OLLAMA_NUM_PARALLEL was not changed; this measures the "
            "concurrency actually available from the running server"
        ),
    }


def run(cfg, args) -> int:
    seed = int(cfg["seed"])
    b = cfg["budget"]
    ex_cfg = cfg["extraction"]
    from src.data import pool_size_for

    pool_sizes = {ds: pool_size_for(cfg, ds) for ds in cfg["data"]["datasets"]}
    n_pilot = args.n or int(b["pilot_paragraphs"])
    model_tag = env("SLM_TAG")

    print(f"--- pilot: {n_pilot} random pool paragraphs | model {model_tag} ---")
    paragraphs = sample_pilot_paragraphs(cfg, n_pilot, seed)
    by_ds = {}
    for p in paragraphs:
        ds = "musique" if "hop" in p.pid.split("::")[0] else "hotpotqa"
        by_ds[ds] = by_ds.get(ds, 0) + 1
    words = [len(p.text.split()) for p in paragraphs]
    print(f"  drawn: {len(paragraphs)} unique paragraphs {by_ds}")
    print(f"  words/paragraph: median {statistics.median(words):.0f}, "
          f"max {max(words)} (truncated to {ex_cfg['max_words_per_paragraph']})")

    # Pilot writes to its own cache file so it never pollutes the M4 cache.
    mt = int(ex_cfg["max_triples"])
    pv = f"{ex_cfg['prompt_version']}-mt{mt}"
    pilot_cache_path = ROOT / "data" / "cache" / f"triples_pilot_mt{mt}.jsonl"
    if not args.resume and pilot_cache_path.exists():
        pilot_cache_path.unlink()
    cache = TripleCache(pilot_cache_path, model_tag, pv)
    client = OllamaClient(model_tag, num_ctx=int(ex_cfg["num_ctx"]),
                          temperature=float(ex_cfg["temperature"]))
    client.assert_model_present()

    print(f"\n--- sequential extraction (format=json, num_predict={ex_cfg['num_predict']}, "
          f"num_ctx={ex_cfg['num_ctx']}, temp={ex_cfg['temperature']}) ---")
    per_para_s, n_triples, failures, tokens = [], [], 0, 0
    salvaged = truncated = 0
    with StageProfiler("indexing_pilot",
                       interval_s=float(cfg["profiling"]["rss_interval_s"]),
                       swap_invalidate_mb=float(cfg["profiling"]["swap_invalidate_mb"])) as prof:
        for i, p in enumerate(paragraphs, 1):
            with prof.item():
                tr = extract_paragraph(
                    p, cache, client,
                    num_predict=int(ex_cfg["num_predict"]),
                    max_words=int(ex_cfg["max_words_per_paragraph"]),
                    retries=int(ex_cfg["retries"]),
                    max_triples=mt,
                )
            meta = cache._meta[cache.key_for(p.text)]
            per_para_s.append(prof.per_item_s[-1])
            n_triples.append(len(tr))
            tokens += int(meta.get("llm_tokens", 0))
            if meta.get("json_failed"):
                failures += 1
            if meta.get("salvaged"):
                salvaged += 1
            if meta.get("truncated"):
                truncated += 1
            if i % 10 == 0:
                print(f"  {i}/{len(paragraphs)}  t_p={statistics.fmean(per_para_s):.2f}s  "
                      f"triples/para={statistics.fmean(n_triples):.1f}  failures={failures}", flush=True)
    cache.close()

    t_p = statistics.fmean(per_para_s)
    fail_rate = failures / len(paragraphs)
    # strict reading of Section 6.2: a truncated response counts as a JSON failure
    # even though salvage recovered its leading triples. Both are reported.
    strict_failures = failures + salvaged
    strict_fail_rate = strict_failures / len(paragraphs)
    lat = prof.latency_summary()
    s = prof.stats

    print(f"\n--- measured ---")
    print(f"  t_p (mean)            : {t_p:.3f} s/paragraph")
    print(f"  t_p (median / p95)    : {lat['median_s']:.3f} / {lat['p95_s']:.3f} s")
    print(f"  JSON failure rate     : {fail_rate:.2%} ({failures}/{len(paragraphs)}) "
          f"-- unparseable even after salvage")
    print(f"    hit num_predict cap  : {truncated}/{len(paragraphs)} ({truncated/len(paragraphs):.1%})")
    print(f"    salvaged from cut-off: {salvaged}/{len(paragraphs)} "
          f"(complete leading triples recovered)")
    print(f"    STRICT rate (Sec 6.2): {strict_fail_rate:.2%} "
          f"({strict_failures}/{len(paragraphs)}) -- counts salvaged as failures")
    print(f"  triples per paragraph : mean {statistics.fmean(n_triples):.2f}, "
          f"median {statistics.median(n_triples):.0f}, zero-triple {sum(1 for x in n_triples if x==0)}")
    print(f"  LLM tokens            : {tokens} total, {tokens/len(paragraphs):.0f}/paragraph")
    print(f"  peak RSS (tree+ollama): {s.peak_rss_mb:.0f} MB "
          f"(baseline {s.baseline_rss_mb:.0f} MB, delta {s.delta_rss_mb:+.0f} MB)")
    print(f"  swap delta            : {s.swap_delta_mb:+.0f} MB "
          f"(invalidates run above {cfg['profiling']['swap_invalidate_mb']} MB)")
    print(f"  cache lines           : {cache.n_lines} (paragraphs: {len(paragraphs)})")

    # --- NUM_PARALLEL=2 probe --------------------------------------------
    print(f"\n--- NUM_PARALLEL=2 probe (6 paragraphs, sequential vs 2 concurrent) ---")
    try:
        par = test_parallel_2(client, paragraphs[:6], cfg)
        print(f"  sequential : {par['sequential_wall_s']:.1f}s wall, {par['sequential_t_p']:.2f} s/para")
        print(f"  2 threads  : {par['parallel2_wall_s']:.1f}s wall, {par['parallel2_t_p']:.2f} s/para")
        print(f"  speedup    : {par['speedup']:.2f}x -> "
              f"{'USE 2' if par['parallel2_faster'] else 'stay sequential (Section 6.2)'}")
    except Exception as exc:
        par = {"error": repr(exc)}
        print(f"  probe failed: {exc!r}")

    # --- gates (Section 8.2 step 3, as amended by D3) ---------------------
    t_p_gate = t_p <= float(b["t_p_stop_s"])
    json_gate = fail_rate <= float(ex_cfg["json_failure_gate"])
    json_gate_strict = strict_fail_rate <= float(ex_cfg["json_failure_gate"])
    print(f"\n--- gates ---")
    print(f"  [----] t_p {t_p:.3f}s (reference only; {b['t_p_stop_s']}s is no longer a gate -- D3)")
    print(f"  [{'PASS' if json_gate else 'FAIL'}] JSON failure {fail_rate:.2%} <= "
          f"{float(ex_cfg['json_failure_gate']):.0%}  (after salvage)")
    print(f"  [{'PASS' if json_gate_strict else 'FAIL'}] JSON failure {strict_fail_rate:.2%} <= "
          f"{float(ex_cfg['json_failure_gate']):.0%}  (strict: salvaged counted as failures)")


    # --- governor ---------------------------------------------------------
    budget_s = float(b["extract_budget_s"])
    max_h = float(b["max_estimated_hours"])
    lo, hi = int(cfg["data"]["n_min"]), int(cfg["data"]["n_max"])
    per_q = sum(pool_sizes.values())
    print(f"\n--- budget governor (EXTRACT_BUDGET_S={budget_s:.0f}s = {budget_s/3600:.1f}h, "
          f"clamp [{lo}, {hi}], hard cap {max_h}h) ---")
    print(f"  pool sizes: {pool_sizes}  (sum per question = {per_q})")
    print(f"  N = clamp(floor({budget_s:.0f} / (2 * {per_q} * {t_p:.3f})), {lo}, {hi})   [D1/D3]")

    chosen = governor_n(t_p, pool_sizes, budget_s, lo, hi, max_h)
    sizes = {"chosen": chosen}
    print(f"\n  N              = {chosen['n']} per (dataset, split)"
          f"{'  (clamped at ' + chosen['clamped_at'] + ')' if chosen['clamped_at'] else ''}")
    print(f"  N_raw          = {chosen['n_raw']}")
    print(f"  paragraphs     = {chosen['total_paragraphs']} nominal, "
          f"~{chosen['unique_paragraphs_est']} unique after dedupe")
    for ds, c in chosen["per_dataset_paragraphs"].items():
        print(f"      {ds:<9} {c} paragraphs ({pool_sizes[ds]}/question x {chosen['n']} x 2 splits)")
    print(f"  estimated time = {chosen['estimated_extraction_h']:.2f} h nominal, "
          f"{chosen['estimated_extraction_h_deduped']:.2f} h after dedupe")

    # sensitivity: what other MuSiQue pool sizes would cost
    print(f"\n  sensitivity (MuSiQue pool size, HotpotQA fixed at {pool_sizes['hotpotqa']}):")
    print(f"  {'musique':>8} {'N':>5} {'paragraphs':>11} {'est. h':>8} {'<=5h':>6}")
    for ps in POOL_CANDIDATES:
        g = governor_n(t_p, {**pool_sizes, "musique": ps}, budget_s, lo, hi, max_h)
        sizes[f"musique_{ps}"] = g
        print(f"  {ps:>8} {g['n']:>5} {g['total_paragraphs']:>11} "
              f"{g['estimated_extraction_h']:>8.2f} {'yes' if g['within_hard_cap'] else 'NO':>6}")

    budget = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_tag": model_tag,
        "prompt_version": pv,
        "max_triples": mt,
        "pilot": {
            "n_paragraphs": len(paragraphs),
            "by_dataset": by_ds,
            "t_p_mean_s": t_p,
            "t_p_median_s": lat["median_s"],
            "t_p_p95_s": lat["p95_s"],
            "json_failures": failures,
            "json_failure_rate": fail_rate,
            "truncated_at_num_predict": truncated,
            "truncation_rate": truncated / len(paragraphs),
            "salvaged_from_truncation": salvaged,
            "json_failures_strict": strict_failures,
            "json_failure_rate_strict": strict_fail_rate,
            "triples_per_paragraph_mean": statistics.fmean(n_triples),
            "triples_per_paragraph_median": statistics.median(n_triples),
            "zero_triple_paragraphs": sum(1 for x in n_triples if x == 0),
            "llm_tokens_total": tokens,
            "llm_tokens_per_paragraph": tokens / len(paragraphs),
            "words_per_paragraph_median": statistics.median(words),
            "cache_lines": cache.n_lines,
            "stage_stats": s.as_dict(),
            "run_invalidated_by_swap": prof.invalidated,
        },
        "num_parallel_2_test": par,
        "gates": {
            "estimated_hours_le_cap": chosen["within_hard_cap"],
            "estimated_hours": chosen["estimated_extraction_h"],
            "max_estimated_hours": max_h,
            "t_p_le_6s_reference_only": t_p_gate,
            "t_p_measured_s": t_p,
            "t_p_threshold_s": float(b["t_p_stop_s"]),
            "json_failure_le_5pct": json_gate,
            "json_failure_le_5pct_strict": json_gate_strict,
            "all_passed": bool(chosen["within_hard_cap"]) and json_gate,
            "all_passed_strict": bool(chosen["within_hard_cap"]) and json_gate_strict,
        },
        "governor": {
            "extract_budget_s": budget_s,
            "formula": "N = clamp(floor(EXTRACT_BUDGET_S / (2 * sum_d(pool_size_d) * t_p)), 60, 300)  [D1/D3]",
            "n_min": lo,
            "n_max": hi,
            "scenarios": sizes,
            "pool_sizes": pool_sizes,
            "chosen_n": chosen["n"],
            "estimated_extraction_hours": chosen["estimated_extraction_h"],
            "estimated_extraction_hours_deduped": chosen["estimated_extraction_h_deduped"],
            "max_estimated_hours": max_h,
            "sample_sizes": {
                f"{ds}/{split}": chosen["n"]
                for ds in cfg["data"]["datasets"]
                for split in ("train", "validation")
            },
        },
    }

    out = ROOT / "results" / "budget.json"
    out.write_text(json.dumps(budget, indent=2))
    print(f"\nbudget -> {out.relative_to(ROOT)}")

    checks = {
        "results/budget.json written": out.exists(),
        "JSON failure rate <= 5% (after salvage)": json_gate,
        "JSON failure rate <= 5% (strict)": json_gate_strict,
        "estimated extraction hours <= hard cap (D3)": bool(chosen["within_hard_cap"]),
        "NUM_PARALLEL=2 test recorded": "error" not in par,
        "cache lines == paragraphs extracted": cache.n_lines == len(paragraphs),
        "no swap invalidation": not prof.invalidated,
    }
    print("\n=== M2 ACCEPTANCE ===")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    budget["acceptance"] = checks
    budget["all_passed"] = all(checks.values())
    out.write_text(json.dumps(budget, indent=2))
    print(f"M2: {'PASS' if budget['all_passed'] else 'FAIL'}")
    return 0 if budget["all_passed"] else 1
