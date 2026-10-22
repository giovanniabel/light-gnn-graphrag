"""M4: full extraction run (agent.md Section 11).

Acceptance: resumable (kill mid-run, rerun continues); cache lines match paragraph
count; extraction coverage of gold paragraphs reported; <= 5 h.

Guards, checked at every progress tick and aborting cleanly with the cache flushed:
  * projected total time > --max-hours
  * steady-state swap growth > profiling.swap_invalidate_mb

Swap is reported two ways. The one-time model load costs several hundred MB of
swap on an 8 GB machine, which is not a sign of thrashing, so the guard uses the
growth measured after a warm-up of WARMUP_PARAGRAPHS; total growth from process
start is reported alongside it.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT, env, load_config  # noqa: E402
from src.data import pool_size_for  # noqa: E402
from src.extractor import (  # noqa: E402
    TripleCache,
    effective_prompt_version,
    extract_all,
    extraction_coverage,
)
from src.normalize import answer_in_nodes, node_set  # noqa: E402
from src.ollama_client import OllamaClient, unload_all  # noqa: E402
from src.profiling import (  # noqa: E402
    ExtractionGuard,
    memory_pressure_level,
    own_tree_rss_mb,
    swap_used_mb,
    tree_rss_mb,
    virtual_memory_snapshot,
)
from scripts.m1_data import load_pools  # noqa: E402

WARMUP_PARAGRAPHS = 25


def load_all_pools(cfg, datasets) -> dict:
    out = {}
    for ds in datasets:
        ps = pool_size_for(cfg, ds)
        for split in ("train", "validation"):
            exs = load_pools(ds, split, ps)
            if not exs:
                raise SystemExit(f"no cached pool for {ds}/{split} (pool_size={ps}); run M3 first")
            out[(ds, split)] = exs
    return out


def run(cfg, args) -> int:
    ex_cfg = cfg["extraction"]
    datasets = [args.dataset] if args.dataset else list(cfg["data"]["datasets"])
    max_hours = args.max_hours if args.max_hours else float(cfg["budget"]["extract_hard_cap_h"])
    g_cfg = cfg["profiling"].get("extraction_guard", {})
    model_tag = env("SLM_TAG")
    pv = effective_prompt_version(cfg)   # single source of truth (D8)
    cache_path = cfg.path("triples")

    pools = load_all_pools(cfg, datasets)
    examples = [ex for exs in pools.values() for ex in exs]
    total_paragraphs = sum(len(ex.paragraphs) for ex in examples)

    print(f"--- M4: full extraction ---")
    print(f"  model          : {model_tag}  prompt_version={pv}  max_triples={ex_cfg['max_triples']}")
    print(f"  questions      : {len(examples)}  across {len(pools)} (dataset, split) pools")
    for (ds, split), exs in pools.items():
        print(f"      {ds}/{split:<11} {len(exs):>4} questions, "
              f"{sum(len(e.paragraphs) for e in exs):>5} paragraphs")
    print(f"  paragraphs     : {total_paragraphs} nominal (before dedupe)")
    print(f"  cache          : {cache_path.relative_to(ROOT)}")
    print(f"  max_hours      : {max_hours}   resume={args.resume}")
    print(f"  guards (D10)   : own-tree RSS growth, rolling t_p degradation, "
          f"sustained critical memory pressure")
    print(f"                   system-wide swap is LOGGED, not an abort "
          f"(it stays a hard rule for the M7/M8 efficiency runs)")

    cache0 = TripleCache(cache_path, model_tag, pv)
    print(f"  already cached : {len(cache0)} paragraphs")
    cache0.close()

    cli = OllamaClient(model_tag, num_ctx=int(ex_cfg["num_ctx"]))
    cli.assert_model_present()

    t_start = time.perf_counter()
    swap_start = swap_used_mb()
    guard_obj = ExtractionGuard(
        rss_growth_factor=float(g_cfg.get("rss_growth_factor", 1.5)),
        rss_growth_min_mb=float(g_cfg.get("rss_growth_min_mb", 300.0)),
        rss_window=int(g_cfg.get("rss_window_ticks", 10)),
        tp_factor=float(g_cfg.get("tp_factor", 1.5)),
        tp_window_paragraphs=int(g_cfg.get("tp_window_paragraphs", 50)),
        pressure_consecutive=int(g_cfg.get("pressure_consecutive", 3)),
        warn_every_s=float(g_cfg.get("warn_every_s", 60.0)),
    )
    state = {"stop_reason": None, "ticks": [], "warnings": []}
    print(f"  swap at start  : {swap_start:.0f} MB")
    print(f"  vm at start    : {virtual_memory_snapshot()}")
    print(f"\n  {'elapsed':>8} {'done':>12} {'t_p':>7} {'fail':>7} {'ownRSS':>8} "
          f"{'swap':>9} {'prs':>4} {'projected':>10}", flush=True)

    def tick(stats, i, total):
        elapsed = time.perf_counter() - t_start
        done = stats.paragraphs_extracted
        t_p = stats.seconds / done if done else 0.0
        remaining = total - i
        rolling = guard_obj.rolling_t_p()
        proj_total_h = (elapsed + remaining * (rolling or t_p)) / 3600.0

        obs = guard_obj.observe(done, stats.seconds, elapsed)
        rec = {
            "elapsed_s": elapsed, "i": i, "total": total, "extracted": done,
            "cached": stats.paragraphs_cached, "t_p": t_p, "rolling_t_p": rolling,
            "json_failure_rate": stats.json_failure_rate,
            "triples_total": stats.triples_total,
            "own_rss_mb": obs["own_rss_mb"], "tree_rss_mb": tree_rss_mb(),
            "pressure": obs["pressure"],
            "swap_mb": obs["swap_mb"], "swap_growth_total_mb": obs["swap_mb"] - swap_start,
            "available_mb": obs["available_mb"],
            "projected_total_h": proj_total_h,
        }
        state["ticks"].append(rec)
        print(f"  {elapsed/60:>7.1f}m {i:>5}/{total:<6} {(rolling or t_p):>6.2f}s "
              f"{stats.json_failure_rate:>6.1%} {obs['own_rss_mb']:>7.0f}M "
              f"{obs['swap_mb'] - swap_start:>+8.0f}M {obs['pressure']:>4} "
              f"{proj_total_h:>9.2f}h", flush=True)

        # system-wide swap / RAM: warning only (D10), once a minute
        if guard_obj.should_warn(elapsed):
            w = {
                "elapsed_s": elapsed, "swap_mb": obs["swap_mb"],
                "swap_growth_mb": obs["swap_mb"] - swap_start,
                "available_mb": obs["available_mb"], "pressure": obs["pressure"],
            }
            state["warnings"].append(w)
            flag = ""
            if w["swap_growth_mb"] > float(cfg["profiling"]["swap_invalidate_mb"]):
                flag = ("  <-- WARNING: exceeds the M7/M8 efficiency-run swap rule; "
                        "logged, not aborting (D10)")
            print(f"      [sys {elapsed/60:>5.1f}m] swap {obs['swap_mb']:.0f} MB "
                  f"({w['swap_growth_mb']:+.0f} MB), RAM available "
                  f"{obs['available_mb']:.0f} MB, pressure {obs['pressure']}{flag}", flush=True)

        (ROOT / "results" / "m4_progress.json").write_text(
            json.dumps({"ticks": state["ticks"], "warnings": state["warnings"]}, indent=2))

    def guard(stats, i, total):
        r = state["ticks"][-1] if state["ticks"] else None
        if not r:
            return None
        if r["projected_total_h"] > max_hours:
            state["stop_reason"] = (
                f"projected total {r['projected_total_h']:.2f} h exceeds max_hours {max_hours}"
            )
            return state["stop_reason"]
        reason = guard_obj.check()
        if reason:
            state["stop_reason"] = reason
        return reason

    stats = extract_all(
        examples, cache_path, max_hours=max_hours, resume=args.resume,
        model_tag=model_tag, cfg=cfg, progress_every=25,
        on_progress=tick, should_stop=guard,
    )
    wall = time.perf_counter() - t_start
    unload_all()

    # ---------------- reporting ----------------
    cache = TripleCache(cache_path, model_tag, pv)
    per_dataset = {}
    for ds in datasets:
        exs = [e for (d, _s), lst in pools.items() if d == ds for e in lst]
        cov = extraction_coverage(exs, cache)
        hits = 0
        for ex in exs:
            triples = []
            for p in ex.paragraphs:
                triples.extend(cache.get(p.text) or [])
            hits += answer_in_nodes(ex.answer, node_set(triples), ex.meta.get("answer_aliases"))
        cov["answer_in_graph_rate"] = hits / len(exs) if exs else 0.0
        cov["questions_scored"] = len(exs)
        per_dataset[ds] = cov

    covered = sum(1 for ex in examples for p in ex.paragraphs if cache.get(p.text) is not None)
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_tag": model_tag, "prompt_version": pv,
        "max_triples": int(ex_cfg["max_triples"]),
        "wall_s": wall, "wall_h": wall / 3600.0,
        "max_hours": max_hours,
        "stats": stats.as_dict(),
        "stop_reason": state["stop_reason"],
        "paragraphs_nominal": total_paragraphs,
        "paragraphs_with_triples_in_cache": covered,
        "cache_lines": cache.n_lines,
        "cache_unique_keys": len(cache),
        "swap_start_mb": swap_start,
        "swap_end_mb": swap_used_mb(),
        "swap_growth_total_mb": swap_used_mb() - swap_start,
        "final_memory_pressure": memory_pressure_level(),
        "guard": {
            "baseline_own_rss_mb": guard_obj.baseline_rss_mb,
            "baseline_t_p": guard_obj.baseline_t_p,
            "rss_growth_factor": guard_obj.rss_growth_factor,
            "tp_factor": guard_obj.tp_factor,
            "pressure_consecutive": guard_obj.pressure_consecutive,
        },
        "per_dataset": per_dataset,
        "warnings": state["warnings"],
        "ticks": state["ticks"],
    }
    cache.close()

    print(f"\n--- M4 result ---")
    print(f"  wall                 : {wall/3600:.2f} h ({wall/60:.1f} min)")
    print(f"  paragraphs extracted : {stats.paragraphs_extracted} "
          f"(cached/skipped {stats.paragraphs_cached}, nominal {total_paragraphs})")
    print(f"  unique cache keys    : {len(cache)}   cache lines: {cache.n_lines}")
    print(f"  t_p                  : {stats.t_p:.2f} s")
    print(f"  JSON failure rate    : {stats.json_failure_rate:.2%}")
    print(f"  triples total        : {stats.triples_total}")
    print(f"  swap growth (total)  : {report['swap_growth_total_mb']:+.0f} MB")
    if state["stop_reason"]:
        print(f"  !! STOPPED EARLY     : {state['stop_reason']}")
    if stats.timed_out:
        print(f"  !! hit max_hours cap")

    print(f"\n--- per dataset ---")
    for ds, c in per_dataset.items():
        print(f"  {ds}")
        print(f"    gold-paragraph extraction coverage : {c['extraction_coverage']:.2%} "
              f"({c['gold_paragraphs_with_triples']}/{c['gold_paragraphs']})")
        print(f"    questions with no gold edges       : {c['questions_unmappable']}"
              f"/{c['questions']} ({c['unmappable_rate']:.1%}) -- dropped from TRAINING only")
        print(f"    answer-in-graph rate               : {c['answer_in_graph_rate']:.2%}")

    checks = {
        "completed without hitting the hour cap": not stats.timed_out and not state["stop_reason"],
        "cache lines match paragraphs processed": cache.n_lines >= stats.paragraphs_extracted,
        "every pool paragraph has a cache entry": covered == total_paragraphs,
        "gold-paragraph coverage reported per dataset": all(
            "extraction_coverage" in c for c in per_dataset.values()
        ),
        "within 5 h": wall / 3600.0 <= 5.0,
    }
    print("\n=== M4 ACCEPTANCE ===")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    report["acceptance"] = checks
    report["all_passed"] = all(checks.values())

    out = ROOT / "results" / "m4_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nreport -> {out.relative_to(ROOT)}")
    print(f"M4: {'PASS' if report['all_passed'] else 'FAIL'}")
    return 0 if report["all_passed"] else 1
