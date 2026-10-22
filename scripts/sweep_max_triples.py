"""max_triples sweep: sustained throughput + extraction quality (steps 1-2).

For each max_triples value:
  (1) the standard 50-paragraph pilot sample, run sustained and sequentially, with
      t_p, generation tok/s, failure/truncation/salvage rates and swap growth;
  (2) every pool paragraph of a FIXED set of 20 MuSiQue train questions, giving
      triples per paragraph and the answer-in-graph rate (normalized gold answer
      exactly matches some node string).

Each value gets its own cache file: the Section 8.1 cache key is
sha1(model_tag + prompt_version + paragraph_text) and does not include
max_triples, so sharing one file would serve stale triples across configs.

The quality check holds POOL_SIZE at 10 for all three values so the comparison is
apples-to-apples; a larger MuSiQue pool only adds paragraphs and lifts all three
rates together.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT, env, load_config  # noqa: E402
from src.extractor import TripleCache, extract_paragraph, truncate_words  # noqa: E402
from src.normalize import answer_in_nodes, node_set  # noqa: E402
from src.ollama_client import OllamaClient, unload_all  # noqa: E402
from src.profiling import StageProfiler, swap_used_mb, virtual_memory_snapshot  # noqa: E402
from scripts.m1_data import load_pools  # noqa: E402
from scripts.pilot import sample_pilot_paragraphs  # noqa: E402

VALUES = (6, 5, 4)
N_QUALITY_QUESTIONS = 20


def run_one(mt: int, pilot_paras, quality_exs, cfg) -> dict:
    ex_cfg = cfg["extraction"]
    model_tag = env("SLM_TAG")
    # prompt_version encodes max_triples so cache keys never collide across configs
    pv = f"{ex_cfg['prompt_version']}-mt{mt}"
    cache_path = ROOT / "data" / "cache" / f"triples_sweep_mt{mt}.jsonl"
    cache = TripleCache(cache_path, model_tag, pv)
    client = OllamaClient(model_tag, num_ctx=int(ex_cfg["num_ctx"]),
                          temperature=float(ex_cfg["temperature"]))

    def extract(paras, label):
        rec = []
        with StageProfiler(f"indexing_mt{mt}_{label}",
                           interval_s=float(cfg["profiling"]["rss_interval_s"]),
                           swap_invalidate_mb=float(cfg["profiling"]["swap_invalidate_mb"])) as prof:
            for i, p in enumerate(paras, 1):
                with prof.item():
                    tr = extract_paragraph(
                        p, cache, client,
                        num_predict=int(ex_cfg["num_predict"]),
                        max_words=int(ex_cfg["max_words_per_paragraph"]),
                        retries=int(ex_cfg["retries"]),
                        max_triples=mt,
                    )
                m = cache._meta[cache.key_for(p.text)]
                rec.append({
                    "pid": p.pid, "n_triples": len(tr),
                    "failed": bool(m.get("json_failed")), "salvaged": bool(m.get("salvaged")),
                    "truncated": bool(m.get("truncated")),
                    "eval_count": int(m.get("eval_count", 0)),
                    "eval_duration_s": float(m.get("eval_duration_s", 0.0)),
                    "llm_tokens": int(m.get("llm_tokens", 0)),
                    "s": prof.per_item_s[-1],
                })
                if i % 25 == 0:
                    print(f"    mt={mt} {label} {i}/{len(paras)} "
                          f"t_p={statistics.fmean(r['s'] for r in rec):.2f}s", flush=True)
        return rec, prof

    print(f"\n### max_triples={mt} | prompt_version={pv}", flush=True)
    print(f"  [1/2] sustained pilot: {len(pilot_paras)} paragraphs", flush=True)
    swap0 = swap_used_mb()
    prec, pprof = extract(pilot_paras, "pilot")

    # --- throughput on the pilot sample ---------------------------------
    ts = [r["s"] for r in prec]
    ev = sum(r["eval_count"] for r in prec)
    evd = sum(r["eval_duration_s"] for r in prec)
    n = len(prec)
    fails = sum(r["failed"] for r in prec)
    salv = sum(r["salvaged"] for r in prec)
    trunc = sum(r["truncated"] for r in prec)
    pilot = {
        "n_paragraphs": n,
        "t_p_mean_s": statistics.fmean(ts),
        "t_p_median_s": statistics.median(ts),
        "t_p_p95_s": sorted(ts)[min(n - 1, int(round(0.95 * (n - 1))))],
        "gen_tok_s": ev / evd if evd else None,
        "eval_tokens_mean": ev / n,
        "llm_tokens_total": sum(r["llm_tokens"] for r in prec),
        "json_failures": fails,
        "json_failure_rate": fails / n,
        "truncated": trunc,
        "truncation_rate": trunc / n,
        "salvaged": salv,
        "json_failure_rate_strict": (fails + salv) / n,
        "triples_per_paragraph_mean": statistics.fmean(r["n_triples"] for r in prec),
        "zero_triple_paragraphs": sum(1 for r in prec if r["n_triples"] == 0),
        "swap_before_mb": swap0,
        "swap_after_mb": swap_used_mb(),
        "swap_growth_mb": pprof.stats.swap_delta_mb,
        "peak_rss_mb": pprof.stats.peak_rss_mb,
        "baseline_rss_mb": pprof.stats.baseline_rss_mb,
        "run_invalidated_by_swap": pprof.invalidated,
        "wall_s": pprof.stats.latency_s,
    }
    print(f"    -> t_p {pilot['t_p_mean_s']:.2f}s | {pilot['gen_tok_s']:.1f} tok/s | "
          f"fail {pilot['json_failure_rate']:.1%} strict {pilot['json_failure_rate_strict']:.1%} | "
          f"swap +{pilot['swap_growth_mb']:.0f} MB", flush=True)

    # --- quality on the fixed 20 MuSiQue questions ------------------------
    qparas = [p for ex in quality_exs for p in ex.paragraphs]
    print(f"  [2/2] quality: {len(qparas)} paragraphs over {len(quality_exs)} MuSiQue questions",
          flush=True)
    qrec, qprof = extract(qparas, "quality")

    per_q, hits, gold_hits = [], 0, 0
    for ex in quality_exs:
        triples = []
        for p in ex.paragraphs:
            triples.extend(cache.get(p.text) or [])
        nodes = node_set(triples)
        ok = answer_in_nodes(ex.answer, nodes)
        ok_alias = answer_in_nodes(ex.answer, nodes, ex.meta.get("answer_aliases"))
        hits += ok
        gold_hits += ok_alias
        per_q.append({
            "qid": ex.qid, "n_hops": ex.n_hops, "n_triples": len(triples),
            "n_nodes": len(nodes), "answer_in_graph": ok, "answer_or_alias_in_graph": ok_alias,
        })

    nq = len(quality_exs)
    by_hop: dict[str, dict] = {}
    for r in per_q:
        d = by_hop.setdefault(str(r["n_hops"]), {"hit": 0, "n": 0})
        d["hit"] += r["answer_in_graph"]
        d["n"] += 1
    quality = {
        "n_questions": nq,
        "n_paragraphs": len(qparas),
        "triples_per_paragraph_mean": statistics.fmean(r["n_triples"] for r in qrec),
        "triples_per_question_mean": statistics.fmean(r["n_triples"] for r in per_q),
        "nodes_per_question_mean": statistics.fmean(r["n_nodes"] for r in per_q),
        "answer_in_graph_rate": hits / nq,
        "answer_or_alias_in_graph_rate": gold_hits / nq,
        "answer_in_graph_by_hop": {
            k: {"rate": v["hit"] / v["n"], "n": v["n"]} for k, v in sorted(by_hop.items())
        },
        "json_failure_rate": sum(r["failed"] for r in qrec) / len(qrec),
        "per_question": per_q,
    }
    print(f"    -> answer-in-graph {quality['answer_in_graph_rate']:.1%} "
          f"(with aliases {quality['answer_or_alias_in_graph_rate']:.1%}) | "
          f"{quality['triples_per_paragraph_mean']:.2f} triples/para", flush=True)

    cache.close()
    return {
        "max_triples": mt,
        "prompt_version": pv,
        "cache_file": str(cache_path.relative_to(ROOT)),
        "pilot": pilot,
        "quality": quality,
    }


def main() -> int:
    cfg = load_config()
    seed = int(cfg["seed"])
    pool_size = int(cfg["data"]["pool_size"])

    print("=== environment at sweep start ===")
    print(f"  {virtual_memory_snapshot()}")

    pilot_paras = sample_pilot_paragraphs(cfg, int(cfg["budget"]["pilot_paragraphs"]), seed)
    musique_train = load_pools("musique", "train", pool_size)
    if not musique_train:
        raise SystemExit("no cached MuSiQue train pool; run M1 first")
    quality_exs = sorted(musique_train, key=lambda e: e.qid)[:N_QUALITY_QUESTIONS]
    print(f"  pilot sample     : {len(pilot_paras)} paragraphs")
    print(f"  quality questions: {len(quality_exs)} MuSiQue train "
          f"(hops {sorted(e.n_hops for e in quality_exs)})")
    print(f"  quality paragraphs: {sum(len(e.paragraphs) for e in quality_exs)} "
          f"(POOL_SIZE held at {pool_size} for comparability)")

    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_tag": env("SLM_TAG"),
        "baseline": virtual_memory_snapshot(),
        "quality_question_ids": [e.qid for e in quality_exs],
        "runs": {},
    }
    for mt in VALUES:
        out["runs"][str(mt)] = run_one(mt, pilot_paras, quality_exs, cfg)
        p = ROOT / "results" / "sweep_max_triples.json"
        p.write_text(json.dumps(out, indent=2))

    unload_all()
    p = ROOT / "results" / "sweep_max_triples.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nsweep -> {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
