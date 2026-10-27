"""M11 step 3: chunked, resumable extraction of the new validation pools (D27).

Frozen extractor, extending data/cache/triples.jsonl with the same key scheme, so
paragraphs shared with the original run are reused rather than re-extracted.

Chunks of ~500 paragraphs. The cache flushes after every paragraph, so a crash or
a sleep loses nothing and a re-run continues. Per chunk: if swap grows past the
§7.3 threshold the chunk pauses for a cooldown and is retried once; if it happens
again the chunk is marked invalid (its triples are kept -- greedy decoding makes
them correct regardless -- but its timings are not used).
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT, env, load_config  # noqa: E402
from src.data import pool_size_for  # noqa: E402
from src.extractor import TripleCache, effective_prompt_version, extract_paragraph  # noqa: E402
from src.ollama_client import OllamaClient, unload_all  # noqa: E402
from src.profiling import StageProfiler  # noqa: E402
from src.validation_guard import allow  # noqa: E402
from scripts.m1_data import load_pools  # noqa: E402

CHUNK = 500
COOLDOWN_S = 60
PROGRESS = ROOT / "results" / "ext_extract_progress.json"


def main() -> int:
    cfg = load_config()
    allow("M11 extension: extracting the new validation pools (D27)")
    ex_cfg = cfg["extraction"]
    tag = env("SLM_TAG")
    pv = effective_prompt_version(cfg)
    cache = TripleCache(cfg.path("triples"), tag, pv)

    todo, seen = [], set()
    for ds in cfg["data"]["datasets"]:
        ps = pool_size_for(cfg, ds)
        for ex in load_pools(ds, "validation_ext", ps):
            for p in ex.paragraphs:
                k = cache.key_for(p.text)
                if k in cache._index or k in seen:
                    continue
                seen.add(k)
                todo.append(p)

    print(f"--- M11 step 3: extraction ---")
    print(f"  cache holds {len(cache)} paragraphs; {len(todo):,} new ones to extract")
    if not todo:
        print("  nothing to do -- already complete")
        return 0

    cli = OllamaClient(tag, num_ctx=int(ex_cfg["num_ctx"]),
                       temperature=float(ex_cfg["temperature"]))
    cli.assert_model_present()
    cli.generate("ready?", num_predict=1)

    chunks = [todo[i:i + CHUNK] for i in range(0, len(todo), CHUNK)]
    state = json.loads(PROGRESS.read_text()) if PROGRESS.exists() else {"chunks": []}
    done_idx = {c["chunk"] for c in state["chunks"] if c.get("complete")}
    t0 = time.perf_counter()
    all_s, all_fail, all_n = [], 0, 0

    print(f"  {len(chunks)} chunks of up to {CHUNK}; "
          f"{len(done_idx)} already complete\n")
    for ci, chunk in enumerate(chunks):
        if ci in done_idx:
            continue
        # A chunk is measured exactly once. Extraction is not idempotent in the
        # sense that matters here: once a paragraph is cached, re-running the
        # chunk re-measures nothing (every paragraph is skipped), so a "retry
        # the measurement" pass would silently report an empty chunk. If swap
        # grows past the threshold the triples are still correct (greedy
        # decoding), so they are kept and only the timings are marked invalid;
        # the cooldown then runs before the NEXT chunk, where it can still help.
        per_s, fails, ntri = [], 0, []
        with StageProfiler(f"ext_chunk{ci}",
                           interval_s=float(cfg["profiling"]["rss_interval_s"]),
                           swap_invalidate_mb=float(cfg["profiling"]["swap_invalidate_mb"])) as prof:
            for p in chunk:
                if cache.key_for(p.text) in cache._index:
                    continue
                with prof.item():
                    tr = extract_paragraph(p, cache, cli,
                                           num_predict=int(ex_cfg["num_predict"]),
                                           max_words=int(ex_cfg["max_words_per_paragraph"]),
                                           retries=int(ex_cfg["retries"]),
                                           max_triples=int(ex_cfg["max_triples"]))
                m = cache._meta[cache.key_for(p.text)]
                per_s.append(prof.per_item_s[-1])
                fails += bool(m.get("json_failed"))
                ntri.append(len(tr))
        grew = prof.stats.swap_delta_mb
        if grew > float(cfg["profiling"]["swap_invalidate_mb"]):
            print(f"  chunk {ci:>3}: swap grew {grew:+.0f} MB -> timings INVALID "
                  f"(triples kept); unloading and cooling down {COOLDOWN_S}s",
                  flush=True)
            unload_all()
            time.sleep(COOLDOWN_S)
            cli.generate("ready?", num_predict=1)

        n = len(per_s)
        all_s += per_s
        all_fail += fails
        all_n += n
        rec = {"chunk": ci, "n_extracted": n, "complete": True,
               "t_p": statistics.fmean(per_s) if per_s else None,
               "json_failures": fails,
               "triples_mean": statistics.fmean(ntri) if ntri else None,
               "swap_growth_mb": grew,
               "valid": grew <= float(cfg["profiling"]["swap_invalidate_mb"]),
               "peak_tree_rss_mb": prof.stats.peak_rss_mb}
        state["chunks"] = [c for c in state["chunks"] if c["chunk"] != ci] + [rec]
        PROGRESS.write_text(json.dumps(state, indent=2))

        done = sum(c["n_extracted"] for c in state["chunks"])
        rate = statistics.fmean(all_s) if all_s else 1.8
        remaining = len(todo) - done
        eta_min = remaining * rate / 60
        print(f"  chunk {ci:>3}/{len(chunks)-1}: {n:>3} extracted, "
              f"t_p {rec['t_p'] or 0:.2f}s, fails {fails}, swap {grew:+.0f} MB"
              f"{'' if rec['valid'] else ' (INVALID timings)'} | "
              f"done {done:,}/{len(todo):,}, ETA {eta_min:.0f} min", flush=True)

    cache.close()
    unload_all()
    wall = time.perf_counter() - t0

    # ---- coverage on the NEW questions -------------------------------
    cache = TripleCache(cfg.path("triples"), tag, pv)
    cov = {}
    for ds in cfg["data"]["datasets"]:
        ps = pool_size_for(cfg, ds)
        exs = load_pools(ds, "validation_ext", ps)
        gold_t = gold_w = slots = with_tri = 0
        for ex in exs:
            for p in ex.paragraphs:
                slots += 1
                has = bool(cache.get(p.text))
                with_tri += has
                if p.is_gold:
                    gold_t += 1
                    gold_w += has
        cov[ds] = {"n_questions": len(exs), "paragraph_slots": slots,
                   "slots_with_triples": with_tri,
                   "gold_paragraphs": gold_t, "gold_with_triples": gold_w,
                   "extraction_coverage": gold_w / gold_t if gold_t else None,
                   "paragraph_coverage": with_tri / slots}

    inval = [c["chunk"] for c in state["chunks"] if not c.get("valid")]
    out = {"wall_s": wall, "wall_h": wall / 3600,
           "paragraphs_extracted": all_n,
           "t_p_mean_s": statistics.fmean(all_s) if all_s else None,
           "json_failures": all_fail,
           "json_failure_rate": all_fail / all_n if all_n else None,
           "chunks": state["chunks"], "invalid_timing_chunks": inval,
           "coverage": cov}
    (ROOT / "results" / "ext_extraction.json").write_text(json.dumps(out, indent=2))

    print(f"\n--- extraction complete in {wall/3600:.2f} h ---")
    print(f"  extracted {all_n:,} paragraphs, t_p {out['t_p_mean_s']:.3f} s")
    print(f"  JSON failure rate {out['json_failure_rate']:.4%}   [original run: 0.02%]")
    print(f"  chunks with invalid timings (swap): {len(inval)} of {len(state['chunks'])}")
    for ds, c in cov.items():
        print(f"  {ds}: gold-paragraph extraction coverage "
              f"{c['extraction_coverage']:.4f} ({c['gold_with_triples']}/{c['gold_paragraphs']})"
              f", paragraph coverage {c['paragraph_coverage']:.4f}")
    print(f"\n-> results/ext_extraction.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
