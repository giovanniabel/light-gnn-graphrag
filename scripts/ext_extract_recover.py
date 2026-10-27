"""M11 step 3 (recovery): rebuild the extraction report from the triple cache.

The first run of scripts/ext_extract.py had a retry-loop bug (now fixed): when a
chunk's swap growth crossed the Section 7.3 threshold, the chunk was re-run to
"retry the measurement", but by then every paragraph was cached, so the second
pass measured nothing and overwrote the chunk's record with n_extracted=0. The
triples themselves were written correctly -- the cache flushes per paragraph --
so 12 of 13 chunks lost only their accounting.

Nothing is re-extracted here. The cache stores `seconds`, `json_failed`,
`n_triples` and `llm_tokens` for every paragraph, so the per-chunk table is
rebuilt from it; the per-chunk swap deltas are read back from the run log, where
the (real) first-pass values were printed. Chunks whose swap delta exceeded the
threshold are marked timing-invalid, which the buggy run had recorded as valid.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT, env, load_config  # noqa: E402
from src.data import pool_size_for  # noqa: E402
from src.extractor import TripleCache, effective_prompt_version  # noqa: E402
from src.validation_guard import allow  # noqa: E402
from scripts.m1_data import load_pools  # noqa: E402

CHUNK = 500
LOG = ROOT / "results" / "logs" / "ext_extract.log"
PRE_EXISTING = 6489  # "cache holds 6489 paragraphs", printed before extraction


def main() -> int:
    cfg = load_config()
    allow("M11 step 3 recovery: rebuilding the extraction report from the cache (D28)")
    tag, pv = env("SLM_TAG"), effective_prompt_version(cfg)
    thresh = float(cfg["profiling"]["swap_invalidate_mb"])

    # ---- cache lines, in write order --------------------------------------
    rows, pos = [], {}
    for i, line in enumerate(cfg.path("triples").read_text().splitlines()):
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("model_tag") != tag or d.get("prompt_version") != pv:
            continue
        pos[d["key"]] = len(rows)
        rows.append(d)
    print(f"  cache: {len(rows):,} paragraphs for {tag} / {pv}")

    # ---- todo order, exactly as the extraction built it -------------------
    cache = TripleCache(cfg.path("triples"), tag, pv)
    todo, seen = [], set()
    for ds in cfg["data"]["datasets"]:
        ps = pool_size_for(cfg, ds)
        for ex in load_pools(ds, "validation_ext", ps):
            for p in ex.paragraphs:
                k = cache.key_for(p.text)
                if k in seen:
                    continue
                seen.add(k)
                if pos.get(k, -1) >= PRE_EXISTING:
                    todo.append(k)
    print(f"  extracted by this run: {len(todo):,} paragraphs")
    assert len(rows) - PRE_EXISTING == len(todo), (len(rows), PRE_EXISTING, len(todo))
    idx = [pos[k] for k in todo]
    assert idx == sorted(idx), "cache order does not match todo order"

    # ---- swap deltas, read back from the run log --------------------------
    log = LOG.read_text()
    # The "swap grew" line is the real (first-pass) delta. The per-chunk summary
    # line reports the bogus second pass for every retried chunk (+0 MB), so it
    # is only used for a chunk that was never retried.
    swap = {int(m[1]): float(m[2])
            for m in re.finditer(r"chunk\s+(\d+): swap grew ([-+0-9.]+) MB", log)}
    for m in re.finditer(r"chunk\s+(\d+)/\d+:\s+\d+ extracted.*?swap ([-+0-9.]+) MB", log):
        swap.setdefault(int(m[1]), float(m[2]))

    # ---- per-chunk table --------------------------------------------------
    chunks, all_s, all_fail, all_tok, all_tri = [], [], 0, [], []
    salv = trunc = 0
    for ci in range(0, (len(todo) + CHUNK - 1) // CHUNK):
        ks = todo[ci * CHUNK:(ci + 1) * CHUNK]
        rs = [rows[pos[k]] for k in ks]
        sec = [float(r["seconds"]) for r in rs]
        fails = sum(bool(r.get("json_failed")) for r in rs)
        tri = [int(r.get("n_triples", len(r["triples"]))) for r in rs]
        tok = [int(r.get("llm_tokens", 0)) for r in rs]
        g = swap.get(ci)
        all_s += sec; all_fail += fails; all_tri += tri; all_tok += tok
        salv += sum(bool(r.get("salvaged")) for r in rs)
        trunc += sum(bool(r.get("truncated")) for r in rs)
        chunks.append({"chunk": ci, "n_extracted": len(rs), "complete": True,
                       "t_p": statistics.fmean(sec), "json_failures": fails,
                       "triples_mean": statistics.fmean(tri),
                       "swap_growth_mb": g,
                       "valid": (g is not None and g <= thresh)})

    inval = [c["chunk"] for c in chunks if not c["valid"]]
    prev = json.loads((ROOT / "results" / "ext_extraction.json").read_text())

    # ---- coverage (unaffected by the bug; recomputed to keep one source) --
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
                   "slots_with_triples": with_tri, "gold_paragraphs": gold_t,
                   "gold_with_triples": gold_w,
                   "extraction_coverage": gold_w / gold_t,
                   "paragraph_coverage": with_tri / slots}

    _valid_s = [float(rows[pos[k]]["seconds"])
                for c in chunks if c["valid"]
                for k in todo[c["chunk"] * CHUNK:(c["chunk"] + 1) * CHUNK]]

    out = {
        "wall_s": prev["wall_s"], "wall_h": prev["wall_h"],
        "paragraphs_extracted": len(todo),
        "t_p_mean_s": statistics.fmean(all_s),
        "t_p_median_s": statistics.median(all_s),
        "t_p_p95_s": sorted(all_s)[int(0.95 * len(all_s))],
        "t_p_valid_chunks_only_s": (statistics.fmean(_valid_s) if _valid_s else None),
        "llm_tokens_mean": statistics.fmean(all_tok),
        "triples_mean": statistics.fmean(all_tri),
        "json_failures": all_fail,
        "json_failure_rate": all_fail / len(todo),
        "salvaged": salv, "truncated": trunc,
        "chunks": chunks, "invalid_timing_chunks": inval,
        "timing_validity_note": (
            f"{len(inval)} of {len(chunks)} chunks exceeded the "
            f"{thresh:.0f} MB swap-growth threshold, so the aggregate t_p is "
            "Section 7.3-invalid; t_p_valid_chunks_only_s covers the rest. "
            "Triples are unaffected (greedy decoding)."),
        "recovered_from_cache": True,
        "superseded_report": {k: prev[k] for k in
                              ("paragraphs_extracted", "t_p_mean_s",
                               "json_failures", "json_failure_rate",
                               "invalid_timing_chunks")},
        "coverage": cov,
    }
    (ROOT / "results" / "ext_extraction.json").write_text(json.dumps(out, indent=2))

    print(f"\n--- recovered step-3 report ---")
    print(f"  wall {out['wall_h']:.2f} h, {out['paragraphs_extracted']:,} paragraphs")
    print(f"  t_p mean {out['t_p_mean_s']:.3f} s  median {out['t_p_median_s']:.3f} s  "
          f"p95 {out['t_p_p95_s']:.3f} s   [timing run: 1.767 s]")
    v = out["t_p_valid_chunks_only_s"]
    print(f"  t_p on timing-valid chunks only: "
          f"{f'{v:.3f} s' if v is not None else 'none valid'}")
    print(f"  LLM tokens/paragraph {out['llm_tokens_mean']:.1f}   [original 408, timing 423]")
    print(f"  triples/paragraph {out['triples_mean']:.2f}   [original 5.48, timing 5.45]")
    print(f"  JSON failures {out['json_failures']}/{out['paragraphs_extracted']} "
          f"= {out['json_failure_rate']:.4%}   [original run: 0.02%]")
    print(f"  salvaged {salv}, truncated {trunc}")
    print(f"  timing-invalid chunks: {len(inval)} of {len(chunks)} -> {inval}")
    for ds, c in cov.items():
        print(f"  {ds}: gold coverage {c['extraction_coverage']:.4f} "
              f"({c['gold_with_triples']}/{c['gold_paragraphs']}), "
              f"paragraph coverage {c['paragraph_coverage']:.4f} "
              f"({c['slots_with_triples']}/{c['paragraph_slots']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
