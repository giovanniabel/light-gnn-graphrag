"""Step 2, corrected: answer-in-graph on a HOP-STRATIFIED set of MuSiQue train questions.

The first pass sorted questions by qid and took 20, which silently selected only
2-hop questions ("2hop__" sorts before "3hop"/"4hop"). Multi-hop is exactly where
a sparser extraction should hurt, so the sample is re-drawn stratified across
2/3/4 hops. Existing per-max_triples caches are reused, so overlapping paragraphs
cost nothing.
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
from src.extractor import TripleCache, extract_paragraph  # noqa: E402
from src.normalize import answer_in_nodes, node_set  # noqa: E402
from src.ollama_client import OllamaClient, unload_all  # noqa: E402
from scripts.m1_data import load_pools  # noqa: E402

VALUES = (6, 5, 4)
PER_HOP = {2: 7, 3: 7, 4: 6}   # 20 questions, stratified


def pick(cfg, seed: int):
    exs = load_pools("musique", "train", int(cfg["data"]["pool_size"]))
    by_hop: dict[int, list] = {}
    for e in exs:
        by_hop.setdefault(e.n_hops, []).append(e)
    rng = random.Random(seed)
    out = []
    for hop, k in PER_HOP.items():
        pool = sorted(by_hop.get(hop, []), key=lambda e: e.qid)
        rng.shuffle(pool)
        out.extend(pool[:k])
    return sorted(out, key=lambda e: e.qid)


def main() -> int:
    cfg = load_config()
    ex_cfg = cfg["extraction"]
    model_tag = env("SLM_TAG")
    exs = pick(cfg, int(cfg["seed"]))
    from collections import Counter
    print(f"=== hop-stratified quality set: {len(exs)} MuSiQue train questions ===")
    print(f"  hops: {dict(sorted(Counter(e.n_hops for e in exs).items()))}")
    print(f"  paragraphs: {sum(len(e.paragraphs) for e in exs)}")

    out = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "question_ids": [e.qid for e in exs],
           "hop_counts": dict(sorted(Counter(e.n_hops for e in exs).items())),
           "runs": {}}

    for mt in VALUES:
        pv = f"{ex_cfg['prompt_version']}-mt{mt}"
        cache = TripleCache(ROOT / "data" / "cache" / f"triples_sweep_mt{mt}.jsonl", model_tag, pv)
        client = OllamaClient(model_tag, num_ctx=int(ex_cfg["num_ctx"]),
                              temperature=float(ex_cfg["temperature"]))
        print(f"\n### max_triples={mt}", flush=True)
        n_new = 0
        t0 = time.perf_counter()
        for i, ex in enumerate(exs, 1):
            for p in ex.paragraphs:
                if cache.key_for(p.text) not in cache._index:
                    n_new += 1
                extract_paragraph(p, cache, client,
                                  num_predict=int(ex_cfg["num_predict"]),
                                  max_words=int(ex_cfg["max_words_per_paragraph"]),
                                  retries=int(ex_cfg["retries"]), max_triples=mt)
            if i % 5 == 0:
                print(f"  {i}/{len(exs)} questions ({n_new} new paragraphs, "
                      f"{time.perf_counter()-t0:.0f}s)", flush=True)

        per_q, by_hop = [], {}
        n_para_triples = []
        for ex in exs:
            triples = []
            for p in ex.paragraphs:
                t = cache.get(p.text) or []
                triples.extend(t)
                n_para_triples.append(len(t))
            nodes = node_set(triples)
            ok = answer_in_nodes(ex.answer, nodes, ex.meta.get("answer_aliases"))
            per_q.append({"qid": ex.qid, "n_hops": ex.n_hops, "n_triples": len(triples),
                          "n_nodes": len(nodes), "answer_in_graph": ok})
            d = by_hop.setdefault(str(ex.n_hops), {"hit": 0, "n": 0})
            d["hit"] += ok
            d["n"] += 1
        rate = sum(q["answer_in_graph"] for q in per_q) / len(per_q)
        out["runs"][str(mt)] = {
            "max_triples": mt,
            "answer_in_graph_rate": rate,
            "by_hop": {k: {"rate": v["hit"] / v["n"], "hit": v["hit"], "n": v["n"]}
                       for k, v in sorted(by_hop.items())},
            "triples_per_paragraph_mean": statistics.fmean(n_para_triples),
            "triples_per_question_mean": statistics.fmean(q["n_triples"] for q in per_q),
            "nodes_per_question_mean": statistics.fmean(q["n_nodes"] for q in per_q),
            "new_paragraphs_extracted": n_new,
            "per_question": per_q,
        }
        hops = "  ".join(f"{k}hop {v['rate']:.0%}({v['hit']}/{v['n']})"
                         for k, v in out["runs"][str(mt)]["by_hop"].items())
        print(f"  -> answer-in-graph {rate:.1%} | {hops} | "
              f"{out['runs'][str(mt)]['triples_per_paragraph_mean']:.2f} triples/para", flush=True)
        cache.close()
        p = ROOT / "results" / "quality_stratified.json"
        p.write_text(json.dumps(out, indent=2))

    unload_all()
    print(f"\n-> results/quality_stratified.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
