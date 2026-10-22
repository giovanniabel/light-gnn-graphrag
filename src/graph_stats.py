"""Shared connectivity / ceiling measurement, so the train sweep, the before-after
table and the one-shot validation report all compute identical numbers
(agent.md Section 4.1.1, deviation D12).
"""

from __future__ import annotations

import statistics

from .graph_builder import KIND_TRIPLE, SAME_AS_PID, is_yes_no


def _m(xs):
    return statistics.fmean(xs) if xs else 0.0


def _frac(a, b):
    return (a / b) if b else None


def bucket_of(qg) -> str:
    return f"{qg.n_hops}hop" if qg.dataset == "musique" else "2hop"


def measure(graphs: list, pool_size: int, orig_gold: dict | None = None) -> dict:
    """Connectivity and ceiling statistics for one (dataset, split), per hop bucket.

    The AllGold@4 ceiling is reported BOTH ways, as D4 requires of AllGold@4
    itself. The distinction is not cosmetic:

      all_gold_in_pool            fraction of questions whose every ORIGINAL gold
                                  paragraph survived pool construction
      allgold_ceiling_conditional_on_pooling
                                  every gold paragraph THAT IS IN THE POOL has an
                                  ego-graph edge. This conditions on pooling by
                                  SHRINKING THE GOLD SET, which is NOT D4's
                                  conditional AllGold@4 (that restricts the
                                  QUESTION SET). The long name exists so the two
                                  are never conflated -- see metrics.all_gold.
      allgold_ceiling             every ORIGINAL gold paragraph has an ego-graph
                                  edge -- UNCONDITIONAL, and the real cap on
                                  AllGold@4 for any graph method

    `allgold_ceiling` <= both of the others, always. Reporting only the
    conditional number overstates the ceiling (measured: MuSiQue train 77.8%
    conditional against 54.1% unconditional, with all-gold-in-pool at 67.4%).

    `orig_gold` maps qid -> the pre-pooling gold pid set. Without it only the
    conditional figure can be computed, and the unconditional keys are None.
    """
    buckets: dict[str, list] = {}
    for qg in graphs:
        buckets.setdefault(bucket_of(qg), []).append(qg)
    buckets["ALL"] = list(graphs)

    out: dict[str, dict] = {}
    for name, gs in buckets.items():
        n = len(gs)
        pos = sum(g.positive_edges for g in gs)
        tot = sum(g.n_edges for g in gs)
        scorable = [g for g in gs if not is_yes_no(g.answer)]

        covered_shares, sprecall = [], []
        allgold_cond, n_cond = 0, 0
        allgold_uncond, in_pool, sprecall_uncond = 0, 0, []
        have_orig = orig_gold is not None
        for g in gs:
            cov = g.covered_pids()
            covered_shares.append(len(cov) / pool_size)
            gold = g.gold_pids                       # gold ∩ pool
            if gold:
                n_cond += 1
                allgold_cond += gold.issubset(cov)
                sprecall.append(len(gold & cov) / len(gold))
            if have_orig:
                go = orig_gold.get(g.qid, set())     # gold BEFORE pooling
                if go:
                    in_pool += go.issubset({p for p in g.pool_pids})
                    allgold_uncond += go.issubset(cov)
                    sprecall_uncond.append(len(go & cov) / len(go))

        out[name] = {
            "n_questions": n,
            "components_per_question": _m([g.n_components_full for g in gs]),
            "bfs_node_share": _m([
                (g.n_nodes / g.n_nodes_full) if g.n_nodes_full else 0.0 for g in gs
            ]),
            "pool_covered_share": _m(covered_shares),
            # UNCONDITIONAL: the real AllGold@4 cap. None when orig_gold is absent.
            "allgold_ceiling": _frac(allgold_uncond, n) if have_orig else None,
            "allgold_ceiling_conditional_on_pooling": _frac(allgold_cond, n_cond),
            "all_gold_in_pool": _frac(in_pool, n) if have_orig else None,
            "sp_recall_ceiling": _m(sprecall_uncond) if have_orig else None,
            "sp_recall_ceiling_conditional": _m(sprecall),
            "answer_in_graph": _frac(sum(g.answer_in_graph(True) for g in scorable), len(scorable)),
            "graph_ceiling_path": _frac(
                sum(g.path_from_seed_to_answer(True) for g in scorable), len(scorable)
            ),
            "nodes_mean": _m([g.n_nodes for g in gs]),
            "edges_mean": _m([g.n_edges for g in gs]),
            "nodes_max": max((g.n_nodes for g in gs), default=0),
            "edges_max": max((g.n_edges for g in gs), default=0),
            "node_cap_rate": _frac(sum(g.node_cap_bound for g in gs), n),
            "edge_cap_rate": _frac(sum(g.edge_cap_bound for g in gs), n),
            "cap_bound_rate": _frac(sum(g.node_cap_bound or g.edge_cap_bound for g in gs), n),
            "positive_edge_rate": _frac(pos, tot),
            "pos_weight": ((tot - pos) / pos) if pos else None,
            "questions_without_positive_edge": sum(1 for g in gs if g.positive_edges == 0),
            "empty_graphs": sum(1 for g in gs if g.n_edges == 0),
            "added_links_mean": {
                k: _m([g.added_links.get(k, 0) for g in gs])
                for k in ("title", "same_as", "para")
            },
        }
    return out


def link_quality_diagnostic(graphs: list) -> dict:
    """TRAIN-ONLY diagnostic (Section 4.1.1): of the cross-paragraph links the
    connectivity mechanisms add, how many join gold-to-gold, gold-to-distractor
    or distractor-to-distractor paragraphs.

    Purely descriptive -- it reads gold flags and therefore must never be computed
    on eval, and its output must never become a feature or a tuning target.
    """
    counts = {"gold_gold": 0, "gold_distractor": 0, "distractor_distractor": 0}
    by_kind: dict[str, dict] = {}
    pairs_total = 0
    for g in graphs:
        gold = g.gold_pids
        for kind, pidsA, pidsB in g.cross_paragraph_links:
            seen = set()
            for a in pidsA:
                for b in pidsB:
                    if a == b or (a, b) in seen or (b, a) in seen:
                        continue
                    seen.add((a, b))
                    ga, gb = a in gold, b in gold
                    key = "gold_gold" if (ga and gb) else (
                        "distractor_distractor" if not (ga or gb) else "gold_distractor")
                    counts[key] += 1
                    by_kind.setdefault(kind, dict(gold_gold=0, gold_distractor=0,
                                                  distractor_distractor=0))[key] += 1
                    pairs_total += 1
    out = {"cross_paragraph_pairs": pairs_total,
           "counts": counts,
           "shares": {k: _frac(v, pairs_total) for k, v in counts.items()},
           "by_kind": {}}
    for kind, c in by_kind.items():
        t = sum(c.values())
        out["by_kind"][kind] = {"pairs": t, "shares": {k: _frac(v, t) for k, v in c.items()}}
    return out


def meets_target(stats_all: dict, min_cover: float, max_cap_rate: float) -> bool:
    """Section 4.1.1 target: >= min_cover of pool paragraphs have an ego edge, and
    the cap-binding rate stays <= max_cap_rate."""
    return (
        stats_all["pool_covered_share"] >= min_cover
        and (stats_all["cap_bound_rate"] or 0.0) <= max_cap_rate
    )
