"""Edge scores -> ranked paragraphs (agent.md Section 4.4), plus the optional
hybrid fallback from Section 4.1.1.

Section 4.4: a paragraph's score is the mean of its top-2 edge scores (config flag
`max`); pool paragraphs are ranked by score and the top K go to the reader.

Hybrid fallback: paragraphs with no edge in the ego-graph cannot be scored by a
graph method at all. The `+hybrid` variant appends them, ordered by bge cosine to
the question, below every graph-scored paragraph. It is reported as a SEPARATE,
labelled variant for both PPR and the GNN router; the C1 test stays graph-only,
because a hybrid row is partly measuring Vector RAG.
"""

from __future__ import annotations

from .graph_builder import KIND_TRIPLE, SAME_AS_PID

GRAPH_ONLY = "graph_only"
HYBRID = "hybrid"          # graph-scored first, unreached tail by cosine
RRF = "rrf"                # reciprocal-rank fusion of a graph ranker with vector

# Fixed by D15 and NOT tuned. 60 is the value from the original RRF paper
# (Cormack et al. 2009) and the common default; leaving it fixed is what keeps the
# matched hybrid pair an honest comparison rather than a fitted one.
RRF_K = 60


def paragraph_scores(qg, edge_scores, mode: str = "mean", top2: int = 2) -> dict[str, float]:
    """Section 4.4 step 1: paragraph score = mean of its top-2 edge scores.

    ONLY extracted triples contribute. The Section 4.1.1 scaffolding (title,
    same_as, para) exists to make paragraphs reachable, not to vouch for them, so
    a paragraph reached solely through its own hub gets no score and is not
    rankable by the graph -- which is what the coverage figure measures.
    """
    kinds = getattr(qg, "edge_kind", None) or [KIND_TRIPLE] * len(qg.edge_pid)
    by_pid: dict[str, list[float]] = {}
    for pid, s, kind in zip(qg.edge_pid, edge_scores, kinds):
        if pid == SAME_AS_PID or kind != KIND_TRIPLE:
            continue
        by_pid.setdefault(pid, []).append(float(s))
    out = {}
    for pid, xs in by_pid.items():
        xs.sort(reverse=True)
        out[pid] = max(xs) if mode == "max" else (sum(xs[:top2]) / len(xs[:top2]))
    return out


def rank_paragraphs(
    qg,
    edge_scores,
    pool_pids: list[str],
    k: int = 4,
    mode: str = "mean",
    top2: int = 2,
    variant: str = GRAPH_ONLY,
    cosines: dict[str, float] | None = None,
) -> list[str]:
    """Ranked pool paragraph ids, length >= k where the pool allows it.

    variant=GRAPH_ONLY  unreached paragraphs keep their pool order after the
                        scored ones (they are unrankable by the graph, and the
                        method is not credited with ordering them)
    variant=HYBRID      unreached paragraphs are ordered by `cosines`
    """
    scores = paragraph_scores(qg, edge_scores, mode, top2)
    scored = [p for p in pool_pids if p in scores]
    unreached = [p for p in pool_pids if p not in scores]
    scored.sort(key=lambda p: (-scores[p], pool_pids.index(p)))

    if variant == HYBRID:
        if cosines is None:
            raise ValueError("hybrid variant requires `cosines` for the unreached paragraphs")
        unreached.sort(key=lambda p: (-cosines.get(p, float("-inf")), pool_pids.index(p)))
    # Section 10 requires len(paragraph_ids) >= K, so the whole pool is returned
    # ranked; the caller takes the top K.
    return scored + unreached


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = RRF_K, pool_pids: list[str] | None = None
) -> list[str]:
    """Fuse several ranked pid lists: score(p) = sum_r 1/(k + rank_r(p)), rank 1-based.

    A ranker that cannot rank a paragraph simply contributes nothing for it, which
    is the standard treatment of partial lists and exactly what is needed here: a
    graph method has no opinion about a paragraph it has no edge for, and RRF lets
    the vector ranker speak for it without the graph being credited or penalised.

    `k` is fixed by D15 and must not be tuned.
    """
    scores: dict[str, float] = {}
    for r in rankings:
        for i, pid in enumerate(r, start=1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + i)
    order = pool_pids if pool_pids is not None else list(scores)
    for pid in order:
        scores.setdefault(pid, 0.0)
    tie = {pid: i for i, pid in enumerate(order)}
    return sorted(scores, key=lambda p: (-scores[p], tie.get(p, len(tie))))


def fuse_with_vector(
    graph_ranking: list[str], vector_ranking: list[str], pool_pids: list[str], k: int = RRF_K
) -> list[str]:
    """The D15 matched hybrid: one graph ranker fused with Vector RAG by RRF.

    Used identically for GNN+Vector and PPR+Vector so the pair differs only in the
    graph ranker, which is the comparison C1 can actually win or lose.
    """
    return reciprocal_rank_fusion([graph_ranking, vector_ranking], k=k, pool_pids=pool_pids)


def coverage(qg, pool_pids: list[str]) -> float:
    """Share of pool paragraphs the graph can score at all."""
    cov = qg.covered_pids()
    return len([p for p in pool_pids if p in cov]) / len(pool_pids) if pool_pids else 0.0
