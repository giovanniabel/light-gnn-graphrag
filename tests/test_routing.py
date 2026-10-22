"""Tests for Section 4.4 paragraph scoring and the Section 4.1.1 hybrid fallback."""

from __future__ import annotations

import pytest

from src.graph_builder import SAME_AS_PID
from src.routing import GRAPH_ONLY, HYBRID, coverage, paragraph_scores, rank_paragraphs


class FakeQG:
    """Minimal stand-in exposing only what routing reads."""

    def __init__(self, edge_pid):
        self.edge_pid = list(edge_pid)
        self.n_edges = len(self.edge_pid)

    def covered_pids(self):
        return {p for p in self.edge_pid if p != SAME_AS_PID}


# ------------------------------------------------------- Section 4.4 scoring
def test_paragraph_score_is_mean_of_top_two_edges():
    qg = FakeQG(["p0", "p0", "p0", "p1"])
    s = paragraph_scores(qg, [0.9, 0.7, 0.1, 0.5], mode="mean", top2=2)
    assert s["p0"] == pytest.approx((0.9 + 0.7) / 2)
    assert s["p1"] == pytest.approx(0.5)


def test_paragraph_score_max_mode():
    qg = FakeQG(["p0", "p0"])
    assert paragraph_scores(qg, [0.2, 0.8], mode="max")["p0"] == pytest.approx(0.8)


def test_single_edge_paragraph_scores_that_edge():
    qg = FakeQG(["p0"])
    assert paragraph_scores(qg, [0.42], mode="mean", top2=2)["p0"] == pytest.approx(0.42)


def test_same_as_edges_excluded_from_scoring():
    qg = FakeQG(["p0", SAME_AS_PID])
    s = paragraph_scores(qg, [0.1, 0.99])
    assert set(s) == {"p0"} and s["p0"] == pytest.approx(0.1)


# ------------------------------------------------------- ranking
def test_ranking_is_by_descending_score():
    qg = FakeQG(["p0", "p1", "p2"])
    pool = ["p0", "p1", "p2"]
    assert rank_paragraphs(qg, [0.1, 0.9, 0.5], pool, k=3)[:3] == ["p1", "p2", "p0"]


def test_graph_only_leaves_unreached_paragraphs_below_scored_ones():
    qg = FakeQG(["p1"])
    pool = ["p0", "p1", "p2", "p3"]
    out = rank_paragraphs(qg, [0.5], pool, k=4, variant=GRAPH_ONLY)
    assert out[0] == "p1", "the only scored paragraph must rank first"
    assert set(out[1:]) == {"p0", "p2", "p3"}


def test_hybrid_orders_unreached_paragraphs_by_cosine():
    qg = FakeQG(["p1"])
    pool = ["p0", "p1", "p2", "p3"]
    cos = {"p0": 0.1, "p2": 0.9, "p3": 0.5}
    out = rank_paragraphs(qg, [0.5], pool, k=4, variant=HYBRID, cosines=cos)
    assert out == ["p1", "p2", "p3", "p0"]


def test_hybrid_never_outranks_a_graph_scored_paragraph():
    """The graph's own evidence always comes first; the fallback only fills."""
    qg = FakeQG(["p3"])
    pool = ["p0", "p1", "p2", "p3"]
    out = rank_paragraphs(qg, [0.01], pool, k=4, variant=HYBRID,
                          cosines={"p0": 0.99, "p1": 0.98, "p2": 0.97})
    assert out[0] == "p3"


def test_hybrid_requires_cosines():
    qg = FakeQG(["p0"])
    with pytest.raises(ValueError):
        rank_paragraphs(qg, [0.5], ["p0", "p1"], variant=HYBRID)


def test_graph_only_and_hybrid_agree_when_everything_is_reached():
    """With full coverage the fallback is inert, so the variants must coincide."""
    qg = FakeQG(["p0", "p1", "p2", "p3"])
    pool = ["p0", "p1", "p2", "p3"]
    sc = [0.1, 0.4, 0.3, 0.2]
    a = rank_paragraphs(qg, sc, pool, k=4, variant=GRAPH_ONLY)
    b = rank_paragraphs(qg, sc, pool, k=4, variant=HYBRID, cosines={p: 0.0 for p in pool})
    assert a == b


def test_ranking_returns_every_pool_paragraph():
    qg = FakeQG(["p1"])
    pool = [f"p{i}" for i in range(10)]
    out = rank_paragraphs(qg, [0.5], pool, k=4)
    assert sorted(out) == sorted(pool), "ranking must not drop pool paragraphs"


def test_ties_fall_back_to_pool_order():
    qg = FakeQG(["p0", "p1"])
    pool = ["p0", "p1"]
    assert rank_paragraphs(qg, [0.5, 0.5], pool, k=2) == ["p0", "p1"]


def test_coverage_counts_scorable_pool_paragraphs():
    qg = FakeQG(["p0", "p0", SAME_AS_PID])
    assert coverage(qg, ["p0", "p1", "p2", "p3"]) == pytest.approx(0.25)


def test_coverage_is_zero_for_an_empty_graph():
    assert coverage(FakeQG([]), ["p0", "p1"]) == 0.0


# ------------------------------------------------- RRF fusion (D15)
def test_rrf_fuses_two_rankings():
    from src.routing import RRF_K, reciprocal_rank_fusion

    pool = ["p0", "p1", "p2", "p3"]
    out = reciprocal_rank_fusion([["p3", "p1"], ["p0", "p1", "p2", "p3"]], pool_pids=pool)
    assert out[0] == "p1", "agreed-upon-by-both should win"
    assert set(out) == set(pool)
    assert RRF_K == 60


def test_rrf_k_is_fixed_at_60():
    """D15 freezes k=60; tuning it would make the hybrid a fitted result."""
    from src.routing import RRF_K

    assert RRF_K == 60


def test_rrf_partial_list_contributes_nothing_for_missing_items():
    from src.routing import reciprocal_rank_fusion

    pool = ["p0", "p1"]
    # graph ranks only p1; vector ranks both with p0 first
    out = reciprocal_rank_fusion([["p1"], ["p0", "p1"]], pool_pids=pool)
    assert out == ["p1", "p0"], "1/61+1/62 beats 1/61"


def test_rrf_returns_every_pool_paragraph():
    from src.routing import reciprocal_rank_fusion

    pool = [f"p{i}" for i in range(8)]
    out = reciprocal_rank_fusion([["p7"], ["p0"]], pool_pids=pool)
    assert sorted(out) == sorted(pool)


def test_fuse_with_vector_is_symmetric_in_construction():
    """The matched pair must apply identical fusion to both graph rankers."""
    from src.routing import fuse_with_vector

    pool = ["p0", "p1", "p2", "p3"]
    vec = ["p0", "p1", "p2", "p3"]
    gnn = fuse_with_vector(["p3"], vec, pool)
    ppr = fuse_with_vector(["p3"], vec, pool)
    assert gnn == ppr, "same graph ranking must fuse identically for both arms"


def test_rrf_identical_rankings_preserve_order():
    from src.routing import reciprocal_rank_fusion

    pool = ["p0", "p1", "p2"]
    assert reciprocal_rank_fusion([pool, pool], pool_pids=pool) == pool
