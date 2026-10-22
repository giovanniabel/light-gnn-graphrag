"""Tests for D17 cross-validation folds and D17/§5 graph masking."""

from __future__ import annotations

import torch

from evaluation.cv import (
    ABLATION_MASKS,
    hop_dist_of,
    inner_holdout,
    make_folds,
    mask_graph,
    stratum_of,
)
from src.graph_builder import EDGE_FEAT_DIM, NODE_FEAT_DIM
from src.schemas import Example, Paragraph


def mk(qid, hops=2, ds="musique", typ=None):
    return Example(qid, ds, "q?", "a", [Paragraph("p0", "T", "t", True)], hops,
                   {"hotpot_type": typ} if typ else {})


# ------------------------------------------------------------ folds
def test_every_question_is_tested_exactly_once():
    exs = [mk(f"q{i}", hops=2 + i % 3) for i in range(60)]
    folds = make_folds(exs, 5, 1234)
    tested = [q for f in folds for q in f["test"]]
    assert sorted(tested) == sorted(e.qid for e in exs)
    assert len(tested) == len(set(tested)), "a question appeared in two test folds"


def test_train_and_test_are_disjoint_within_each_fold():
    exs = [mk(f"q{i}", hops=2 + i % 3) for i in range(60)]
    for f in make_folds(exs, 5, 1234):
        assert not (set(f["train"]) & set(f["test"]))
        assert len(f["train"]) + len(f["test"]) == len(exs)


def test_folds_are_stratified_by_hop_count():
    exs = [mk(f"q{i}", hops=2 + i % 3) for i in range(60)]   # 20 each of 2/3/4
    for f in make_folds(exs, 5, 1234):
        by = {}
        for q in f["test"]:
            h = next(e.n_hops for e in exs if e.qid == q)
            by[h] = by.get(h, 0) + 1
        assert set(by) == {2, 3, 4} and all(v == 4 for v in by.values())


def test_folds_are_seed_stable_and_identical_across_configs():
    exs = [mk(f"q{i}", hops=2 + i % 3) for i in range(30)]
    a = make_folds(exs, 5, 1234)
    b = make_folds(exs, 5, 1234)
    assert [f["test"] for f in a] == [f["test"] for f in b]


def test_different_seed_gives_different_folds():
    exs = [mk(f"q{i}", hops=2 + i % 3) for i in range(30)]
    assert ([f["test"] for f in make_folds(exs, 5, 1)]
            != [f["test"] for f in make_folds(exs, 5, 2)])


def test_hotpotqa_stratifies_by_question_type():
    exs = [mk(f"q{i}", ds="hotpotqa", typ="bridge" if i % 2 else "comparison")
           for i in range(40)]
    assert stratum_of(exs[0]) in ("bridge", "comparison")
    for f in make_folds(exs, 5, 1234):
        types = [stratum_of(next(e for e in exs if e.qid == q)) for q in f["test"]]
        assert types.count("bridge") == types.count("comparison") == 4


def test_inner_holdout_is_disjoint_and_seeded():
    train = [f"q{i}" for i in range(100)]
    tr, ho = inner_holdout(train, 0.10, 7)
    assert len(ho) == 10 and not (set(tr) & set(ho))
    assert sorted(tr + ho) == sorted(train)
    assert inner_holdout(train, 0.10, 7)[1] == ho


# ------------------------------------------------------------ masking
def fake(n=10, kinds=None):
    from torch_geometric.data import Data

    kinds = kinds or ["triple", "triple", "title", "same_as", "para"]
    e = len(kinds)
    x = torch.randn(n, NODE_FEAT_DIM)
    x[:, 384 * 2 + 2: 384 * 2 + 6] = 0.0
    for i in range(n):
        x[i, 384 * 2 + 2 + min(i, 3)] = 1.0          # hop_dist = min(i, 3)
    return Data(
        x=x,
        edge_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]]),
        edge_attr=torch.randn(e, EDGE_FEAT_DIM),
        y_edge=torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]),
        q=torch.randn(1, 384),
        seed_mask=torch.tensor([True] + [False] * (n - 1)),
        evidence_mask=torch.tensor([k == "triple" for k in kinds]),
        edge_kind=list(kinds), edge_pid=[f"p{i}" for i in range(e)],
        qid="q1", dataset="musique", n_hops=2,
    )


def test_hop_dist_recovered_from_features():
    d = fake()
    assert hop_dist_of(d).tolist()[:5] == [0, 1, 2, 3, 3]


def test_drop_kinds_removes_only_those_edges():
    d = mask_graph(fake(), drop_kinds=("para",))
    assert "para" not in d.edge_kind
    assert d.edge_index.size(1) == 4 and d.edge_attr.size(0) == 4
    assert len(d.edge_pid) == 4 and d.y_edge.numel() == 4


def test_triples_only_keeps_just_extracted_edges():
    d = mask_graph(fake(), **ABLATION_MASKS["triples_only"])
    assert set(d.edge_kind) == {"triple"}


def test_masking_preserves_feature_widths():
    """The whole point of masking over rebuilding (§5)."""
    d = mask_graph(fake(), drop_kinds=("title", "para"), max_hop=2)
    assert d.x.size(1) == NODE_FEAT_DIM and d.edge_attr.size(1) == EDGE_FEAT_DIM


def test_max_hop_drops_far_nodes_and_reindexes_edges():
    d = mask_graph(fake(), max_hop=1)
    assert d.x.size(0) == 2, "only hop 0 and hop 1 nodes survive"
    if d.edge_index.numel():
        assert int(d.edge_index.max()) < d.x.size(0), "edge_index must be reindexed"


def test_max_hop_keeps_seeds_even_if_nothing_qualifies():
    d = fake()
    out = mask_graph(d, max_hop=-1)
    assert out.x.size(0) >= 1


def test_mask_does_not_mutate_the_input():
    d = fake()
    before_e = d.edge_index.size(1)
    mask_graph(d, drop_kinds=("para",), max_hop=1)
    assert d.edge_index.size(1) == before_e, "input graph was mutated"


def test_empty_mask_is_a_faithful_copy():
    d = fake()
    out = mask_graph(d, drop_kinds=())
    assert out.edge_index.size(1) == d.edge_index.size(1)
    assert out.edge_kind == d.edge_kind
