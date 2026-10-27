"""Unit tests for seed linking, label construction and the ego-graph caps
(agent.md Section 12 requires tests for seed linking and label construction)."""

from __future__ import annotations

import torch

from src.graph_builder import (
    EDGE_FEAT_DIM,
    NODE_FEAT_DIM,
    SEED_FALLBACK,
    SEED_NONE,
    SEED_SUBSTRING,
    build_graph,
    find_seeds,
    is_yes_no,
    to_pyg,
)
from src.schemas import Example, Paragraph, Triple


class FakeEmbedder:
    """Deterministic, no model. encode_query biases toward the first node."""

    def __init__(self, favour: str | None = None):
        self.favour = favour

    def _vec(self, t: str) -> torch.Tensor:
        g = torch.Generator().manual_seed(abs(hash(t)) % (2**31))
        v = torch.randn(8, generator=g)
        return v / v.norm()

    def encode(self, texts, kind="misc"):
        if isinstance(texts, str):
            return self._vec(texts)
        return torch.stack([self._vec(t) for t in texts])

    def encode_query(self, q):
        return self._vec(self.favour if self.favour else q)


def make_ex(question="Who is the spouse of the Green performer?", answer="Miquette Giraudy",
            gold=("p0",), pids=("p0", "p1"), aliases=None, dataset="musique", n_hops=2):
    paras = [Paragraph(pid=p, title=f"T{p}", text=f"text {p}", is_gold=p in gold) for p in pids]
    return Example("q1", dataset, question, answer, paras, n_hops,
                   {"answer_aliases": list(aliases or [])})


def T(h, r, t, pid="p0"):
    return Triple(h, r, t, pid)


# ------------------------------------------------------------- seed linking
def test_seed_substring_match_is_preferred():
    nodes = ["grant green", "blue note", "unrelated thing"]
    seeds, mode = find_seeds("Who recorded with Grant Green?", nodes, FakeEmbedder())
    assert mode == SEED_SUBSTRING
    assert nodes[seeds[0]] == "grant green"


def test_seed_substring_is_normalized_on_both_sides():
    """Question punctuation/case must not block the match."""
    seeds, mode = find_seeds("The GRANT GREEN, story!", ["grant green"], FakeEmbedder())
    assert mode == SEED_SUBSTRING and seeds == [0]


def test_seed_falls_back_to_embedding_when_no_substring():
    nodes = ["alpha", "beta", "gamma", "delta"]
    seeds, mode = find_seeds("totally disjoint question", nodes, FakeEmbedder())
    assert mode == SEED_FALLBACK
    assert 1 <= len(seeds) <= 3, "fallback takes the top-3 (Section 4.1)"


def test_seed_fallback_requires_an_embedder():
    seeds, mode = find_seeds("disjoint", ["alpha"], embedder=None)
    assert mode == SEED_NONE and seeds == []


def test_seed_cap_is_respected():
    nodes = [f"n{i}" for i in range(20)]
    q = "n0 n1 n2 n3 n4 n5 n6 n7 n8 n9 n10"
    seeds, mode = find_seeds(q, nodes, FakeEmbedder(), max_seeds=5)
    assert mode == SEED_SUBSTRING and len(seeds) == 5, "Section 4.1 caps seeds at 5"


def test_seed_longest_match_first():
    nodes = ["green", "grant green"]
    seeds, _ = find_seeds("grant green plays", nodes, FakeEmbedder(), max_seeds=1)
    assert nodes[seeds[0]] == "grant green"


def test_no_seeds_when_no_nodes():
    assert find_seeds("anything", [], FakeEmbedder()) == ([], SEED_NONE)


# ------------------------------------------------- label construction (4.2)
def test_label_is_one_iff_source_paragraph_is_gold():
    ex = make_ex(gold=("p0",), pids=("p0", "p1"))
    triples = [T("Alice", "knows", "Bob", "p0"), T("Carol", "knows", "Dave", "p1")]
    qg = build_graph(ex, triples, FakeEmbedder(), merge_near_duplicates=False)
    for y, pid in zip(qg.edge_y, qg.edge_pid):
        assert y == (1 if pid == "p0" else 0)
    assert set(qg.edge_y) == {0, 1}


def test_reverse_edge_inherits_the_same_label_and_pid():
    ex = make_ex(gold=("p0",))
    qg = build_graph(ex, [T("Alice", "knows", "Bob", "p0")], FakeEmbedder(),
                     merge_near_duplicates=False)
    assert qg.n_edges == 2, "one directed edge plus a flagged reverse (Section 4.1)"
    assert qg.edge_y == [1, 1]
    assert set(qg.edge_dir) == {0, 1}, "direction bit distinguishes them"
    assert len(set(qg.edge_pid)) == 1


def test_all_labels_zero_when_no_gold_paragraph_produced_edges():
    ex = make_ex(gold=("p0",))
    qg = build_graph(ex, [T("Carol", "knows", "Dave", "p1")], FakeEmbedder(),
                     merge_near_duplicates=False)
    assert qg.positive_edges == 0, "this question is dropped from TRAINING only (Section 4.2)"


def test_labels_follow_gold_flags_not_paragraph_order():
    ex = make_ex(gold=("p1",), pids=("p0", "p1"))
    triples = [T("A", "r", "B", "p0"), T("C", "r", "D", "p1")]
    qg = build_graph(ex, triples, FakeEmbedder(), merge_near_duplicates=False)
    pos_pids = {p for p, y in zip(qg.edge_pid, qg.edge_y) if y == 1}
    assert pos_pids == {"p1"}


# -------------------------------------------------------- ego-graph caps
def _chain(n: int, pid="p0"):
    return [T(f"n{i}", "next", f"n{i+1}", pid) for i in range(n)]


def test_node_cap_binds_and_is_reported():
    ex = make_ex(question="about n0", gold=("p0",))
    qg = build_graph(ex, _chain(300), FakeEmbedder(), hops=100, max_nodes=50, max_edges=10_000,
                     merge_near_duplicates=False)
    assert qg.n_nodes == 50 and qg.node_cap_bound is True


def test_edge_cap_binds_and_is_reported():
    ex = make_ex(question="about n0", gold=("p0",))
    qg = build_graph(ex, _chain(300), FakeEmbedder(), hops=100, max_nodes=10_000, max_edges=40,
                     merge_near_duplicates=False)
    assert qg.n_edges == 40 and qg.edge_cap_bound is True


def test_caps_not_marked_bound_when_graph_is_small():
    ex = make_ex(question="about n0", gold=("p0",))
    qg = build_graph(ex, _chain(5), FakeEmbedder(), hops=10, max_nodes=200, max_edges=400,
                     merge_near_duplicates=False)
    assert qg.node_cap_bound is False and qg.edge_cap_bound is False


def test_hop_radius_limits_the_ball():
    """H=2 from n0 must not reach n5."""
    ex = make_ex(question="about n0", gold=("p0",))
    qg = build_graph(ex, _chain(10), FakeEmbedder(), hops=2, max_nodes=200, max_edges=400,
                     merge_near_duplicates=False)
    assert "n0" in qg.nodes and "n5" not in qg.nodes
    assert max(qg.hop_dist) <= 2


def test_edges_only_connect_kept_nodes():
    ex = make_ex(question="about n0", gold=("p0",))
    qg = build_graph(ex, _chain(300), FakeEmbedder(), hops=100, max_nodes=30, max_edges=400,
                     merge_near_duplicates=False)
    for u, v in qg.edges:
        assert 0 <= u < qg.n_nodes and 0 <= v < qg.n_nodes


# -------------------------------------------------------- nodes / merging
def test_exact_node_matches_are_merged():
    ex = make_ex(gold=("p0",))
    triples = [T("Grant Green", "on", "Blue Note", "p0"), T("grant green!", "born", "1931", "p0")]
    qg = build_graph(ex, triples, FakeEmbedder(), merge_near_duplicates=False)
    assert qg.nodes.count("grant green") == 1


def test_self_loops_and_blank_triples_dropped():
    ex = make_ex(gold=("p0",))
    triples = [T("A", "r", "A", "p0"), T("", "r", "B", "p0"), T("C", "", "D", "p0"),
               T("E", "r", "F", "p0")]
    qg = build_graph(ex, triples, FakeEmbedder(), merge_near_duplicates=False)
    assert set(qg.nodes) == {"e", "f"}


# -------------------------------------------------------- ceilings
def test_answer_in_graph_and_alias_matching():
    ex = make_ex(answer="Miquette Giraudy", aliases=["M. Giraudy"], gold=("p0",))
    qg = build_graph(ex, [T("Steve Hillage", "spouse", "M. Giraudy", "p0")], FakeEmbedder(),
                     merge_near_duplicates=False)
    assert qg.answer_in_graph(False) is False
    assert qg.answer_in_graph(True) is True, "alias matching must be reported separately"


def test_path_from_seed_to_answer():
    ex = make_ex(question="about n0", answer="n2", gold=("p0",))
    qg = build_graph(ex, _chain(4), FakeEmbedder(), hops=3, merge_near_duplicates=False)
    assert qg.answer_in_graph() and qg.path_from_seed_to_answer()


def test_path_false_when_answer_absent():
    ex = make_ex(question="about n0", answer="nowhere", gold=("p0",))
    qg = build_graph(ex, _chain(4), FakeEmbedder(), hops=3, merge_near_duplicates=False)
    assert not qg.answer_in_graph() and not qg.path_from_seed_to_answer()


def test_is_yes_no_excludes_only_yes_and_no():
    assert is_yes_no("yes") and is_yes_no("No.") and not is_yes_no("yesterday")


# -------------------------------------------------------- PyG conversion
class FakeCache:
    def get(self, text, kind="misc"):
        g = torch.Generator().manual_seed(abs(hash((text, kind))) % (2**31))
        return torch.randn(384, generator=g)


def test_to_pyg_shapes_match_section_43():
    ex = make_ex(question="about n0", gold=("p0",))
    qg = build_graph(ex, _chain(6), FakeEmbedder(), hops=3, merge_near_duplicates=False)
    d = to_pyg(qg, FakeCache(), hops=3)
    N, E = qg.n_nodes, qg.n_edges
    assert d.x.shape == (N, NODE_FEAT_DIM)
    assert tuple(d.edge_index.shape) == (2, E)
    assert d.edge_attr.shape == (E, EDGE_FEAT_DIM)
    assert tuple(d.y_edge.shape) == (E,)
    assert tuple(d.q.shape) == (1, 384)
    assert tuple(d.seed_mask.shape) == (N,)
    assert len(d.edge_pid) == E


def test_to_pyg_edge_feature_layout():
    """[r_e ; cos(r_e,q) ; direction_bit ; edge-kind one-hot] -- the direction bit
    sits just before the kind block, which is last."""
    from src.graph_builder import EDGE_KINDS

    ex = make_ex(gold=("p0",))
    qg = build_graph(ex, [T("A", "r", "B", "p0")], FakeEmbedder(), merge_near_duplicates=False)
    d = to_pyg(qg, FakeCache(), hops=3)
    nk = len(EDGE_KINDS)
    dirbit = d.edge_attr[:, -(nk + 1)]
    assert set(dirbit.tolist()) == {0.0, 1.0}
    kinds = d.edge_attr[:, -nk:]
    assert bool((kinds.sum(dim=1) == 1).all()), "edge kind must be exactly one-hot"
    assert bool((kinds[:, EDGE_KINDS.index("triple")] == 1).all())


def test_to_pyg_node_kind_block_is_last_and_multi_hot():
    from src.graph_builder import NODE_KINDS

    ex = _multi_para_ex({"p0"})
    qg = build_graph(ex, _triples_three_paras(), FakeEmbedder(), connectivity=CONN_FULL)
    d = to_pyg(qg, FakeCache(), hops=3)
    block = d.x[:, -len(NODE_KINDS):]
    assert bool(((block == 0) | (block == 1)).all())
    assert bool((block.sum(dim=1) >= 1).all()), "every node has at least one kind"
    assert float(block.sum()) >= qg.n_nodes


def test_to_pyg_handles_empty_graph():
    ex = make_ex(gold=("p0",))
    qg = build_graph(ex, [], FakeEmbedder(), merge_near_duplicates=False)
    d = to_pyg(qg, FakeCache(), hops=3)
    assert d.x.shape == (0, NODE_FEAT_DIM) and d.y_edge.numel() == 0


def test_to_pyg_seed_mask_marks_the_seeds():
    ex = make_ex(question="about n0", gold=("p0",))
    qg = build_graph(ex, _chain(5), FakeEmbedder(), hops=3, merge_near_duplicates=False)
    d = to_pyg(qg, FakeCache(), hops=3)
    assert int(d.seed_mask.sum()) == len(qg.seeds) >= 1


# ------------------------------------------- D12 / Section 4.1.1: no gold leakage
CONN_FULL = {
    "title_hubs": True, "title_fuzzy": True, "title_token_jaccard": 0.5,
    "same_as_edges": True, "same_as_cosine": 0.85, "same_as_token_match": True,
    "paragraph_nodes": True,
}


def _structure(qg):
    """Everything about the graph except the labels."""
    return (
        tuple(qg.nodes), tuple(qg.edges), tuple(qg.edge_rel), tuple(qg.edge_pid),
        tuple(qg.edge_dir), tuple(qg.edge_kind), tuple(qg.seeds), tuple(qg.hop_dist),
        qg.n_nodes_full, qg.n_edges_full, qg.n_components_full,
        qg.node_cap_bound, qg.edge_cap_bound,
    )


def _multi_para_ex(gold):
    paras = [
        Paragraph("p0", "Grant Green", "Grant Green played guitar.", "p0" in gold),
        Paragraph("p1", "Blue Note", "Blue Note released records.", "p1" in gold),
        Paragraph("p2", "Idris Muhammad", "Idris Muhammad played drums.", "p2" in gold),
    ]
    return Example("qX", "musique", "Who played with Grant Green?", "Idris Muhammad",
                   paras, 2, {"answer_aliases": []})


def _triples_three_paras():
    return [
        T("Grant Green", "plays", "guitar", "p0"),
        T("Blue Note", "released", "First Stand", "p1"),
        T("Idris Muhammad", "plays", "drums", "p2"),
    ]


def test_flipping_every_gold_flag_leaves_the_graph_structure_identical():
    """Section 4.1.1: no connectivity decision may read is_gold."""
    from dataclasses import replace

    base = _multi_para_ex({"p0"})
    flipped = replace(base, paragraphs=[replace(p, is_gold=not p.is_gold)
                                        for p in base.paragraphs])
    tri = _triples_three_paras()
    for conn in ({}, CONN_FULL):
        a = build_graph(base, tri, FakeEmbedder(), connectivity=conn)
        b = build_graph(flipped, tri, FakeEmbedder(), connectivity=conn)
        assert _structure(a) == _structure(b), f"structure changed with gold flags (conn={conn})"


def test_flipping_gold_flips_only_the_labels():
    from dataclasses import replace

    base = _multi_para_ex({"p0"})
    flipped = replace(base, paragraphs=[replace(p, is_gold=not p.is_gold)
                                        for p in base.paragraphs])
    tri = _triples_three_paras()
    a = build_graph(base, tri, FakeEmbedder(), connectivity=CONN_FULL)
    b = build_graph(flipped, tri, FakeEmbedder(), connectivity=CONN_FULL)
    for ya, yb, pid, kind in zip(a.edge_y, b.edge_y, a.edge_pid, a.edge_kind):
        if kind == "triple":
            assert ya != yb, "an extracted edge's label follows its paragraph's gold flag"
        else:
            assert ya == yb == 0, "scaffolding edges are never positive, either way"


def test_removing_all_gold_leaves_structure_identical():
    from dataclasses import replace

    base = _multi_para_ex({"p0", "p2"})
    nogold = replace(base, paragraphs=[replace(p, is_gold=False) for p in base.paragraphs])
    tri = _triples_three_paras()
    a = build_graph(base, tri, FakeEmbedder(), connectivity=CONN_FULL)
    b = build_graph(nogold, tri, FakeEmbedder(), connectivity=CONN_FULL)
    assert _structure(a) == _structure(b)
    assert sum(b.edge_y) == 0


# ------------------------------------------- connectivity mechanisms
def test_title_hubs_bridge_two_paragraphs():
    """A title node merges with an identically named entity elsewhere, which is
    the mechanism that joins components."""
    paras = [Paragraph("p0", "Grant Green", "about the album", True),
             Paragraph("p1", "Other", "text", False)]
    ex = Example("q", "musique", "Who is Grant Green?", "guitar", paras, 2, {})
    tri = [T("First Stand", "by", "Grant Green", "p1"), T("Grant Green", "plays", "guitar", "p0")]
    off = build_graph(ex, tri, FakeEmbedder(), connectivity={})
    on = build_graph(ex, tri, FakeEmbedder(),
                     connectivity={"title_hubs": True, "title_fuzzy": False})
    assert on.n_components_full <= off.n_components_full
    assert len(on.covered_pids()) >= len(off.covered_pids())


def test_same_as_edges_are_never_positive_and_never_score_paragraphs():
    from src.graph_builder import SAME_AS_PID
    from src.routing import paragraph_scores

    ex = _multi_para_ex({"p0", "p1", "p2"})
    qg = build_graph(ex, _triples_three_paras(), FakeEmbedder(), connectivity=CONN_FULL)
    for y, pid in zip(qg.edge_y, qg.edge_pid):
        if pid == SAME_AS_PID:
            assert y == 0
    scores = paragraph_scores(qg, [1.0] * qg.n_edges)
    assert SAME_AS_PID not in scores, "same_as must not create a scorable paragraph"


def test_paragraph_nodes_cover_the_reachable_paragraphs():
    """Paragraph nodes guarantee a paragraph is *representable*, not that BFS can
    reach it: a pid-keyed node never merges, so it adds no bridge of its own. The
    seeded paragraph must be covered; disconnected ones still are not."""
    ex = _multi_para_ex({"p0"})
    qg = build_graph(ex, _triples_three_paras(), FakeEmbedder(),
                     connectivity={**CONN_FULL, "same_as_edges": False})
    assert "__para__p0" in qg.nodes
    assert "p0" in qg.covered_pids(), "the seeded paragraph must be scorable"


def test_paragraph_nodes_cover_all_paragraphs_once_components_are_joined():
    """With a shared entity joining the paragraphs, paragraph nodes make every
    one of them scorable."""
    paras = [Paragraph("p0", "A", "t", True), Paragraph("p1", "B", "t", False),
             Paragraph("p2", "C", "t", False)]
    ex = Example("q", "musique", "Who is linked to hub?", "x", paras, 2, {})
    tri = [T("hub", "r", "a1", "p0"), T("hub", "r", "b1", "p1"), T("hub", "r", "c1", "p2")]
    qg = build_graph(ex, tri, FakeEmbedder(),
                     connectivity={**CONN_FULL, "same_as_edges": False})
    assert qg.covered_pids() >= {"p0", "p1", "p2"}


def test_connectivity_off_by_default_matches_baseline():
    ex = _multi_para_ex({"p0"})
    tri = _triples_three_paras()
    a = build_graph(ex, tri, FakeEmbedder())
    b = build_graph(ex, tri, FakeEmbedder(), connectivity={})
    assert _structure(a) == _structure(b)
    assert a.added_links == {"title": 0, "same_as": 0, "para": 0}


def test_cross_paragraph_links_recorded_only_for_added_kinds():
    ex = _multi_para_ex({"p0"})
    qg = build_graph(ex, _triples_three_paras(), FakeEmbedder(), connectivity=CONN_FULL)
    assert all(kind != "triple" for kind, _a, _b in qg.cross_paragraph_links)


# --------------------------- scaffolding is never evidence (item 3 proof)
def test_scaffolding_edges_can_never_be_positive():
    """same_as / title / para edges must never carry y=1, whatever the gold flags."""
    ex = _multi_para_ex({"p0", "p1", "p2"})      # EVERY paragraph is gold
    qg = build_graph(ex, _triples_three_paras(), FakeEmbedder(), connectivity=CONN_FULL)
    for y, kind in zip(qg.edge_y, qg.edge_kind):
        if kind != "triple":
            assert y == 0, f"{kind} edge carried y=1 with all paragraphs gold"
    assert sum(qg.edge_y) > 0, "extracted triples from gold paragraphs must still be positive"


def test_scaffolding_alone_cannot_make_a_paragraph_scorable():
    """A paragraph whose only ego-graph edges are scaffolding is not rankable."""
    from src.routing import paragraph_scores

    ex = _multi_para_ex({"p0"})
    qg = build_graph(ex, _triples_three_paras(), FakeEmbedder(), connectivity=CONN_FULL)
    scores = paragraph_scores(qg, [1.0] * qg.n_edges)
    evidence_pids = {
        pid for pid, k in zip(qg.edge_pid, qg.edge_kind) if k == "triple"
    }
    assert set(scores) == evidence_pids
    assert qg.covered_pids() == evidence_pids
    # the hub-only view is strictly larger, and is a diagnostic rather than a score
    assert qg.covered_pids_any_edge() >= qg.covered_pids()


def test_para_edges_do_not_inflate_coverage():
    """Turning on paragraph nodes must not add a scorable paragraph on its own."""
    ex = _multi_para_ex({"p0"})
    tri = _triples_three_paras()
    without = build_graph(ex, tri, FakeEmbedder(),
                          connectivity={**CONN_FULL, "paragraph_nodes": False})
    with_ = build_graph(ex, tri, FakeEmbedder(), connectivity=CONN_FULL)
    extra = with_.covered_pids() - without.covered_pids()
    for pid in extra:
        kinds = {k for p, k in zip(with_.edge_pid, with_.edge_kind) if p == pid}
        assert "triple" in kinds, f"{pid} became scorable without any extracted edge"


def test_evidence_edges_counts_only_triples():
    ex = _multi_para_ex({"p0"})
    qg = build_graph(ex, _triples_three_paras(), FakeEmbedder(), connectivity=CONN_FULL)
    assert qg.evidence_edges == sum(1 for k in qg.edge_kind if k == "triple")
    assert qg.evidence_edges < qg.n_edges, "scaffolding should exist in this fixture"


def test_paragraph_nodes_carry_a_real_embedding_not_zeros():
    """L=0 feature parity requires every node to have a genuine embedding; a
    pid-keyed paragraph node embeds its paragraph title."""
    ex = _multi_para_ex({"p0"})
    qg = build_graph(ex, _triples_three_paras(), FakeEmbedder(), connectivity=CONN_FULL)
    for name, text, kinds in zip(qg.nodes, qg.node_text, qg.node_kind):
        if "para" in kinds:
            assert not text.startswith("__para__"), f"{name} would embed as zeros"
            assert text, "paragraph node needs embeddable text"
    d = to_pyg(qg, FakeCache(), hops=3)
    assert bool((d.x[:, :384].abs().sum(dim=1) > 0).all()), "a node entered with zeros"


def test_node_text_defaults_to_the_node_name():
    ex = make_ex(gold=("p0",))
    qg = build_graph(ex, [T("A", "r", "B", "p0")], FakeEmbedder(), merge_near_duplicates=False)
    assert qg.node_text == qg.nodes


def test_to_pyg_refuses_to_zero_fill_the_question():
    """A missing question embedding must raise, not silently zero the query.

    Zero-filling it makes e_v*q, cos(e_v,q) and cos(r_e,q) all zero, so the
    router runs query-blind and still produces plausible-looking scores -- which
    is what happened to the M11 extension's HotpotQA pools, whose questions had
    never been through the M3 embedding pass (D29).
    """
    import pytest

    ex = make_ex(question="about n0", gold=("p0",))
    qg = build_graph(ex, _chain(6), FakeEmbedder(), hops=3, merge_near_duplicates=False)

    class NoQuestionCache(FakeCache):
        def get(self, text, kind="misc"):
            if kind == "questions":
                return None
            return super().get(text, kind)

    with pytest.raises(KeyError, match="cached questions embedding"):
        to_pyg(qg, NoQuestionCache(), hops=3)

    # the same graph still builds once the question embedding is there
    d = to_pyg(qg, FakeCache(), hops=3)
    assert d.x.shape[0] == qg.n_nodes


def test_to_pyg_empty_graph_also_requires_the_question():
    """The n == 0 early return must not slip past the question check."""
    import pytest

    qg = build_graph(make_ex(), [], FakeEmbedder(), merge_near_duplicates=False)

    class NoQuestionCache(FakeCache):
        def get(self, text, kind="misc"):
            return None if kind == "questions" else super().get(text, kind)

    with pytest.raises(KeyError, match="cached questions embedding"):
        to_pyg(qg, NoQuestionCache(), hops=3)


def test_to_pyg_refuses_to_zero_fill_a_relation():
    """A missing relation embedding must raise too (D30).

    Zero-filling r_e leaves that edge's entire edge_attr signal-free -- r_e = 0
    and cos(r_e,q) = 0 -- which is how 10% of the M11 extension's edges were
    scored before this was caught. Nothing announced it: the scores stayed in
    range and the rankings stayed plausible.
    """
    import pytest

    ex = make_ex(question="about n0", gold=("p0",))
    qg = build_graph(ex, _chain(6), FakeEmbedder(), hops=3, merge_near_duplicates=False)

    class NoRelationCache(FakeCache):
        def get(self, text, kind="misc"):
            return None if kind == "relations" else super().get(text, kind)

    with pytest.raises(KeyError, match="cached relations embedding"):
        to_pyg(qg, NoRelationCache(), hops=3)


def test_to_pyg_refuses_to_zero_fill_an_entity():
    """Same rule for node texts: no feature is ever silently zeroed (D30)."""
    import pytest

    ex = make_ex(question="about n0", gold=("p0",))
    qg = build_graph(ex, _chain(6), FakeEmbedder(), hops=3, merge_near_duplicates=False)

    class NoEntityCache(FakeCache):
        def get(self, text, kind="misc"):
            return None if kind == "entities" else super().get(text, kind)

    with pytest.raises(KeyError, match="cached entities embedding"):
        to_pyg(qg, NoEntityCache(), hops=3)
