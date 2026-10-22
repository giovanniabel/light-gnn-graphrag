"""Tests for Section 7 metrics, especially AllGold@4's two conditionings (D4)."""

from __future__ import annotations

import pytest

from evaluation.metrics import (
    all_gold,
    all_gold_conditional,
    average_precision,
    em_f1,
    gold_in_pool,
    normalize_squad,
    paired_bootstrap,
    sp_recall,
)
from src.schemas import Example, Paragraph, Routed, StageStats


def ex_with(pool, gold, qid="q1"):
    return Example(qid, "musique", "q?", "ans",
                   [Paragraph(p, f"T{p}", "t", p in gold) for p in pool], 2, {})


def routed(pids):
    return Routed(paragraph_ids=list(pids), edges=[], stats=StageStats())


# ------------------------------------------------------------ SP-Recall
def test_sp_recall_counts_gold_in_topk():
    e = ex_with(["p0", "p1", "p2", "p3"], {"p0", "p1"})
    assert sp_recall(routed(["p0", "p9", "p8", "p7"]), e, k=4) == pytest.approx(0.5)
    assert sp_recall(routed(["p0", "p1", "p8", "p7"]), e, k=4) == 1.0


def test_sp_recall_respects_k():
    e = ex_with(["p0", "p1"], {"p0", "p1"})
    assert sp_recall(routed(["p0", "p1"]), e, k=1) == pytest.approx(0.5)


# ------------------------------------------------------------ AllGold, unconditional
def test_all_gold_is_all_or_nothing():
    e = ex_with(["p0", "p1", "p2", "p3"], {"p0", "p1"})
    assert all_gold(routed(["p0", "p1", "p2", "p3"]), e, k=4) == 1.0
    assert all_gold(routed(["p0", "p2", "p3", "p9"]), e, k=4) == 0.0


def test_all_gold_accepts_a_plain_list():
    e = ex_with(["p0", "p1"], {"p0"})
    assert all_gold(["p0", "p1"], e, k=2) == 1.0


# ------------------------------------------------------------ D4 conditional
def test_conditional_restricts_the_question_set_not_the_gold_set():
    """The question whose gold was dropped by pooling must leave the denominator
    entirely -- not have its gold set quietly shrunk."""
    kept = ex_with(["p0", "p1", "p2", "p3"], {"p0", "p1"}, qid="kept")
    lost = ex_with(["p0", "p2", "p3", "p4"], {"p0"}, qid="lost")   # gold p1 not pooled
    orig = {"kept": {"p0", "p1"}, "lost": {"p0", "p1"}}
    r = {"kept": routed(["p0", "p1", "p2", "p3"]), "lost": routed(["p0", "p2", "p3", "p4"])}
    out = all_gold_conditional(r, [kept, lost], orig, k=4)
    assert out["n_all"] == 2
    assert out["n_conditional"] == 1, "only 'kept' has all its original gold in the pool"
    assert out["conditional"] == 1.0
    assert out["unconditional"] == pytest.approx(0.5)


def test_conditional_differs_from_shrinking_the_gold_set():
    """Shrinking gold would score 'lost' as a hit; D4's conditioning excludes it."""
    lost = ex_with(["p0", "p2"], {"p0"}, qid="lost")
    orig = {"lost": {"p0", "p1"}}
    out = all_gold_conditional({"lost": routed(["p0", "p2"])}, [lost], orig, k=2)
    assert out["n_conditional"] == 0 and out["conditional"] is None
    assert out["unconditional"] == 0.0, "p1 was never retrievable, so this is a miss"


def test_conditional_reports_its_denominator():
    exs = [ex_with(["p0", "p1"], {"p0"}, qid=f"q{i}") for i in range(10)]
    orig = {e.qid: {"p0"} for e in exs}
    r = {e.qid: routed(["p0", "p1"]) for e in exs}
    out = all_gold_conditional(r, exs, orig, k=2)
    assert out["n_conditional"] == 10 and out["conditioning_rate"] == 1.0


def test_gold_in_pool():
    e = ex_with(["p0", "p2"], {"p0"})
    assert gold_in_pool(e, {"p0"}) is True
    assert gold_in_pool(e, {"p0", "p1"}) is False


# ------------------------------------------------------------ EM / F1
def test_em_f1_normalization():
    assert em_f1("The  Beatles.", "beatles")[0] == 1.0
    em, f1 = em_f1("John Lennon", "Lennon")
    assert em == 0.0 and 0 < f1 < 1


def test_em_f1_uses_best_alias():
    assert em_f1("Bob", ["Robert", "Bob"])[0] == 1.0


def test_em_f1_no_overlap():
    assert em_f1("cat", "dog") == (0.0, 0.0)


def test_normalize_squad_strips_articles_and_punctuation():
    assert normalize_squad("The, quick! brown") == "quick brown"


# ------------------------------------------------------------ AP
def test_average_precision_bounds():
    assert average_precision([0.9, 0.1], [1, 0]) == 1.0
    assert average_precision([0.9, 0.1], [0, 0]) == 0.0
    assert 0 < average_precision([0.9, 0.8, 0.7], [0, 1, 1]) < 1


def test_average_precision_is_rank_based_not_threshold_based():
    a = average_precision([0.99, 0.98, 0.01], [1, 1, 0])
    b = average_precision([0.6, 0.5, 0.4], [1, 1, 0])
    assert a == b == 1.0


# ------------------------------------------------------------ bootstrap
def test_paired_bootstrap_detects_a_real_difference():
    a = [1.0] * 40 + [0.0] * 10
    b = [0.0] * 40 + [1.0] * 10
    ci = paired_bootstrap(a, b, n=2000)
    assert ci.point > 0 and ci.excludes_zero


def test_paired_bootstrap_on_identical_inputs_contains_zero():
    a = [1.0, 0.0] * 25
    ci = paired_bootstrap(a, list(a), n=2000)
    assert ci.point == 0.0 and not ci.excludes_zero


def test_paired_bootstrap_requires_equal_lengths():
    with pytest.raises(ValueError):
        paired_bootstrap([1.0], [1.0, 0.0])


def test_paired_bootstrap_is_seed_stable():
    a, b = [1.0, 0.0] * 20, [0.0, 0.0] * 20
    x = paired_bootstrap(a, b, n=500, seed=7)
    y = paired_bootstrap(a, b, n=500, seed=7)
    assert (x.lo, x.hi) == (y.lo, y.hi)
