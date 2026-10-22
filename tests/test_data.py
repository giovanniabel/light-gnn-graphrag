"""Unit tests for pool building (no gold leakage), splits and cache keys.

agent.md Section 12 requires tests for: pool building (no gold leakage), seed
linking, label construction, metric functions, cache keys. Seed linking, labels
and metrics arrive with M5/M7; this file covers what M1 ships.
"""

from __future__ import annotations

import random
from dataclasses import replace

import pytest
import torch

from src.data import (
    GoldBlind,
    _musique_hops,
    _reservoir_by_stratum,
    _select_pool_pids,
    assert_disjoint,
    build_pool,
    carve_dev,
    gold_blind,
    pool_recall,
)
from src.embedder import EmbeddingCache, text_key
from src.schemas import Example, GoldLeakError, Paragraph


class FakeEmbedder:
    """Deterministic stand-in: no model download, no network."""

    dim = 8

    def _vec(self, t: str) -> torch.Tensor:
        rng = random.Random(hash(t) % (2**31))
        v = torch.tensor([rng.uniform(-1, 1) for _ in range(self.dim)])
        return v / v.norm()

    def encode(self, texts, kind="misc"):
        if isinstance(texts, str):
            return self._vec(texts)
        return torch.stack([self._vec(t) for t in texts])

    def encode_query(self, q):
        return self._vec(q) if isinstance(q, str) else torch.stack([self._vec(x) for x in q])


def make_example(n_paragraphs=20, n_gold=2, qid="2hop__1_2", dataset="musique") -> Example:
    rng = random.Random(0)
    paras = [
        Paragraph(
            pid=f"{qid}::p{i}",
            title=f"Title {i}",
            text=f"Paragraph {i} about topic {rng.randint(0, 99)} and some filler words.",
            is_gold=i < n_gold,
        )
        for i in range(n_paragraphs)
    ]
    rng.shuffle(paras)
    return Example(qid, dataset, "Who is the spouse of the performer?", "Someone", paras, 2)


# ------------------------------------------------------------- gold leakage
def test_goldblind_raises_on_is_gold():
    p = Paragraph("p0", "T", "text", True)
    b = GoldBlind(p)
    assert b.pid == "p0" and b.title == "T" and b.text == "text"
    for attr in ("is_gold", "gold", "is_supporting", "gold_pids"):
        with pytest.raises(GoldLeakError):
            getattr(b, attr)


def test_goldblind_is_read_only():
    b = GoldBlind(Paragraph("p0", "T", "text", True))
    with pytest.raises(AttributeError):
        b.pid = "other"


def test_select_pool_pids_cannot_see_gold():
    """The selector runs against GoldBlind views: a leak raises, not silently passes."""
    ex = make_example()
    pids = _select_pool_pids(ex.question, gold_blind(ex.paragraphs), FakeEmbedder(), 10)
    assert len(pids) == 10
    assert len(set(pids)) == 10


def test_build_pool_output_invariant_to_gold_flags():
    """Strongest leakage check: flipping every gold flag must not change the pool."""
    ex = make_example(n_paragraphs=20, n_gold=2)
    flipped = replace(
        ex,
        paragraphs=[replace(p, is_gold=not p.is_gold) for p in ex.paragraphs],
    )
    a = [p.pid for p in build_pool(ex, FakeEmbedder(), 10).paragraphs]
    b = [p.pid for p in build_pool(flipped, FakeEmbedder(), 10).paragraphs]
    assert a == b


def test_build_pool_all_gold_removed_still_same_pool():
    ex = make_example(n_paragraphs=20, n_gold=2)
    nogold = replace(ex, paragraphs=[replace(p, is_gold=False) for p in ex.paragraphs])
    a = [p.pid for p in build_pool(ex, FakeEmbedder(), 10).paragraphs]
    b = [p.pid for p in build_pool(nogold, FakeEmbedder(), 10).paragraphs]
    assert a == b


# ------------------------------------------------------------- pool mechanics
def test_build_pool_size_and_gold_flags_reattached():
    ex = make_example(n_paragraphs=20, n_gold=2)
    pooled = build_pool(ex, FakeEmbedder(), 10)
    assert len(pooled.paragraphs) == 10
    # gold flags come back from the originals, matched by pid
    orig = {p.pid: p.is_gold for p in ex.paragraphs}
    assert all(p.is_gold == orig[p.pid] for p in pooled.paragraphs)


def test_build_pool_noop_when_pool_already_small():
    """HotpotQA ships exactly 10, so pooling must keep all of them in order."""
    ex = make_example(n_paragraphs=10, n_gold=2, dataset="hotpotqa")
    pooled = build_pool(ex, FakeEmbedder(), 10)
    assert [p.pid for p in pooled.paragraphs] == [p.pid for p in ex.paragraphs]


def test_build_pool_is_deterministic():
    ex = make_example()
    a = [p.pid for p in build_pool(ex, FakeEmbedder(), 10).paragraphs]
    b = [p.pid for p in build_pool(ex, FakeEmbedder(), 10).paragraphs]
    assert a == b


def test_build_pool_preserves_meta():
    ex = replace(make_example(), meta={"hotpot_type": "bridge"})
    assert build_pool(ex, FakeEmbedder(), 10).meta == {"hotpot_type": "bridge"}


# ------------------------------------------------------------- splits
def test_assert_disjoint_passes_and_fails():
    a = [make_example(qid="q1"), make_example(qid="q2")]
    b = [make_example(qid="q3")]
    assert assert_disjoint(a, b) == set()
    with pytest.raises(AssertionError):
        assert_disjoint(a, [make_example(qid="q1")])


def test_carve_dev_partitions_without_overlap():
    train = [make_example(qid=f"q{i}") for i in range(100)]
    tr, dev = carve_dev(train, 0.10, seed=1234)
    assert len(dev) == 10 and len(tr) == 90
    assert not ({e.qid for e in tr} & {e.qid for e in dev})
    assert {e.qid for e in tr} | {e.qid for e in dev} == {e.qid for e in train}


def test_carve_dev_is_seed_stable():
    train = [make_example(qid=f"q{i}") for i in range(100)]
    d1 = [e.qid for e in carve_dev(train, 0.1, 1234)[1]]
    d2 = [e.qid for e in carve_dev(train, 0.1, 1234)[1]]
    assert d1 == d2


# ------------------------------------------------------------- sampling
def test_reservoir_stratified_balances_strata():
    rows = [(f"{h}hop", {"id": f"{h}-{i}"}) for h in (2, 3, 4) for i in range(500)]
    got = _reservoir_by_stratum(iter(rows), 90, seed=1234)
    assert len(got) == 90
    from collections import Counter

    c = Counter(r["id"].split("-")[0] for r in got)
    assert set(c) == {"2", "3", "4"} and all(v == 30 for v in c.values())


def test_reservoir_handles_small_stratum():
    rows = [("a", {"id": f"a{i}"}) for i in range(5)] + [("b", {"id": f"b{i}"}) for i in range(100)]
    got = _reservoir_by_stratum(iter(rows), 40, seed=1)
    assert len(got) == 40            # shortfall in 'a' refilled from 'b'
    assert len({r["id"] for r in got}) == 40


def test_reservoir_is_seed_stable():
    rows = [("a", {"id": str(i)}) for i in range(1000)]
    a = [r["id"] for r in _reservoir_by_stratum(iter(rows), 50, 7)]
    b = [r["id"] for r in _reservoir_by_stratum(iter(rows), 50, 7)]
    assert a == b


def test_reservoir_caps_at_n_when_source_smaller():
    rows = [("a", {"id": str(i)}) for i in range(10)]
    assert len(_reservoir_by_stratum(iter(rows), 50, 1)) == 10


# ------------------------------------------------------------- hop parsing
@pytest.mark.parametrize(
    "qid,expected",
    [
        ("2hop__460946_294723", 2),
        ("3hop1__443556_726923_858380", 3),
        ("3hop2__1_2_3", 3),
        ("4hop1__a_b_c_d", 4),
        ("4hop3__a_b_c_d", 4),
    ],
)
def test_musique_hops(qid, expected):
    assert _musique_hops(qid) == expected


# ------------------------------------------------------------- metrics
def test_pool_recall_counts_correctly():
    orig = make_example(n_paragraphs=20, n_gold=2)
    pooled = build_pool(orig, FakeEmbedder(), 10)
    r = pool_recall([pooled], [orig])
    assert r["gold_paragraphs_total"] == 2
    assert r["gold_paragraphs_in_pool"] == len(pooled.gold_pids)
    assert 0.0 <= r["pool_recall"] <= 1.0


def test_pool_recall_perfect_when_nothing_dropped():
    orig = make_example(n_paragraphs=10, n_gold=2)
    pooled = build_pool(orig, FakeEmbedder(), 10)
    r = pool_recall([pooled], [orig])
    assert r["pool_recall"] == 1.0 and r["all_gold_rate"] == 1.0


# ------------------------------------------------------------- cache keys
def test_text_key_is_stable_and_model_scoped():
    assert text_key("hello", "m1") == text_key("hello", "m1")
    assert text_key("hello", "m1") != text_key("hello", "m2")
    assert text_key("hello", "m1") != text_key("hell", "o" + "m1")   # no delimiter collision


def test_embedding_cache_roundtrip_and_flush(tmp_path):
    c = EmbeddingCache(tmp_path, "BAAI/bge-small-en-v1.5")
    assert c.get("abc", "questions") is None
    v = torch.randn(384)
    c.put("abc", v, "questions")
    assert torch.allclose(c.get("abc", "questions"), v)
    assert c.flush() == 1
    c2 = EmbeddingCache(tmp_path, "BAAI/bge-small-en-v1.5")
    assert torch.allclose(c2.get("abc", "questions"), v)       # survives restart
    assert c2.get("abc", "entities") is None                   # kinds are separate
    c3 = EmbeddingCache(tmp_path, "other-model")
    assert c3.get("abc", "questions") is None                  # model-scoped


def test_embedding_cache_stores_rows_not_views(tmp_path):
    """Regression: storing a row of an encode() batch stored a VIEW, and
    torch.save serializes a view's whole underlying storage. That wrote ~4096x
    more bytes per entry and grew the cache to 63 GB instead of 0.3 GB."""
    import os

    c = EmbeddingCache(tmp_path, "m")
    batch = torch.randn(4096, 384)          # one encode() batch
    for i in range(4):
        c.put(f"text{i}", batch[i], "entities")
    c.flush()
    shards = list((tmp_path / "entities").glob("*.pt"))
    total = sum(os.path.getsize(p) for p in shards)
    assert total < 200_000, f"{total} bytes for 4 vectors -- a view leaked into the file"
    c2 = EmbeddingCache(tmp_path, "m")
    for i in range(4):
        assert torch.allclose(c2.get(f"text{i}", "entities"), batch[i]), "value changed"


def test_embedding_cache_entries_are_contiguous_and_own_their_storage(tmp_path):
    c = EmbeddingCache(tmp_path, "m")
    batch = torch.randn(64, 384)
    c.put("t", batch[7], "entities")
    v = c.get("t", "entities")
    assert v.is_contiguous() and v.storage_offset() == 0
    assert v.untyped_storage().nbytes() // 4 == 384
