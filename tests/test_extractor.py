"""Unit tests for cache keys, JSON parsing and truncation salvage (agent.md Section 12)."""

from __future__ import annotations

import json

from src.extractor import (
    PROMPT_VERSION,
    TripleCache,
    cache_key,
    parse_triples,
    salvage_triples,
    triple_schema,
    truncate_words,
)
from src.schemas import Paragraph, Triple

TAG = "qwen2.5:1.5b-instruct-q4_K_M"


# ------------------------------------------------------------- cache keys
def test_cache_key_components_all_matter():
    """Key is sha1(model_tag + prompt_version + paragraph_text) -- Section 8.1."""
    base = cache_key("some text", TAG, "v2")
    assert base == cache_key("some text", TAG, "v2")
    assert base != cache_key("some text", TAG, "v1")        # prompt version
    assert base != cache_key("some text", "other:tag", "v2")  # model tag
    assert base != cache_key("other text", TAG, "v2")       # paragraph text


def test_cache_key_has_no_delimiter_collision():
    assert cache_key("a", "b" + "\x00" + "c", "v2") != cache_key("a", "b", "c")


def test_cache_key_uses_truncated_text(tmp_path):
    """Keying must match what is actually sent to the model (120 words)."""
    c = TripleCache(tmp_path / "t.jsonl", TAG, "v2")
    long_a = " ".join(["word"] * 300)
    long_b = long_a + " extra tail beyond the cut"
    assert c.key_for(long_a) == c.key_for(long_b)


def test_truncate_words():
    assert truncate_words("a b c d", 2) == "a b"
    assert truncate_words("a b", 10) == "a b"


# ------------------------------------------------------------- cache io
def test_cache_append_persists_and_flushes(tmp_path):
    path = tmp_path / "triples.jsonl"
    c = TripleCache(path, TAG, "v2")
    t = [Triple("A", "rel", "B", "q1::p0")]
    c.append(c.key_for("text one"), "q1::p0", t, json_failed=False, llm_tokens=10)
    assert path.exists()                       # flushed immediately, before close
    assert c.n_lines == 1
    c.close()

    c2 = TripleCache(path, TAG, "v2")          # survives a restart
    got = c2.get("text one")
    assert got and got[0].head == "A" and got[0].tail == "B"
    assert "text one" in c2


def test_cache_ignores_truncated_tail_line(tmp_path):
    """A run killed mid-write leaves a partial last line; loading must not crash."""
    path = tmp_path / "triples.jsonl"
    c = TripleCache(path, TAG, "v2")
    c.append(c.key_for("good"), "q::p0", [Triple("A", "r", "B", "q::p0")])
    c.close()
    with open(path, "a") as f:
        f.write('{"key": "half-written", "tripl')
    c2 = TripleCache(path, TAG, "v2")
    assert len(c2) == 1 and c2.get("good")


def test_cache_restamps_provenance_across_questions(tmp_path):
    """Dedupe is by paragraph text, so a cache hit must carry the asking pid."""
    from src.extractor import extract_paragraph

    c = TripleCache(tmp_path / "t.jsonl", TAG, "v2")
    shared = "Shared paragraph text."
    c.append(c.key_for(shared), "qA::p0", [Triple("A", "r", "B", "qA::p0")])
    p = Paragraph("qB::p7", "Title", shared, False)
    got = extract_paragraph(p, c, client=None)   # cache hit: no client needed
    assert [t.pid for t in got] == ["qB::p7"]


# ------------------------------------------------------------- parsing
def test_parse_triples_list_form():
    out, failed = parse_triples('{"triples": [["A", "rel", "B"]]}', "p1")
    assert not failed and len(out) == 1
    assert (out[0].head, out[0].rel, out[0].tail, out[0].pid) == ("A", "rel", "B", "p1")


def test_parse_triples_dict_form():
    out, failed = parse_triples(
        '{"triples": [{"head": "A", "relation": "rel", "tail": "B"}]}', "p1"
    )
    assert not failed and len(out) == 1 and out[0].rel == "rel"


def test_parse_triples_rejects_malformed():
    for bad in ("not json at all", "", "{", '{"triples": "nope"}'):
        out, failed = parse_triples(bad, "p1")
        assert failed and out == []


def test_parse_triples_empty_list_is_not_a_failure():
    """Valid JSON that yielded nothing is an empty extraction, not a parse failure."""
    out, failed = parse_triples('{"triples": []}', "p1")
    assert not failed and out == []


def test_parse_triples_drops_incomplete_and_blank():
    out, failed = parse_triples(
        '{"triples": [["A", "rel", "B"], ["only", "two"], ["", "rel", "B"], ["C", "r2", "D"]]}',
        "p1",
    )
    assert not failed and [(t.head, t.tail) for t in out] == [("A", "B"), ("C", "D")]


def test_parse_triples_stamps_pid_on_every_triple():
    out, _ = parse_triples('{"triples": [["A","r","B"],["C","r","D"]]}', "qX::p3")
    assert {t.pid for t in out} == {"qX::p3"}


# ------------------------------------------------------------- salvage
def test_salvage_recovers_leading_triples_from_truncated_output():
    cut = '{"triples": [["Libya", "capital", "Tripoli"], ["Libya", "area", "1.76M km2"], ["Libya", "largest'
    out = salvage_triples(cut, "p1")
    assert [(t.head, t.rel, t.tail) for t in out] == [
        ("Libya", "capital", "Tripoli"),
        ("Libya", "area", "1.76M km2"),
    ]
    assert {t.pid for t in out} == {"p1"}


def test_salvage_handles_escapes():
    cut = '{"triples": [["A \\"quoted\\" name", "rel", "B"], ["C", "r'
    out = salvage_triples(cut, "p1")
    assert out and out[0].head == 'A "quoted" name'


def test_salvage_returns_nothing_when_no_complete_triple():
    assert salvage_triples('{"triples": [["Libya", "capi', "p1") == []
    assert salvage_triples("garbage", "p1") == []


def test_salvage_agrees_with_parse_on_complete_json():
    good = '{"triples": [["A", "r", "B"], ["C", "r2", "D"]]}'
    parsed, _ = parse_triples(good, "p1")
    rescued = salvage_triples(good, "p1")
    assert [(t.head, t.rel, t.tail) for t in parsed] == [(t.head, t.rel, t.tail) for t in rescued]


# ------------------------------------------------------------- schema
def test_triple_schema_bounds_the_triple_count():
    s = triple_schema(6)
    assert s["properties"]["triples"]["maxItems"] == 6
    item = s["properties"]["triples"]["items"]
    assert item["minItems"] == item["maxItems"] == 3
    json.dumps(s)   # must be serializable as Ollama's `format`


# ------------------------------------------------------------- D8 regression
def test_effective_prompt_version_folds_in_max_triples():
    """Regression: extract_all wrote keys under 'v2' while the M4 report read
    'v2-mt6', so a fully populated cache reported 0% extraction coverage."""
    from src.extractor import effective_prompt_version

    cfg = {"extraction": {"prompt_version": "v2", "max_triples": 6}}
    assert effective_prompt_version(cfg) == "v2-mt6"
    cfg["extraction"]["max_triples"] = 4
    assert effective_prompt_version(cfg) == "v2-mt4"


def test_extract_all_and_readers_agree_on_cache_keys(tmp_path, monkeypatch):
    """The writer and any later reader must resolve the same key for one text."""
    from src.extractor import effective_prompt_version

    cfg = {"extraction": {"prompt_version": "v2", "max_triples": 6}}
    pv = effective_prompt_version(cfg)
    writer = TripleCache(tmp_path / "t.jsonl", TAG, pv)
    writer.append(writer.key_for("a paragraph"), "q::p0", [Triple("A", "r", "B", "q::p0")])
    writer.close()
    reader = TripleCache(tmp_path / "t.jsonl", TAG, effective_prompt_version(cfg))
    assert reader.get("a paragraph"), "reader must resolve the writer's key"


def test_different_max_triples_do_not_collide(tmp_path):
    from src.extractor import effective_prompt_version

    c6 = TripleCache(tmp_path / "t.jsonl", TAG,
                     effective_prompt_version({"extraction": {"prompt_version": "v2", "max_triples": 6}}))
    c6.append(c6.key_for("shared"), "q::p0", [Triple("A", "r", "B", "q::p0")])
    c6.close()
    c4 = TripleCache(tmp_path / "t.jsonl", TAG,
                     effective_prompt_version({"extraction": {"prompt_version": "v2", "max_triples": 4}}))
    assert c4.get("shared") is None, "mt=4 must not read mt=6's triples"
