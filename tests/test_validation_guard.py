"""Tests for the validation-split guard (agent.md §12, D12).

Every test redirects the log to tmp_path, so the real
results/validation_access.log stays a clean audit trail.
"""

from __future__ import annotations

import pytest

from src import validation_guard as vg


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(vg, "LOG", tmp_path / "validation_access.log")
    monkeypatch.setattr(vg, "_allowed", False)
    monkeypatch.setattr(vg, "_reason", "")
    monkeypatch.delenv("ALLOW_VALIDATION", raising=False)
    monkeypatch.delenv("ALLOW_VALIDATION_REASON", raising=False)
    yield


def test_require_refuses_by_default():
    with pytest.raises(vg.ValidationAccessError):
        vg.require("test_context")


def test_refusal_is_logged_as_denied():
    with pytest.raises(vg.ValidationAccessError):
        vg.require("some_reader")
    log = vg.log_text()
    assert "DENIED" in log and "some_reader" in log
    assert "ALLOWED" not in log


def test_allow_unlocks_and_logs_the_reason():
    vg.allow("M7 retrieval eval")
    vg.require("load_examples")
    log = vg.log_text()
    assert "UNLOCK" in log and "M7 retrieval eval" in log
    assert "ALLOWED\tload_examples" in log


def test_env_var_unlocks(monkeypatch):
    monkeypatch.setenv("ALLOW_VALIDATION", "1")
    monkeypatch.setenv("ALLOW_VALIDATION_REASON", "scripted eval")
    vg.require("script")
    assert "ALLOWED" in vg.log_text() and "scripted eval" in vg.log_text()


def test_every_access_appends_a_line():
    vg.allow("r")
    for i in range(3):
        vg.require(f"reader{i}")
    assert len([l for l in vg.log_text().splitlines() if "ALLOWED" in l]) == 3


def test_train_split_is_never_guarded():
    """Only validation is gated; training must stay freely readable."""
    from src.data import load_examples
    import inspect

    src = inspect.getsource(load_examples)
    assert 'if split == "validation"' in src
    assert src.count("require(") == 1


def test_guard_is_hooked_into_every_validation_reader():
    """A new reader of the validation split must not slip past the guard."""
    import inspect

    from scripts.m1_data import load_pools
    from scripts.m6_train import load_graphs

    for fn in (load_pools, load_graphs):
        s = inspect.getsource(fn)
        assert 'split == "validation"' in s and "require(" in s, f"{fn.__name__} ungated"
