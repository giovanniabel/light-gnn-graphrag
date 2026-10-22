"""Tests for the extraction guards and RSS accounting (deviation D10)."""

from __future__ import annotations

import psutil

from src.profiling import (
    PRESSURE_CRITICAL,
    PRESSURE_NORMAL,
    ExtractionGuard,
    memory_pressure_level,
    ollama_processes,
    own_tree_rss_mb,
    tree_rss_mb,
)


# ----------------------------------------------------------- RSS accounting
def test_own_tree_rss_is_positive_and_below_full_tree():
    own, full = own_tree_rss_mb(), tree_rss_mb()
    assert own > 0, "own tree RSS must never be zero -- it once was, via a bad ollama match"
    assert own <= full + 1.0, "excluding Ollama cannot exceed including it"


def test_ollama_match_does_not_capture_this_process():
    """Regression: matching 'ollama' anywhere in the cmdline classified our own
    process (whose argv mentioned ollama) as Ollama, zeroing own_tree_rss_mb."""
    assert psutil.Process().pid not in {p.pid for p in ollama_processes()}


def test_memory_pressure_level_is_a_known_value():
    assert memory_pressure_level() in (1, 2, 4)


# ----------------------------------------------------------- guard: healthy
def _feed(guard, n_ticks, t_p=1.6, rss=800.0, pressure=PRESSURE_NORMAL, step=25):
    """Drive the guard with synthetic observations."""
    extracted = 0
    cum = 0.0
    for _ in range(n_ticks):
        extracted += step
        cum += step * (t_p(extracted) if callable(t_p) else t_p)
        guard.history.append({
            "extracted": extracted, "cum_seconds": cum, "elapsed_s": cum,
            "own_rss_mb": rss(extracted) if callable(rss) else rss,
            "pressure": pressure(extracted) if callable(pressure) else pressure,
            "swap_mb": 0.0, "available_mb": 2000.0,
        })
        guard._critical_streak = (
            guard._critical_streak + 1
            if guard.history[-1]["pressure"] >= PRESSURE_CRITICAL else 0
        )
        window = [h for h in guard.history
                  if guard.baseline_after_paragraphs <= h["extracted"] <= guard.baseline_until_paragraphs]
        if guard.baseline_rss_mb is None and window and extracted >= guard.baseline_until_paragraphs:
            import statistics
            guard.baseline_rss_mb = statistics.median(h["own_rss_mb"] for h in window)
            f, l = window[0], window[-1]
            dn = l["extracted"] - f["extracted"]
            if dn:
                guard.baseline_t_p = (l["cum_seconds"] - f["cum_seconds"]) / dn
    return guard


def test_healthy_run_does_not_abort():
    g = _feed(ExtractionGuard(), 40)
    assert g.check() is None


def test_swap_is_not_an_abort_condition():
    """D10: the first M4 died on system swap while the workload was healthy."""
    g = ExtractionGuard()
    _feed(g, 40)
    for h in g.history:
        h["swap_mb"] = 5000.0          # enormous system-wide swap
    assert g.check() is None, "system-wide swap must not abort an extraction run"


# ----------------------------------------------------------- guard: (a) RSS
def test_sustained_rss_growth_aborts():
    g = _feed(ExtractionGuard(), 40, rss=lambda n: 800.0 + max(0, n - 200) * 3.0)
    r = g.check()
    assert r and "RSS grew steadily" in r


def test_single_rss_spike_does_not_abort():
    g = ExtractionGuard()
    _feed(g, 40)
    g.history[-1]["own_rss_mb"] = 9000.0      # one bad reading
    assert g.check() is None, "a lone spike must not abort; growth must be sustained"


# ----------------------------------------------------------- guard: (b) t_p
def test_throughput_degradation_aborts():
    g = _feed(ExtractionGuard(), 40, t_p=lambda n: 1.6 if n <= 200 else 6.0)
    r = g.check()
    assert r and "throughput degraded" in r


def test_mild_slowdown_within_tolerance_does_not_abort():
    g = _feed(ExtractionGuard(), 40, t_p=lambda n: 1.6 if n <= 200 else 2.1)  # 1.3x
    assert g.check() is None


def test_rolling_t_p_uses_recent_window_only():
    g = _feed(ExtractionGuard(), 40, t_p=lambda n: 10.0 if n <= 100 else 1.0)
    roll = g.rolling_t_p()
    assert roll is not None and roll < 2.0, "rolling t_p must reflect recent paragraphs"


# ----------------------------------------------------------- guard: (c) pressure
def test_sustained_critical_pressure_aborts():
    g = _feed(ExtractionGuard(), 40,
              pressure=lambda n: PRESSURE_CRITICAL if n > 925 else PRESSURE_NORMAL)
    r = g.check()
    assert r and "memory pressure critical" in r


def test_transient_critical_pressure_does_not_abort():
    g = ExtractionGuard(pressure_consecutive=3)
    _feed(g, 40, pressure=lambda n: PRESSURE_CRITICAL if n == 500 else PRESSURE_NORMAL)
    assert g.check() is None


# ----------------------------------------------------------- warnings
def test_warning_throttle_is_once_per_interval():
    g = ExtractionGuard(warn_every_s=60.0)
    assert g.should_warn(0.0) is True
    assert g.should_warn(30.0) is False
    assert g.should_warn(61.0) is True
