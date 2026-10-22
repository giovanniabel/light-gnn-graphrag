"""Stage timer, token counter and process-tree RSS sampler (agent.md Section 7.3).

Rules this module exists to enforce:
  * RSS is the whole process tree PLUS the Ollama server/runner processes.
    Python-process-only RSS must never be used -- it undercounts every method
    that calls Ollama.
  * An idle baseline is measured before the stage; both the absolute peak and
    the delta are reported.
  * Swap growth > profiling.swap_invalidate_mb invalidates the run.
"""

from __future__ import annotations

import statistics
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import psutil

from .schemas import StageStats, Usage

MB = 1024.0 * 1024.0
# Matched on the EXECUTABLE, never the full cmdline: a cmdline substring match
# classifies any process that merely mentions "ollama" in its arguments as an
# Ollama process -- including our own, which made own_tree_rss_mb() report 0 MB.
OLLAMA_EXE_NAMES = {"ollama", "ollama_llama_server", "ollama-runner", "llama-server"}
OLLAMA_PATH_HINT = "/ollama.app/"


def ollama_processes() -> list[psutil.Process]:
    """Every live Ollama process (server and model runners)."""
    me = psutil.Process().pid
    out = []
    for p in psutil.process_iter(["name", "exe"]):
        if p.pid == me:
            continue
        try:
            name = (p.info["name"] or "").lower()
            exe = (p.info["exe"] or "").lower()
            if name in OLLAMA_EXE_NAMES or OLLAMA_PATH_HINT in exe:
                out.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return out


def tree_rss_mb(root: psutil.Process | None = None) -> float:
    """RSS in MB of this process tree plus all Ollama processes."""
    root = root or psutil.Process()
    procs = [root]
    try:
        procs += root.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    procs += ollama_processes()

    seen: set[int] = set()
    total = 0.0
    for p in procs:
        if p.pid in seen:
            continue
        seen.add(p.pid)
        try:
            total += p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total / MB


def swap_used_mb() -> float:
    return psutil.swap_memory().used / MB


def virtual_memory_snapshot() -> dict:
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    return {
        "vm_total_mb": vm.total / MB,
        "vm_available_mb": vm.available / MB,
        "vm_percent": vm.percent,
        "swap_used_mb": sw.used / MB,
    }


class RSSSampler:
    """Background sampler of process-tree + Ollama RSS."""

    def __init__(self, interval_s: float = 0.1):
        self.interval_s = interval_s
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._root = psutil.Process()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(tree_rss_mb(self._root))
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    @property
    def peak_mb(self) -> float:
        return max(self.samples) if self.samples else 0.0

    @property
    def mean_mb(self) -> float:
        return statistics.fmean(self.samples) if self.samples else 0.0


@dataclass
class TokenCounter:
    """Sums prompt_eval_count + eval_count across Ollama responses (Section 7.3)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    def add_response(self, resp: dict) -> Usage:
        u = Usage(
            prompt_tokens=int(resp.get("prompt_eval_count", 0) or 0),
            completion_tokens=int(resp.get("eval_count", 0) or 0),
        )
        self.prompt_tokens += u.prompt_tokens
        self.completion_tokens += u.completion_tokens
        self.calls += 1
        return u

    def add_usage(self, u: Usage) -> None:
        self.prompt_tokens += u.prompt_tokens
        self.completion_tokens += u.completion_tokens
        self.calls += 1

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class StageProfiler:
    """Context manager measuring one stage: indexing | routing | reading.

    with StageProfiler("routing") as prof:
        ...
        prof.tokens.add_response(resp)
    prof.stats  -> StageStats
    """

    stage: str
    interval_s: float = 0.1
    baseline_settle_s: float = 0.2
    swap_invalidate_mb: float = 500.0
    tokens: TokenCounter = field(default_factory=TokenCounter)
    per_item_s: list[float] = field(default_factory=list)
    stats: StageStats | None = None
    invalidated: bool = False

    def __post_init__(self) -> None:
        self._sampler = RSSSampler(self.interval_s)
        self._baseline = 0.0
        self._swap0 = 0.0
        self._t0 = 0.0

    def __enter__(self) -> StageProfiler:
        time.sleep(self.baseline_settle_s)
        self._baseline = tree_rss_mb()
        self._swap0 = swap_used_mb()
        self._sampler.start()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> bool:
        elapsed = time.perf_counter() - self._t0
        self._sampler.stop()
        swap_delta = swap_used_mb() - self._swap0
        peak = max(self._sampler.peak_mb, self._baseline)
        self.stats = StageStats(
            latency_s=elapsed,
            llm_tokens=self.tokens.total,
            peak_rss_mb=peak,
            delta_rss_mb=peak - self._baseline,
            baseline_rss_mb=self._baseline,
            swap_delta_mb=swap_delta,
            stage=self.stage,
        )
        self.invalidated = swap_delta > self.swap_invalidate_mb
        return False

    @contextmanager
    def item(self):
        """Time one question/paragraph inside the stage (time.perf_counter)."""
        t = time.perf_counter()
        try:
            yield
        finally:
            self.per_item_s.append(time.perf_counter() - t)

    def latency_summary(self) -> dict:
        """Median and p95 per item (Section 7.3)."""
        xs = sorted(self.per_item_s)
        if not xs:
            return {"n": 0, "median_s": None, "p95_s": None}
        return {
            "n": len(xs),
            "median_s": statistics.median(xs),
            "p95_s": xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))],
            "mean_s": statistics.fmean(xs),
        }


# --------------------------------------------------------------------------
# Extraction guards (deviation D10)
#
# System-wide swap is the wrong abort signal for a long extraction on an 8 GB
# machine: it is a global counter, so an unrelated macOS daemon allocating a
# gigabyte aborts a run that is behaving perfectly. That is not hypothetical --
# it killed the first M4 attempt at 250/6489 paragraphs while our own RSS was
# *falling* (1448 -> 1089 MB) and t_p was flat at 1.65 s.
#
# These guards watch our own workload instead: our process tree excluding Ollama,
# our throughput, and sustained OS memory pressure. System-wide swap is still
# sampled and logged as a warning, and remains a hard invalidation rule for the
# M7/M8 efficiency measurements (Section 7.3), where it is the quantity under
# measurement rather than ambient noise.
# --------------------------------------------------------------------------

PRESSURE_NORMAL, PRESSURE_WARN, PRESSURE_CRITICAL = 1, 2, 4


def own_tree_rss_mb(root: psutil.Process | None = None) -> float:
    """RSS of this process tree, EXCLUDING Ollama (the opposite of tree_rss_mb).

    Section 7.3 requires Ollama in the memory *metric*; this is not that metric,
    it is a leak detector for our own code, so Ollama's own footprint would only
    add noise.
    """
    root = root or psutil.Process()
    procs = [root]
    try:
        procs += root.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    ollama_pids = {p.pid for p in ollama_processes()} - {root.pid}
    total = 0.0
    seen: set[int] = set()
    for p in procs:
        if p.pid in seen or p.pid in ollama_pids:
            continue
        seen.add(p.pid)
        try:
            total += p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total / MB


def memory_pressure_level() -> int:
    """macOS kern.memorystatus_vm_pressure_level: 1 normal, 2 warn, 4 critical."""
    import subprocess

    try:
        out = subprocess.run(
            ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
            capture_output=True, text=True, timeout=5,
        )
        return int(out.stdout.strip())
    except Exception:
        return PRESSURE_NORMAL


@dataclass
class ExtractionGuard:
    """Abort conditions for a long extraction run (D10).

    (a) own process-tree RSS (excluding Ollama) stays above baseline x
        rss_growth_factor for rss_window consecutive ticks -- a real leak, not a
        spike;
    (b) rolling seconds/paragraph over the last tp_window_paragraphs exceeds
        baseline x tp_factor -- the machine is actually slowing us down;
    (c) OS memory pressure reads critical on pressure_consecutive ticks in a row.

    System-wide swap and available RAM are logged every warn_every_s as a
    warning only.
    """

    rss_growth_factor: float = 1.5
    rss_growth_min_mb: float = 300.0
    rss_window: int = 10
    tp_factor: float = 1.5
    tp_window_paragraphs: int = 50
    pressure_consecutive: int = 3
    warn_every_s: float = 60.0
    baseline_after_paragraphs: int = 25
    baseline_until_paragraphs: int = 75

    history: list[dict] = field(default_factory=list)
    baseline_rss_mb: float | None = None
    baseline_t_p: float | None = None
    _critical_streak: int = 0
    _last_warn_s: float = -1e9
    warnings: list[dict] = field(default_factory=list)

    def observe(self, extracted: int, cum_seconds: float, elapsed_s: float) -> dict:
        rss = own_tree_rss_mb()
        pressure = memory_pressure_level()
        rec = {
            "extracted": extracted, "cum_seconds": cum_seconds, "elapsed_s": elapsed_s,
            "own_rss_mb": rss, "pressure": pressure,
            "swap_mb": swap_used_mb(),
            "available_mb": psutil.virtual_memory().available / MB,
        }
        self.history.append(rec)

        self._critical_streak = (
            self._critical_streak + 1 if pressure >= PRESSURE_CRITICAL else 0
        )

        # baselines, taken once past warm-up
        window = [
            h for h in self.history
            if self.baseline_after_paragraphs <= h["extracted"] <= self.baseline_until_paragraphs
        ]
        if self.baseline_rss_mb is None and window and extracted >= self.baseline_until_paragraphs:
            self.baseline_rss_mb = statistics.median(h["own_rss_mb"] for h in window)
            first, last = window[0], window[-1]
            dn = last["extracted"] - first["extracted"]
            if dn > 0:
                self.baseline_t_p = (last["cum_seconds"] - first["cum_seconds"]) / dn
        return rec

    def rolling_t_p(self) -> float | None:
        """Seconds per paragraph over the most recent tp_window_paragraphs."""
        if len(self.history) < 2:
            return None
        last = self.history[-1]
        for h in reversed(self.history[:-1]):
            if last["extracted"] - h["extracted"] >= self.tp_window_paragraphs:
                dn = last["extracted"] - h["extracted"]
                return (last["cum_seconds"] - h["cum_seconds"]) / dn if dn else None
        return None

    def should_warn(self, elapsed_s: float) -> bool:
        if elapsed_s - self._last_warn_s >= self.warn_every_s:
            self._last_warn_s = elapsed_s
            return True
        return False

    def check(self) -> str | None:
        """Returns an abort reason, or None to continue."""
        if self._critical_streak >= self.pressure_consecutive:
            return (
                f"OS memory pressure critical on {self._critical_streak} consecutive "
                f"readings (kern.memorystatus_vm_pressure_level=4)"
            )
        if self.baseline_rss_mb is not None and len(self.history) >= self.rss_window:
            recent = [h["own_rss_mb"] for h in self.history[-self.rss_window:]]
            floor = min(recent)
            limit = max(
                self.baseline_rss_mb * self.rss_growth_factor,
                self.baseline_rss_mb + self.rss_growth_min_mb,
            )
            if floor > limit:
                return (
                    f"own process-tree RSS grew steadily: every one of the last "
                    f"{self.rss_window} readings above {limit:.0f} MB "
                    f"(baseline {self.baseline_rss_mb:.0f} MB, current floor {floor:.0f} MB)"
                )
        if self.baseline_t_p:
            roll = self.rolling_t_p()
            if roll and roll > self.baseline_t_p * self.tp_factor:
                return (
                    f"throughput degraded: rolling t_p over the last "
                    f"{self.tp_window_paragraphs} paragraphs is {roll:.2f} s vs baseline "
                    f"{self.baseline_t_p:.2f} s (> {self.tp_factor}x)"
                )
        return None
