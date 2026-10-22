"""Dataclasses from agent.md Section 10. Signatures here are stable; do not change
field names or order without flagging it as a deviation.

These live in one module (rather than inside data.py) because Triple is needed by
extractor.py and graph_builder.py, and StageStats by profiling.py, which data.py
must not import.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class GoldLeakError(AssertionError):
    """Raised when pool building touches a gold label (agent.md Section 3.1)."""


@dataclass
class Paragraph:
    pid: str
    title: str
    text: str
    is_gold: bool


@dataclass
class Example:
    qid: str
    dataset: str
    question: str
    answer: str
    paragraphs: list[Paragraph]
    n_hops: int | None
    # Additive trailing field (the Section 10 fields above are unchanged). Holds
    # per-dataset extras that must be reported separately, e.g. HotpotQA's
    # bridge/comparison type (Section 3). Survives dataclasses.replace().
    meta: dict = field(default_factory=dict)

    @property
    def gold_pids(self) -> set[str]:
        return {p.pid for p in self.paragraphs if p.is_gold}


@dataclass
class Triple:
    head: str
    rel: str
    tail: str
    pid: str


@dataclass
class StageStats:
    """Per-stage cost (agent.md Section 7.3). peak_rss_mb is process-tree + Ollama."""

    latency_s: float = 0.0
    llm_tokens: int = 0
    peak_rss_mb: float = 0.0
    delta_rss_mb: float = 0.0
    baseline_rss_mb: float = 0.0
    swap_delta_mb: float = 0.0
    stage: str = ""

    def as_dict(self) -> dict:
        return {
            "stage": self.stage,
            "latency_s": self.latency_s,
            "llm_tokens": self.llm_tokens,
            "peak_rss_mb": self.peak_rss_mb,
            "delta_rss_mb": self.delta_rss_mb,
            "baseline_rss_mb": self.baseline_rss_mb,
            "swap_delta_mb": self.swap_delta_mb,
        }


@dataclass
class Routed:
    paragraph_ids: list[str]          # ranked, length >= K
    edges: list[tuple[int, float]]    # (edge_idx, score), ranked; empty for Vector RAG
    stats: StageStats


@dataclass
class Usage:
    """Ollama token accounting: prompt_eval_count + eval_count (Section 7.3)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class ExtractionStats:
    paragraphs_total: int = 0
    paragraphs_extracted: int = 0
    paragraphs_cached: int = 0
    triples_total: int = 0
    json_failures: int = 0
    empty_results: int = 0
    seconds: float = 0.0
    llm_tokens: int = 0
    timed_out: bool = False
    stopped_reason: str | None = None

    @property
    def t_p(self) -> float:
        """Measured seconds per freshly extracted paragraph."""
        n = self.paragraphs_extracted
        return self.seconds / n if n else 0.0

    @property
    def json_failure_rate(self) -> float:
        n = self.paragraphs_extracted
        return self.json_failures / n if n else 0.0

    def as_dict(self) -> dict:
        d = {
            "paragraphs_total": self.paragraphs_total,
            "paragraphs_extracted": self.paragraphs_extracted,
            "paragraphs_cached": self.paragraphs_cached,
            "triples_total": self.triples_total,
            "json_failures": self.json_failures,
            "empty_results": self.empty_results,
            "seconds": self.seconds,
            "llm_tokens": self.llm_tokens,
            "timed_out": self.timed_out,
            "stopped_reason": self.stopped_reason,
            "t_p": self.t_p,
            "json_failure_rate": self.json_failure_rate,
        }
        return d


@dataclass
class CI:
    """Paired-bootstrap result (Section 7.4)."""

    point: float
    lo: float
    hi: float
    n_resamples: int = 10_000

    @property
    def excludes_zero(self) -> bool:
        return (self.lo > 0.0) or (self.hi < 0.0)

    def as_dict(self) -> dict:
        return {
            "point": self.point,
            "lo": self.lo,
            "hi": self.hi,
            "n_resamples": self.n_resamples,
            "excludes_zero": self.excludes_zero,
        }
