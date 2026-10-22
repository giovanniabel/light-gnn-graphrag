"""Retrieval, answer and statistical metrics (agent.md Sections 7.1, 7.2, 7.4).

The one subtlety worth stating up front is AllGold@4's two forms (D4), which are
genuinely different quantities and were conflated once already:

  unconditional   P(gold ⊆ top-K) over ALL evaluated questions. Includes loss from
                  pool construction, so it is bounded by all-gold-in-pool.
  conditional     P(gold ⊆ top-K | gold ⊆ pool), which restricts the QUESTION SET
                  to those whose gold paragraphs all survived pooling. This is the
                  metric claim C1 is tested on.

Neither is `allgold_ceiling_conditional_on_pooling` from graph_stats, which
conditions by shrinking the gold set instead of the question set.
"""

from __future__ import annotations

import re
import statistics
import string
from collections import Counter

from src.schemas import CI


# ---------------------------------------------------------------- helpers
def _ranked(routed) -> list[str]:
    """Accept a Routed or a plain ranked pid list."""
    return list(getattr(routed, "paragraph_ids", routed))


def _gold(ex) -> set[str]:
    return {p.pid for p in ex.paragraphs if p.is_gold}


def _pool(ex) -> set[str]:
    return {p.pid for p in ex.paragraphs}


# ---------------------------------------------------------------- Section 7.1
def sp_recall(routed, ex, k: int = 4) -> float:
    """Fraction of this question's gold paragraphs that are in the top-k."""
    gold = _gold(ex)
    if not gold:
        return 0.0
    top = set(_ranked(routed)[:k])
    return len(gold & top) / len(gold)


def all_gold(routed, ex, k: int = 4) -> float:
    """1.0 iff every gold paragraph is in the top-k (unconditional per question)."""
    gold = _gold(ex)
    if not gold:
        return 0.0
    return float(gold.issubset(set(_ranked(routed)[:k])))


def gold_in_pool(ex, orig_gold: set[str] | None = None) -> bool:
    """Did every ORIGINAL gold paragraph survive pool construction?

    `orig_gold` is the pre-pooling gold set. Without it the question's own
    (already pooled) gold set is used, which makes this trivially True -- so
    callers that care must pass it.
    """
    gold = set(orig_gold) if orig_gold is not None else _gold(ex)
    return gold.issubset(_pool(ex))


def all_gold_conditional(
    routed_by_qid: dict,
    examples: list,
    orig_gold: dict,
    k: int = 4,
) -> dict:
    """D4's conditional AllGold@4: P(gold ⊆ top-K | gold ⊆ pool).

    Restricts the QUESTION SET to questions whose every original gold paragraph is
    in the pool, then scores the unconditional metric on that subset. Returns the
    rate, both set sizes and the unconditional rate, so a report can never quote
    the conditional number without saying how many questions it covers.
    """
    kept, hits = [], 0
    uncond_hits = 0
    for ex in examples:
        r = routed_by_qid.get(ex.qid)
        if r is None:
            continue
        go = set(orig_gold.get(ex.qid, _gold(ex)))
        top = set(_ranked(r)[:k])
        uncond_hits += float(go.issubset(top))
        if go and go.issubset(_pool(ex)):
            kept.append(ex.qid)
            hits += float(go.issubset(top))
    n_all = sum(1 for ex in examples if ex.qid in routed_by_qid)
    return {
        "conditional": (hits / len(kept)) if kept else None,
        "n_conditional": len(kept),
        "n_all": n_all,
        "conditioning_rate": (len(kept) / n_all) if n_all else None,
        "unconditional": (uncond_hits / n_all) if n_all else None,
        "k": k,
    }


def edge_precision(routed, qg, b: int = 10) -> float:
    """Fraction of the top-b scored EDGES that come from gold paragraphs.

    Only extracted (kind == triple) edges are eligible: Section 4.1.1 scaffolding
    carries no provenance claim, so counting it here would be meaningless.
    """
    from src.graph_builder import KIND_TRIPLE

    scored = getattr(routed, "edges", None)
    if not scored:
        return 0.0
    gold = set(qg.gold_pids)
    kinds = qg.edge_kind or [KIND_TRIPLE] * len(qg.edge_pid)
    elig = [
        (i, s) for i, s in scored
        if i < len(kinds) and kinds[i] == KIND_TRIPLE
    ]
    if not elig:
        return 0.0
    top = sorted(elig, key=lambda t: -t[1])[:b]
    return sum(1 for i, _s in top if qg.edge_pid[i] in gold) / len(top)


def path_recall(routed, qg, b: int = 10, use_aliases: bool = True) -> float:
    """1.0 iff the top-b scored edges contain a connected path from a seed to a
    node matching the gold answer (Section 7.1). Reported next to the graph
    ceiling, which bounds it."""
    from collections import deque

    scored = getattr(routed, "edges", None)
    if not scored or not qg.seeds:
        return 0.0
    targets = set(qg.answer_node_ids(use_aliases))
    if not targets:
        return 0.0
    top = [i for i, _s in sorted(scored, key=lambda t: -t[1])[:b]]
    adj: dict[int, list[int]] = {}
    for i in top:
        if i >= len(qg.edges):
            continue
        u, v = qg.edges[i]
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    seen = set(qg.seeds)
    dq = deque(qg.seeds)
    while dq:
        x = dq.popleft()
        if x in targets:
            return 1.0
        for y in adj.get(x, []):
            if y not in seen:
                seen.add(y)
                dq.append(y)
    return 0.0


# ---------------------------------------------------------------- Section 7.2
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def normalize_squad(s: str) -> str:
    """Standard SQuAD answer normalization (Section 7.2)."""
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def em_f1(pred: str, gold) -> tuple[float, float]:
    """Exact match and token F1 against one gold string or several aliases."""
    golds = [gold] if isinstance(gold, str) else list(gold)
    best_em, best_f1 = 0.0, 0.0
    p = normalize_squad(pred)
    pt = p.split()
    for g in golds:
        gn = normalize_squad(g)
        best_em = max(best_em, float(p == gn))
        gt = gn.split()
        if not pt or not gt:
            best_f1 = max(best_f1, float(pt == gt))
            continue
        common = Counter(pt) & Counter(gt)
        n_same = sum(common.values())
        if n_same == 0:
            continue
        prec, rec = n_same / len(pt), n_same / len(gt)
        best_f1 = max(best_f1, 2 * prec * rec / (prec + rec))
    return best_em, best_f1


# ---------------------------------------------------------------- Section 7.4
def paired_bootstrap(a: list[float], b: list[float], n: int = 10_000, seed: int = 1234) -> CI:
    """Paired bootstrap over questions on the DIFFERENCE a - b (Section 7.4).

    Resamples question indices, so the pairing between methods is preserved -- the
    whole point of the paired test.
    """
    import random

    if len(a) != len(b):
        raise ValueError(f"paired bootstrap needs equal lengths, got {len(a)} and {len(b)}")
    if not a:
        return CI(point=0.0, lo=0.0, hi=0.0, n_resamples=n)
    rng = random.Random(seed)
    m = len(a)
    point = statistics.fmean(a) - statistics.fmean(b)
    diffs = []
    for _ in range(n):
        idx = [rng.randrange(m) for _ in range(m)]
        diffs.append(
            statistics.fmean([a[i] for i in idx]) - statistics.fmean([b[i] for i in idx])
        )
    diffs.sort()
    lo = diffs[int(0.025 * (n - 1))]
    hi = diffs[int(0.975 * (n - 1))]
    return CI(point=point, lo=lo, hi=hi, n_resamples=n)


def average_precision(scores: list[float], labels: list[float]) -> float:
    """Edge-level average precision, used for M6 early stopping.

    AP rather than AllGold@4 because AllGold@4 on ~15 dev questions is a coarse
    step function (16 reachable values), which makes patience-based stopping noisy.
    """
    pairs = sorted(zip(scores, labels), key=lambda t: -t[0])
    n_pos = sum(1 for _s, y in pairs if y > 0)
    if n_pos == 0:
        return 0.0
    tp = 0
    total = 0.0
    for i, (_s, y) in enumerate(pairs, 1):
        if y > 0:
            tp += 1
            total += tp / i
    return total / n_pos
