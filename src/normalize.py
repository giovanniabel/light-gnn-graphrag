"""Entity-string normalization and answer matching (agent.md Section 4.1/4.2).

Node identity: "normalized entity strings (lowercase, strip punctuation/articles).
Merge exact matches." Kept in its own module because graph_builder.py (M5),
metrics.py (M7) and the extraction quality checks must all normalize identically.
"""

from __future__ import annotations

import re
import string
import unicodedata

_ARTICLES = ("the ", "a ", "an ")
_PUNCT = str.maketrans("", "", string.punctuation)
_WS = re.compile(r"\s+")


def normalize_entity(s: str) -> str:
    """Lowercase, strip accents, drop punctuation, drop leading articles, collapse space."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    s = s.translate(_PUNCT)
    s = _WS.sub(" ", s).strip()
    for art in _ARTICLES:
        if s.startswith(art):
            s = s[len(art):]
            break
    return s.strip()


def normalize_answer(s: str) -> str:
    """SQuAD-style answer normalization, shared with EM/F1 in metrics.py (Section 7.2)."""
    return normalize_entity(s)


def node_set(triples) -> set[str]:
    """Normalized node strings implied by a triple list (heads and tails)."""
    out: set[str] = set()
    for t in triples:
        for name in (t.head, t.tail):
            n = normalize_entity(name)
            if n:
                out.add(n)
    return out


def answer_in_nodes(answer: str, nodes: set[str], aliases=None) -> bool:
    """True iff the normalized answer exactly matches some node string.

    Exact normalized match only -- no substring matching, which would inflate the
    rate by counting any node that happens to contain the answer as a word.
    """
    cands = [answer] + list(aliases or [])
    return any(normalize_answer(c) in nodes for c in cands if c)
