"""Dataset loading, splits and pool building (agent.md Section 3).

Hard rules enforced here:
  * Train and eval come from different official splits; qids asserted disjoint.
  * Datasets are kept separate -- nothing in this module pools them.
  * Nothing is loaded whole: both readers stream and reservoir-sample (Section 12).
  * build_pool never reads is_gold. It is handed GoldBlind views that raise
    GoldLeakError on any access to a gold label (Section 3.1).

Field names below were read off the actual files before this code was written:
  MuSiQue  : id, question, answer, answer_aliases, answerable,
             paragraphs[{idx, title, paragraph_text, is_supporting}]; hop count
             from the id prefix ("2hop__", "3hop1__", "4hop2__", ...).
  HotpotQA : id, question, answer, type (bridge|comparison), level,
             supporting_facts{title[], sent_id[]}, context{title[], sentences[][]}.
"""

from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Iterator

import torch

from .schemas import Example, GoldLeakError, Paragraph

DATASETS = ("musique", "hotpotqa")


def pool_size_for(cfg, dataset: str) -> int:
    """Per-dataset pool size (deviation D1).

    HotpotQA ships exactly 10 candidates so 10 is the whole pool; MuSiQue has 20
    and uses 14 to clear the Section 3.1 recall gate. Falls back to the legacy
    single `pool_size` when `pool_sizes` is absent.
    """
    sizes = cfg["data"].get("pool_sizes")
    if sizes and dataset in sizes:
        return int(sizes[dataset])
    return int(cfg["data"]["pool_size"])


# ---------------------------------------------------------------- gold blinding
class GoldBlind:
    """Read-only paragraph view that hides the gold label.

    Any attempt to reach is_gold (however spelled) raises GoldLeakError, so a
    leak in pool building fails loudly in tests instead of silently inflating
    pool recall.
    """

    __slots__ = ("pid", "title", "text")
    _FORBIDDEN = {"is_gold", "gold", "is_supporting", "gold_pids", "supporting"}

    def __init__(self, p: Paragraph):
        object.__setattr__(self, "pid", p.pid)
        object.__setattr__(self, "title", p.title)
        object.__setattr__(self, "text", p.text)

    def __getattr__(self, name: str):
        if name in self._FORBIDDEN:
            raise GoldLeakError(
                f"pool building touched gold label {name!r} (agent.md Section 3.1)"
            )
        raise AttributeError(name)

    def __setattr__(self, name: str, value) -> None:
        raise AttributeError("GoldBlind is read-only")

    def __repr__(self) -> str:
        return f"GoldBlind(pid={self.pid!r}, title={self.title!r})"


def gold_blind(paragraphs: list[Paragraph]) -> list[GoldBlind]:
    return [GoldBlind(p) for p in paragraphs]


# ---------------------------------------------------------------- raw downloads
def ensure_raw(cfg) -> dict[str, Path]:
    """Download the official files into data/raw via the HF hub cache."""
    from huggingface_hub import hf_hub_download

    raw = Path(cfg.path("raw"))
    raw.mkdir(parents=True, exist_ok=True)
    d = cfg["data"]
    out: dict[str, Path] = {}

    for key, fname in (
        ("musique_train", d["musique_train_file"]),
        ("musique_eval", d["musique_eval_file"]),
    ):
        out[key] = Path(
            hf_hub_download(
                repo_id=d["musique_repo"], filename=fname,
                repo_type="dataset", local_dir=raw / "musique",
            )
        )

    cfgname = d["hotpotqa_config"]
    for key, fname in (
        ("hotpotqa_train_0", f"{cfgname}/train-00000-of-00002.parquet"),
        ("hotpotqa_train_1", f"{cfgname}/train-00001-of-00002.parquet"),
        ("hotpotqa_eval", f"{cfgname}/validation-00000-of-00001.parquet"),
    ):
        out[key] = Path(
            hf_hub_download(
                repo_id=d["hotpotqa_repo"], filename=fname,
                repo_type="dataset", local_dir=raw / "hotpotqa",
            )
        )
    return out


# ---------------------------------------------------------------- row streaming
def _musique_hops(qid: str) -> int | None:
    """'2hop__...' -> 2, '3hop1__...' -> 3, '4hop2__...' -> 4."""
    head = qid.split("__", 1)[0]
    if head.endswith(("hop", "hop1", "hop2", "hop3")) or "hop" in head:
        try:
            return int(head.split("hop")[0])
        except ValueError:
            return None
    return None


def _stream_musique(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _stream_hotpotqa(paths: list[Path]) -> Iterator[dict]:
    """Stream parquet row-group by row-group; never materializes a whole split."""
    import pyarrow.parquet as pq

    for p in paths:
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(batch_size=256):
            yield from batch.to_pylist()


# ---------------------------------------------------------------- row -> Example
def _musique_example(row: dict) -> Example:
    qid = row["id"]
    paragraphs = [
        Paragraph(
            pid=f"{qid}::p{p['idx']}",
            title=p["title"],
            text=p["paragraph_text"],
            is_gold=bool(p.get("is_supporting", False)),
        )
        for p in row["paragraphs"]
    ]
    return Example(
        qid=qid, dataset="musique", question=row["question"],
        answer=row["answer"], paragraphs=paragraphs, n_hops=_musique_hops(qid),
        meta={"answer_aliases": row.get("answer_aliases") or []},
    )


def _hotpotqa_example(row: dict) -> Example:
    qid = row["id"]
    ctx = row["context"]
    gold_titles = set(row["supporting_facts"]["title"])
    paragraphs = [
        Paragraph(
            pid=f"{qid}::p{i}",
            title=title,
            text="".join(sents).strip(),
            is_gold=title in gold_titles,
        )
        for i, (title, sents) in enumerate(zip(ctx["title"], ctx["sentences"]))
    ]
    return Example(
        qid=qid, dataset="hotpotqa", question=row["question"],
        answer=row["answer"], paragraphs=paragraphs, n_hops=2,  # 2-hop by construction
        # bridge | comparison -- reported separately, never pooled (Section 3)
        meta={"hotpot_type": row.get("type"), "level": row.get("level")},
    )


# ---------------------------------------------------------------- sampling
def _reservoir_by_stratum(
    rows: Iterator[tuple[str, dict]], n: int, seed: int
) -> list[dict]:
    """Single-pass stratified reservoir sample: n split evenly over the strata seen.

    Strata are discovered while streaming, so this holds at most n rows plus the
    stratum counters in memory (Section 12: never load a full dataset).
    """
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = {}
    seen: dict[str, int] = {}
    cap = max(n, 1)

    for stratum, row in rows:
        seen[stratum] = seen.get(stratum, 0) + 1
        b = buckets.setdefault(stratum, [])
        if len(b) < cap:
            b.append(row)
        else:
            j = rng.randrange(seen[stratum])
            if j < cap:
                b[j] = row

    strata = sorted(buckets)
    if not strata:
        return []
    # even allocation, remainder to the largest strata (by what we sampled)
    base, rem = divmod(n, len(strata))
    quota = {s: base for s in strata}
    for s in sorted(strata, key=lambda s: -len(buckets[s]))[:rem]:
        quota[s] += 1

    out: list[dict] = []
    shortfall = 0
    for s in strata:
        take = min(quota[s], len(buckets[s]))
        out.extend(buckets[s][:take])
        shortfall += quota[s] - take
    if shortfall:  # some stratum was too small: refill from the rest
        pool = [r for s in strata for r in buckets[s][quota[s]:]]
        rng.shuffle(pool)
        out.extend(pool[:shortfall])

    rng.shuffle(out)
    return out[:n]


def load_examples(dataset: str, split: str, n: int, seed: int, cfg=None) -> list[Example]:
    """Sample n examples from an official split (agent.md Section 10).

    split is "train" or "validation" (the official MuSiQue dev file is the
    validation split). MuSiQue is stratified by hop count (Section 3); HotpotQA
    is sampled uniformly so the natural bridge/comparison ratio is preserved.

    Candidates are restricted to questions carrying at least this dataset's pool
    size in paragraphs,
    because Section 3.1 defines the pool as "all of HotpotQA's 10" / the top-10 of
    MuSiQue's 20 and every method must rank within an equal-sized pool. Both
    datasets have a small short-context tail: HotpotQA validation has 60 rows of
    7405 (0.8%) with 2-9 paragraphs, MuSiQue train 21 of 19938 (0.1%). The filter
    is applied identically to train and eval and uses only the paragraph count,
    never a gold label.
    """
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {DATASETS}")
    if split not in ("train", "validation"):
        raise ValueError(f"split must be 'train' or 'validation', got {split!r}")
    if split == "validation":
        from .validation_guard import require

        require(f"load_examples({dataset!r}, 'validation')")
    if cfg is None:
        from .config import load_config

        cfg = load_config()

    files = ensure_raw(cfg)
    pool_size = pool_size_for(cfg, dataset)

    if dataset == "musique":
        path = files["musique_train" if split == "train" else "musique_eval"]
        rows = (
            (f"{_musique_hops(r['id'])}hop", r)
            for r in _stream_musique(path)
            if r.get("answerable", True)            # answerable subset only (Section 3)
            and len(r["paragraphs"]) >= pool_size   # equal-sized pools (Section 3.1)
        )
        picked = _reservoir_by_stratum(rows, n, seed)
        return [_musique_example(r) for r in picked]

    paths = (
        [files["hotpotqa_train_0"], files["hotpotqa_train_1"]]
        if split == "train" else [files["hotpotqa_eval"]]
    )
    # D5: official validation is 100% level=hard, official train is a mix. Train is
    # restricted to hard so the two splits match in difficulty. Validation is left
    # exactly as the official split ships.
    want_level = cfg["data"].get("hotpotqa_train_level") if split == "train" else None
    rows = (
        ("all", r)
        for r in _stream_hotpotqa(paths)
        if len(r["context"]["title"]) >= pool_size   # equal-sized pools (Section 3.1)
        and (want_level is None or r.get("level") == want_level)
    )
    picked = _reservoir_by_stratum(rows, n, seed)
    return [_hotpotqa_example(r) for r in picked]


# ---------------------------------------------------------------- pool building
def _normalize_words(text: str, max_words: int) -> str:
    return " ".join(text.split()[:max_words])


def _select_pool_pids(
    question: str, blind: list[GoldBlind], embedder, pool_size: int
) -> list[str]:
    """Pick pool_size pids using only question + (pid, title, text).

    Receives GoldBlind views, so a gold-label read here raises GoldLeakError.
    HotpotQA already ships exactly 10, so this is a no-op there; MuSiQue's 20 are
    cut to the top-10 by cosine to the question (Section 3.1).
    """
    if len(blind) <= pool_size:
        return [b.pid for b in blind]

    q = embedder.encode_query(question)
    texts = [f"{b.title}. {b.text}" for b in blind]
    v = embedder.encode(texts, kind="paragraphs")
    sims = (v @ q.unsqueeze(-1)).squeeze(-1)
    order = torch.argsort(sims, descending=True).tolist()
    return [blind[i].pid for i in order[:pool_size]]


def build_pool(ex: Example, embedder, pool_size: int) -> Example:
    """Cut ex.paragraphs down to a pool of pool_size, without reading gold labels.

    Identical for train and eval, and shared by every method (Section 3.1).
    """
    keep = _select_pool_pids(ex.question, gold_blind(ex.paragraphs), embedder, pool_size)
    keep_set = set(keep)
    rank = {pid: i for i, pid in enumerate(keep)}
    # gold flags are re-attached only after selection, from the original objects
    pooled = sorted(
        (p for p in ex.paragraphs if p.pid in keep_set), key=lambda p: rank[p.pid]
    )
    return replace(ex, paragraphs=pooled)


# ---------------------------------------------------------------- splits & stats
def assert_disjoint(train: list[Example], evalset: list[Example]) -> set[str]:
    """agent.md Section 3: assert question ids are disjoint."""
    a = {e.qid for e in train}
    b = {e.qid for e in evalset}
    overlap = a & b
    assert not overlap, (
        f"train/eval qid overlap ({len(overlap)} ids), e.g. {sorted(overlap)[:5]}"
    )
    return overlap


def carve_dev(train: list[Example], dev_frac: float, seed: int) -> tuple[list[Example], list[Example]]:
    """Carve dev_frac of the training questions as a dev split (Section 3).

    Dev is used for early stopping and hyperparameters. Eval is never tuned on.
    """
    rng = random.Random(seed)
    idx = list(range(len(train)))
    rng.shuffle(idx)
    n_dev = max(1, int(round(dev_frac * len(train)))) if train else 0
    dev_idx = set(idx[:n_dev])
    dev = [train[i] for i in sorted(dev_idx)]
    tr = [train[i] for i in range(len(train)) if i not in dev_idx]
    return tr, dev


def pool_recall(pooled: list[Example], originals: list[Example]) -> dict:
    """Pool recall ceiling: fraction of gold paragraphs kept in the pool (Section 3.1)."""
    by_qid = {e.qid: e for e in originals}
    gold_total = gold_kept = 0
    q_all = q_full = 0
    for ex in pooled:
        orig = by_qid[ex.qid]
        g_orig = orig.gold_pids
        g_pool = ex.gold_pids
        gold_total += len(g_orig)
        gold_kept += len(g_pool)
        q_all += 1
        if g_orig and g_pool == g_orig:
            q_full += 1
    return {
        "n_questions": q_all,
        "gold_paragraphs_total": gold_total,
        "gold_paragraphs_in_pool": gold_kept,
        "pool_recall": gold_kept / gold_total if gold_total else 0.0,
        "questions_with_all_gold_in_pool": q_full,
        "all_gold_rate": q_full / q_all if q_all else 0.0,
    }


def hop_distribution(examples: list[Example]) -> dict:
    from collections import Counter

    c = Counter(e.n_hops for e in examples)
    return {str(k): v for k, v in sorted(c.items(), key=lambda kv: (kv[0] is None, kv[0]))}


def type_distribution(examples: list[Example]) -> dict:
    from collections import Counter

    c = Counter(e.meta.get("hotpot_type") for e in examples)
    return {str(k): v for k, v in sorted(c.items(), key=lambda kv: str(kv[0]))}


def level_distribution(examples: list[Example]) -> dict:
    """HotpotQA level mix. Official train is easy/medium/hard; official validation is
    all 'hard'. Reported so the train/eval difficulty mismatch stays visible."""
    from collections import Counter

    c = Counter(e.meta.get("level") for e in examples)
    return {str(k): v for k, v in sorted(c.items(), key=lambda kv: str(kv[0]))}
