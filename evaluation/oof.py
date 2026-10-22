"""Out-of-fold training and evaluation under D17's cross-validation.

For one config: 3 seeds x 5 folds per dataset. Each question is scored by a model
that never saw it in training OR in early stopping, so the OOF scores are clean.
Scores are averaged across seeds before metrics (§7.4), and per-question metric
vectors are kept so configs can be compared by paired bootstrap.
"""

from __future__ import annotations

import copy
import statistics
import time
from dataclasses import dataclass, field

import torch

from evaluation.cv import inner_holdout, mask_graph
from evaluation.metrics import all_gold, all_gold_conditional, average_precision, sp_recall
from src.gnn_router import evidence_mask, train_router
from src.routing import rank_paragraphs


def override_cfg(cfg, model: dict | None = None, train: dict | None = None):
    """A deep copy of cfg with model/train keys overridden, so the base config on
    disk is never mutated by a sweep."""
    c = copy.deepcopy({k: dict(v) if isinstance(v, dict) else v for k, v in cfg.items()})
    if model:
        c["model"].update(model)
    if train:
        c["train"].update(train)

    class _C(dict):
        def __getattr__(self, k):
            return self[k]

        def path(self, key):
            return cfg.path(key)

    return _C(c)


@dataclass
class OOFResult:
    label: str
    dataset: str
    per_question: dict = field(default_factory=dict)   # qid -> metric dict
    edge_ap_pooled: float = 0.0
    fold_ap: list = field(default_factory=list)
    seed_ap: list = field(default_factory=list)
    allgold: dict = field(default_factory=dict)
    wall_s: float = 0.0
    n_runs: int = 0
    epochs: list = field(default_factory=list)
    rankings: dict = field(default_factory=dict)      # qid -> ranked pids (for RRF)
    edge_scores: dict = field(default_factory=dict)   # qid -> seed-averaged edge scores

    def vec(self, metric: str) -> list[float]:
        return [self.per_question[q][metric] for q in sorted(self.per_question)]

    def mean(self, metric: str) -> float:
        v = self.vec(metric)
        return statistics.fmean(v) if v else 0.0


@torch.no_grad()
def _score(model, data) -> list[float]:
    model.eval()
    return model(data).cpu().tolist()


def run_oof(
    cfg,
    dataset: str,
    graphs: dict,
    examples: list,
    folds: list,
    seeds: list,
    orig_gold: dict,
    label: str = "full",
    mask: dict | None = None,
    size_frac: float = 1.0,
    log=lambda *a, **k: None,
) -> OOFResult:
    """Train 3 seeds x 5 folds and return seed-averaged OOF metrics."""
    from src.gnn_router import load_router

    t0 = time.perf_counter()
    k_top = int(cfg["routing"]["k"])
    ex_by_qid = {e.qid: e for e in examples}
    mask = mask or {}

    def prep(qid):
        d = graphs[qid]
        return mask_graph(d, **mask) if mask else d

    # qid -> list of per-seed score lists
    acc: dict[str, list[list[float]]] = {q: [] for q in graphs}
    epochs, seed_ap_rows, n_runs = [], [], 0

    for seed in seeds:
        seed_scores: dict[str, list[float]] = {}
        for f in folds:
            inner_tr, inner_ho = inner_holdout(f["train"], float(cfg["data"]["dev_frac"]), seed)
            if size_frac < 1.0:
                keep = max(8, int(round(size_frac * len(inner_tr))))
                rng = __import__("random").Random(seed + f["fold"])
                inner_tr = sorted(rng.sample(inner_tr, keep))

            tr = []
            for q in inner_tr:
                d = prep(q)
                m = evidence_mask(d)
                if float(d.y_edge[m].sum()) > 0:        # §4.2 drop from TRAINING only
                    tr.append(d)
            ho = [prep(q) for q in inner_ho if q in graphs]
            if not tr or not ho:
                continue

            ck = train_router(
                tr, ho, cfg, seed, dataset=f"{dataset}_{label}_f{f['fold']}",
                out_dir=cfg.path("results") / "cv_checkpoints", log=log,
            )
            n_runs += 1
            hist = __import__("json").loads(
                (ck.parent / f"{ck.stem}_history.json").read_text())
            epochs.append(hist["epochs_run"])
            model = load_router(ck, cfg, cfg.get("device") or "cpu")
            for q in f["test"]:
                if q in graphs:
                    seed_scores[q] = _score(model, prep(q))
            ck.unlink(missing_ok=True)
            (ck.parent / f"{ck.stem}_history.json").unlink(missing_ok=True)

        for q, sc in seed_scores.items():
            acc[q].append(sc)
        # this seed's own pooled edge AP, for seed-to-seed spread
        s_all, y_all = [], []
        for q, sc in seed_scores.items():
            d = prep(q)
            m = evidence_mask(d).tolist()
            s_all += [x for x, keep in zip(sc, m) if keep]
            y_all += [float(x) for x, keep in zip(d.y_edge.tolist(), m) if keep]
        seed_ap_rows.append(average_precision(s_all, y_all))

    # ---- seed-averaged scores -> metrics ----
    res = OOFResult(label=label, dataset=dataset)
    pooled_s, pooled_y = [], []
    routed = {}
    for q, runs in acc.items():
        if not runs:
            continue
        avg = [statistics.fmean(vals) for vals in zip(*runs)]
        d = prep(q)
        ex = ex_by_qid[q]
        m = evidence_mask(d).tolist()
        es = [x for x, keep in zip(avg, m) if keep]
        ey = [float(x) for x, keep in zip(d.y_edge.tolist(), m) if keep]
        pooled_s += es
        pooled_y += ey

        class _V:
            edge_pid = list(d.edge_pid)
            edge_kind = list(d.edge_kind)

        pool = [p.pid for p in ex.paragraphs]
        ranked = rank_paragraphs(_V(), avg, pool, k=k_top,
                                 mode=str(cfg["routing"]["paragraph_score"]),
                                 top2=int(cfg["routing"]["top2"]))
        routed[q] = ranked
        res.rankings[q] = ranked
        res.edge_scores[q] = avg
        res.per_question[q] = {
            "edge_ap": average_precision(es, ey),
            "sp_recall4": sp_recall(ranked, ex, k=k_top),
            "allgold4": all_gold(ranked, ex, k=k_top),
        }

    res.edge_ap_pooled = average_precision(pooled_s, pooled_y)
    res.seed_ap = seed_ap_rows
    res.fold_ap = [
        average_precision(
            [x for q in f["test"] if q in res.per_question for x in [res.per_question[q]["edge_ap"]]],
            [1.0] * sum(1 for q in f["test"] if q in res.per_question),
        ) for f in folds
    ]
    # fold-level mean of per-question AP is the honest fold spread
    res.fold_ap = [
        statistics.fmean([res.per_question[q]["edge_ap"] for q in f["test"] if q in res.per_question])
        for f in folds
    ]
    res.allgold = all_gold_conditional(routed, examples, orig_gold, k=k_top)
    res.wall_s = time.perf_counter() - t0
    res.n_runs = n_runs
    res.epochs = epochs
    return res
