"""Stratified k-fold cross-validation and graph masking (deviation D17).

Why CV: a 10% dev carve of 150 training questions leaves 15 questions, on which
AllGold@4 has 16 possible values and AP is noisy enough that stage A's best epoch
landed at 1-3. Ablation effects are smaller than that, so config selection and
ablations run on out-of-fold scores instead.

Masking (not rebuilding) is how the H and edge-type ablations are run: rebuilding
would change the node feature width and re-run merging and seeding, so the
ablation would differ from the full model in its inputs as well as the thing being
ablated. See the caveat on `mask_graph`.
"""

from __future__ import annotations

import random
from collections import defaultdict

import torch

from src.graph_builder import EDGE_KINDS, KIND_TRIPLE, NODE_KINDS

# x = [ev(384) | ev*q(384) | cos(1) | is_seed(1) | onehot(hop 0..H)(4) | logdeg(1) | kind(3)]
HOP_ONEHOT_SLICE = slice(384 * 2 + 2, 384 * 2 + 2 + 4)


def stratum_of(ex) -> str:
    """MuSiQue stratifies by hop count, HotpotQA by bridge/comparison (Section 3)."""
    if ex.dataset == "musique":
        return f"{ex.n_hops}hop"
    return str(ex.meta.get("hotpot_type") or "unknown")


def make_folds(examples: list, n_folds: int = 5, seed: int = 1234) -> list[dict]:
    """Stratified folds, fixed by seed and therefore identical for every config.

    Returns a list of {"test": [qid...], "train": [qid...]} in fold order.
    """
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for ex in examples:
        by_stratum[stratum_of(ex)].append(ex.qid)

    rng = random.Random(seed)
    assign: dict[str, int] = {}
    for stratum in sorted(by_stratum):
        qids = sorted(by_stratum[stratum])
        rng.shuffle(qids)
        for i, q in enumerate(qids):
            assign[q] = i % n_folds          # round-robin keeps folds balanced per stratum

    all_qids = [ex.qid for ex in examples]
    folds = []
    for f in range(n_folds):
        test = [q for q in all_qids if assign[q] == f]
        train = [q for q in all_qids if assign[q] != f]
        folds.append({"fold": f, "test": test, "train": train})
    return folds


def inner_holdout(train_qids: list[str], frac: float, seed: int) -> tuple[list[str], list[str]]:
    """Split a fold's training questions into (inner_train, inner_holdout).

    The holdout is used ONLY for early stopping. The fold's test questions never
    touch stopping, which is what keeps the OOF scores clean.
    """
    rng = random.Random(seed)
    qs = sorted(train_qids)
    rng.shuffle(qs)
    n = max(1, int(round(frac * len(qs))))
    return sorted(qs[n:]), sorted(qs[:n])


def fold_summary(folds: list[dict], examples: list) -> dict:
    strat = {ex.qid: stratum_of(ex) for ex in examples}
    out = {}
    for f in folds:
        c: dict[str, int] = defaultdict(int)
        for q in f["test"]:
            c[strat[q]] += 1
        out[f"fold{f['fold']}"] = {"n_test": len(f["test"]), "n_train": len(f["train"]),
                                   "strata": dict(sorted(c.items()))}
    return out


# ---------------------------------------------------------------- masking
def hop_dist_of(data) -> torch.Tensor:
    """Recover per-node hop distance from the one-hot block already in `x`."""
    return data.x[:, HOP_ONEHOT_SLICE].argmax(dim=1)


def mask_graph(data, drop_kinds: tuple[str, ...] = (), max_hop: int | None = None):
    """Return a copy of `data` with edges and/or far nodes removed.

    Feature WIDTHS are untouched (778 / 390), so an ablated graph is still
    consumable by the same model with no rebuild.

    Caveat, stated because it matters when reading the ablations: the node
    features were computed on the FULL graph, so `log1p(degree)` and the hop
    one-hot still describe the unmasked graph. The ablation therefore removes
    those edges from MESSAGE PASSING and from scoring, not from the node features.
    A clean removal would require rebuilding, which would change the feature width
    and confound the comparison -- D17 and §5 choose masking deliberately.
    """
    from torch_geometric.data import Data

    e_total = data.edge_index.size(1)
    keep_e = torch.ones(e_total, dtype=torch.bool)

    kinds = list(getattr(data, "edge_kind", []) or [KIND_TRIPLE] * e_total)
    if drop_kinds:
        for i, k in enumerate(kinds):
            if k in drop_kinds:
                keep_e[i] = False

    node_map = None
    n_total = data.x.size(0)
    if max_hop is not None:
        hd = hop_dist_of(data)
        keep_n = hd <= max_hop
        if int(keep_n.sum()) == 0:
            keep_n = torch.zeros(n_total, dtype=torch.bool)
            keep_n[data.seed_mask.bool()] = True
        src, dst = data.edge_index
        keep_e &= keep_n[src] & keep_n[dst]
        node_map = torch.full((n_total,), -1, dtype=torch.long)
        node_map[keep_n] = torch.arange(int(keep_n.sum()))
    else:
        keep_n = torch.ones(n_total, dtype=torch.bool)

    ei = data.edge_index[:, keep_e]
    if node_map is not None:
        ei = node_map[ei]

    ev = getattr(data, "evidence_mask", None)
    out = Data(
        x=data.x[keep_n],
        edge_index=ei,
        edge_attr=data.edge_attr[keep_e],
        y_edge=data.y_edge[keep_e],
        q=data.q,
        seed_mask=data.seed_mask[keep_n],
        evidence_mask=(ev[keep_e] if ev is not None else None),
        edge_pid=[p for p, k in zip(data.edge_pid, keep_e.tolist()) if k],
        edge_kind=[p for p, k in zip(kinds, keep_e.tolist()) if k],
        qid=data.qid, dataset=data.dataset, n_hops=data.n_hops,
    )
    return out


ABLATION_MASKS = {
    "full": {},
    "no_para": {"drop_kinds": ("para",)},
    "no_same_as": {"drop_kinds": ("same_as",)},
    "no_title": {"drop_kinds": ("title",)},
    "triples_only": {"drop_kinds": ("title", "same_as", "para")},
    "H2": {"max_hop": 2},
}
