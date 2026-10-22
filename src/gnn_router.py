"""EdgeRouter and its training loop (agent.md Sections 4.3, 4.5, 10).

Architecture, exactly as Section 4.3 specifies:
    input linear -> L x [GATv2Conv(hidden/heads, heads, edge_dim) + residual
                         + LayerNorm + ReLU + dropout] -> edge MLP head
Edge head is an MLP over [h_u ; h_v ; proj(edge_attr) ; proj(q)] -> one logit.
Under 1M parameters.

Two things that are easy to get wrong and are handled explicitly here:
  * the loss covers ONLY extracted (kind == triple) edges. Section 4.1.1
    scaffolding is all-negative by construction and 71% of the graph, so training
    on it would optimise a mostly synthetic class balance. The router still SCORES
    every edge -- message passing needs them -- it just is not supervised on them.
  * `q` is per graph, so in a batch each edge takes the `q` of the graph its
    source node belongs to, via `data.batch`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, SAGEConv

from .graph_builder import EDGE_FEAT_DIM, NODE_FEAT_DIM
from .schemas import Routed, StageStats

Q_DIM = 384


class EdgeRouter(nn.Module):
    def __init__(
        self,
        node_dim: int = NODE_FEAT_DIM,
        edge_dim: int = EDGE_FEAT_DIM,
        hidden: int = 128,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
        aggregator: str = "gatv2",
        query_conditioning: bool = True,
        q_dim: int = Q_DIM,
    ):
        super().__init__()
        self.layers = layers
        self.aggregator = aggregator
        self.query_conditioning = query_conditioning
        self.inp = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            if aggregator == "sage":
                self.convs.append(SAGEConv(hidden, hidden, aggr="mean"))
            else:
                self.convs.append(
                    GATv2Conv(hidden, hidden // heads, heads=heads, edge_dim=edge_dim)
                )
            self.norms.append(nn.LayerNorm(hidden))
        self.drop = nn.Dropout(dropout)
        self.edge_proj = nn.Linear(edge_dim, hidden)
        self.q_proj = nn.Linear(q_dim, hidden)
        self.head = nn.Sequential(
            nn.Linear(4 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _q_per_edge(self, data, src: torch.Tensor) -> torch.Tensor:
        q = data.q
        if q.dim() == 1:
            q = q.unsqueeze(0)
        batch = getattr(data, "batch", None)
        if batch is None:
            return q.expand(src.size(0), -1)
        return q[batch[src]]

    def forward(self, data) -> torch.Tensor:
        """Returns one logit per edge, shape [E] -- for EVERY edge, scaffolding
        included. Which edges are supervised or scored into paragraphs is decided
        by the caller, not here."""
        h = self.inp(data.x)
        for conv, norm in zip(self.convs, self.norms):
            if self.aggregator == "sage":
                m = conv(h, data.edge_index)
            else:
                m = conv(h, data.edge_index, data.edge_attr)
            h = self.drop(torch.relu(norm(h + m)))

        src, dst = data.edge_index
        q = self._q_per_edge(data, src)
        if not self.query_conditioning:
            q = torch.zeros_like(q)
        z = torch.cat([h[src], h[dst], self.edge_proj(data.edge_attr), self.q_proj(q)], dim=-1)
        return self.head(z).squeeze(-1)


# ---------------------------------------------------------------- training
@dataclass
class TrainResult:
    checkpoint: Path
    seed: int
    dataset: str
    epochs_run: int
    best_epoch: int
    best_dev_ap: float
    best_dev_allgold: float | None
    history: list[dict]
    wall_s: float
    n_params: int
    pos_weight: float
    stopped_early: bool


def evidence_mask(data) -> torch.Tensor:
    m = getattr(data, "evidence_mask", None)
    if m is None:
        return torch.ones(data.edge_index.size(1), dtype=torch.bool)
    return m.bool()


def compute_pos_weight(graphs, evidence_only: bool = True) -> float:
    """pos_weight = #neg/#pos over the TRAINING set (Section 4.2).

    With evidence_only (the default), the count covers extracted edges alone, which
    is what the loss and paragraph scoring both use.
    """
    pos = neg = 0
    for d in graphs:
        y = d.y_edge
        m = evidence_mask(d) if evidence_only else torch.ones_like(y, dtype=torch.bool)
        yy = y[m]
        pos += int((yy > 0).sum())
        neg += int((yy == 0).sum())
    return (neg / pos) if pos else 1.0


@torch.no_grad()
def dev_scores(model, graphs, device: str = "cpu") -> list[tuple[list[float], list[float], object]]:
    """Per-graph (evidence scores, evidence labels, graph) on the dev split."""
    model.eval()
    out = []
    for d in graphs:
        dd = d.to(device)
        logits = model(dd)
        m = evidence_mask(dd)
        out.append((logits[m].cpu().tolist(), dd.y_edge[m].cpu().tolist(), d))
    return out


def train_router(train, dev, cfg, seed: int, dataset: str = "", out_dir: Path | None = None,
                 dev_allgold_fn=None, log=print) -> Path:
    """Train one EdgeRouter and return its checkpoint path (Section 10 signature).

    Early stopping is on dev edge-level average precision with patience from the
    config. Dev AllGold@4 is computed each epoch and logged for information only --
    with ~15 dev questions it takes 16 distinct values, which is too coarse to stop
    on reliably.
    """
    from evaluation.metrics import average_precision

    t0 = time.perf_counter()
    torch.manual_seed(seed)
    m_cfg, t_cfg = cfg["model"], cfg["train"]
    device = cfg.get("device") or "cpu"
    out_dir = Path(out_dir or (Path(cfg.path("results")) / "checkpoints"))
    out_dir.mkdir(parents=True, exist_ok=True)

    model = EdgeRouter(
        hidden=int(m_cfg["hidden"]), heads=int(m_cfg["heads"]),
        layers=int(m_cfg["layers"]), dropout=float(m_cfg["dropout"]),
        aggregator=str(m_cfg["aggregator"]),
        query_conditioning=bool(m_cfg["query_conditioning"]),
    ).to(device)

    pw = compute_pos_weight(train, evidence_only=True)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device))
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(t_cfg["lr"]), weight_decay=float(t_cfg["weight_decay"])
    )

    from torch_geometric.loader import DataLoader

    loader = DataLoader(train, batch_size=int(t_cfg["batch_graphs"]), shuffle=True)

    best_ap, best_epoch, best_allgold = -1.0, -1, None
    patience = int(t_cfg["patience"])
    ckpt = out_dir / f"router_{dataset}_L{m_cfg['layers']}_seed{seed}.pt"
    history, bad = [], 0
    stopped_early = False

    for epoch in range(1, int(t_cfg["max_epochs"]) + 1):
        model.train()
        tot, nb = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            logits = model(batch)
            m = evidence_mask(batch)
            if int(m.sum()) == 0:
                continue
            loss = lossf(logits[m], batch.y_edge[m])
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        train_loss = tot / max(nb, 1)

        rows = dev_scores(model, dev, device)
        flat_s = [s for ss, _ll, _g in rows for s in ss]
        flat_y = [y for _ss, ll, _g in rows for y in ll]
        ap = average_precision(flat_s, flat_y)
        ag = dev_allgold_fn(model) if dev_allgold_fn else None

        history.append({"epoch": epoch, "train_loss": train_loss, "dev_ap": ap,
                        "dev_allgold4": ag})
        log(f"    epoch {epoch:>2}  loss {train_loss:.4f}  dev_AP {ap:.4f}"
            + (f"  dev_AllGold@4 {ag:.3f}" if ag is not None else ""))

        if ap > best_ap:
            best_ap, best_epoch, best_allgold, bad = ap, epoch, ag, 0
            torch.save({"state_dict": model.state_dict(), "seed": seed,
                        "dataset": dataset, "epoch": epoch, "dev_ap": ap,
                        "dev_allgold4": ag, "pos_weight": pw,
                        "config": {k: dict(cfg[k]) for k in ("model", "train")}}, ckpt)
        else:
            bad += 1
            if bad >= patience:
                log(f"    early stop: dev AP has not improved for {patience} epochs")
                stopped_early = True
                break

    res = TrainResult(
        checkpoint=ckpt, seed=seed, dataset=dataset, epochs_run=len(history),
        best_epoch=best_epoch, best_dev_ap=best_ap, best_dev_allgold=best_allgold,
        history=history, wall_s=time.perf_counter() - t0, n_params=model.n_params,
        pos_weight=pw, stopped_early=stopped_early,
    )
    (out_dir / f"{ckpt.stem}_history.json").write_text(
        json.dumps({**res.__dict__, "checkpoint": str(ckpt)}, indent=2, default=str)
    )
    return ckpt


def load_router(ckpt: Path, cfg, device: str = "cpu") -> EdgeRouter:
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    m_cfg = blob.get("config", {}).get("model", cfg["model"])
    model = EdgeRouter(
        hidden=int(m_cfg["hidden"]), heads=int(m_cfg["heads"]), layers=int(m_cfg["layers"]),
        dropout=float(m_cfg["dropout"]), aggregator=str(m_cfg["aggregator"]),
        query_conditioning=bool(m_cfg["query_conditioning"]),
    ).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def route(model, data, k: int = 4, pool_pids: list[str] | None = None,
          mode: str = "mean", top2: int = 2) -> Routed:
    """Score a graph and return its ranked paragraphs (Section 10 signature).

    Scores every edge; paragraph scoring then uses extracted edges only
    (Section 4.4 / 4.1.1), so scaffolding cannot vouch for a paragraph.
    """
    from .routing import rank_paragraphs

    model.eval()
    t0 = time.perf_counter()
    logits = model(data)
    scores = logits.cpu().tolist()

    class _View:
        edge_pid = list(getattr(data, "edge_pid", []))
        edge_kind = list(getattr(data, "edge_kind", []))

    pool = pool_pids if pool_pids is not None else list(dict.fromkeys(_View.edge_pid))
    ranked = rank_paragraphs(_View(), scores, pool, k=k, mode=mode, top2=top2)
    edges = sorted(enumerate(scores), key=lambda t: -t[1])
    return Routed(
        paragraph_ids=ranked,
        edges=edges,
        stats=StageStats(latency_s=time.perf_counter() - t0, stage="routing"),
    )
