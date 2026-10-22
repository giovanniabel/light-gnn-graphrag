"""Tests for EdgeRouter wiring and the evidence-only loss mask (Section 4.3/4.2)."""

from __future__ import annotations

import torch

from src.gnn_router import EdgeRouter, compute_pos_weight, evidence_mask
from src.graph_builder import EDGE_FEAT_DIM, NODE_FEAT_DIM


def fake_data(n=12, e=30, n_evidence=10, seed=0):
    from torch_geometric.data import Data

    g = torch.Generator().manual_seed(seed)
    kinds = ["triple"] * n_evidence + ["para"] * (e - n_evidence)
    y = torch.zeros(e)
    y[:n_evidence // 2] = 1.0            # half the evidence edges are positive
    return Data(
        x=torch.randn(n, NODE_FEAT_DIM, generator=g),
        edge_index=torch.randint(0, n, (2, e), generator=g),
        edge_attr=torch.randn(e, EDGE_FEAT_DIM, generator=g),
        y_edge=y,
        q=torch.randn(1, 384, generator=g),
        seed_mask=torch.zeros(n, dtype=torch.bool),
        evidence_mask=torch.tensor([k == "triple" for k in kinds]),
        edge_kind=kinds,
        edge_pid=[f"p{i % 4}" for i in range(e)],
        qid="q1", dataset="musique", n_hops=2,
    )


# ------------------------------------------------------------ shapes
def test_forward_returns_one_logit_per_edge():
    d = fake_data()
    out = EdgeRouter()(d)
    assert out.shape == (d.edge_index.size(1),)
    assert torch.isfinite(out).all()


def test_model_is_under_one_million_params():
    assert EdgeRouter().n_params < 1_000_000


def test_l0_has_no_conv_layers_but_same_input_width():
    """L=0 must consume identical features -- only depth differs (Section 5)."""
    d = fake_data()
    m0, m2 = EdgeRouter(layers=0), EdgeRouter(layers=2)
    assert len(m0.convs) == 0 and len(m2.convs) == 2
    assert m0.inp.in_features == m2.inp.in_features == NODE_FEAT_DIM
    assert m0.edge_proj.in_features == m2.edge_proj.in_features == EDGE_FEAT_DIM
    assert m0(d).shape == m2(d).shape


def test_sage_aggregator_runs():
    assert EdgeRouter(aggregator="sage", layers=2)(fake_data()).shape[0] == 30


def test_query_conditioning_off_changes_the_output():
    d = fake_data()
    torch.manual_seed(0)
    a = EdgeRouter(query_conditioning=True)
    torch.manual_seed(0)
    b = EdgeRouter(query_conditioning=False)
    assert not torch.allclose(a(d), b(d)), "disabling q must actually change scoring"


# ------------------------------------------------------------ batching
def test_batched_forward_matches_per_graph_forward():
    """Each edge must take the q of ITS OWN graph when batched."""
    from torch_geometric.loader import DataLoader

    ds = [fake_data(seed=i) for i in range(3)]
    torch.manual_seed(0)
    m = EdgeRouter()
    m.eval()
    with torch.no_grad():
        single = torch.cat([m(d) for d in ds])
        batched = m(next(iter(DataLoader(ds, batch_size=3, shuffle=False))))
    assert torch.allclose(single, batched, atol=1e-5)


# ------------------------------------------------------------ evidence mask
def test_evidence_mask_selects_only_triples():
    d = fake_data(e=30, n_evidence=10)
    assert int(evidence_mask(d).sum()) == 10


def test_evidence_mask_defaults_to_all_edges_when_absent():
    from torch_geometric.data import Data

    d = Data(edge_index=torch.zeros(2, 5, dtype=torch.long))
    assert int(evidence_mask(d).sum()) == 5


def test_pos_weight_evidence_only_differs_from_all_edges():
    """Scaffolding is all-negative, so including it inflates pos_weight."""
    d = fake_data(e=30, n_evidence=10)
    pw_ev = compute_pos_weight([d], evidence_only=True)
    pw_all = compute_pos_weight([d], evidence_only=False)
    assert pw_ev < pw_all
    assert pw_ev == (10 - 5) / 5          # 5 negative, 5 positive evidence edges
    assert pw_all == (30 - 5) / 5


def test_pos_weight_handles_no_positives():
    d = fake_data()
    d.y_edge = torch.zeros_like(d.y_edge)
    assert compute_pos_weight([d]) == 1.0


# ------------------------------------------------------------ training step
def test_loss_ignores_scaffolding_edges():
    """Gradients must not flow from scaffolding labels into the loss."""
    import torch.nn as nn

    d = fake_data()
    m = EdgeRouter()
    lossf = nn.BCEWithLogitsLoss()
    mask = evidence_mask(d)
    masked = lossf(m(d)[mask], d.y_edge[mask])
    # flipping a scaffolding label must leave the masked loss unchanged
    d2 = fake_data()
    d2.y_edge[-1] = 1.0
    torch.manual_seed(0)
    m2 = EdgeRouter().eval()        # eval(): dropout would make this comparison random
    torch.manual_seed(0)
    m3 = EdgeRouter().eval()
    with torch.no_grad():
        a = lossf(m2(d)[mask], d.y_edge[mask])
        b = lossf(m3(d2)[mask], d2.y_edge[mask])
    assert torch.allclose(a, b), "a scaffolding label changed the evidence-only loss"
    assert torch.isfinite(masked)
