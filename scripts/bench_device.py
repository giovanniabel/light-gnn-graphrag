"""M0 device benchmark: mps vs cpu on an EdgeRouter-shaped training epoch.

agent.md Section 6.4: tiny graphs may run faster on CPU than on mps, so benchmark
one training epoch on both and set DEVICE in config accordingly. If mps raises an
unsupported-op error even with fallback enabled, use CPU and record it.

The real EdgeRouter arrives in M6, so this benchmark uses a synthetic stand-in with
the Section 4.3 shape (input linear -> L x GATv2Conv(edge_dim) + residual + LayerNorm
-> edge MLP head) over random graphs at the Section 4.1 caps (200 nodes / 400 edges,
batch 32). Re-checked against the real model in M6.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT, load_config, set_config_value, set_seeds  # noqa: E402  (sets MPS fallback)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch_geometric.data import Data  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402
from torch_geometric.nn import GATv2Conv  # noqa: E402

# Taken from the real feature layout rather than restated, so the benchmark cannot
# drift from the model it is meant to represent (Section 4.3).
from src.graph_builder import EDGE_FEAT_DIM as EDGE_DIM  # noqa: E402
from src.graph_builder import NODE_FEAT_DIM as NODE_DIM  # noqa: E402

Q_DIM = 384


class _BenchRouter(nn.Module):
    """Shape-faithful stand-in for EdgeRouter (Section 4.3)."""

    def __init__(self, hidden=128, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(NODE_DIM, hidden)
        self.convs = nn.ModuleList(
            GATv2Conv(hidden, hidden // heads, heads=heads, edge_dim=EDGE_DIM)
            for _ in range(layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.drop = nn.Dropout(dropout)
        self.edge_proj = nn.Linear(EDGE_DIM, hidden)
        self.q_proj = nn.Linear(Q_DIM, hidden)
        self.head = nn.Sequential(
            nn.Linear(4 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, data):
        h = self.inp(data.x)
        for conv, norm in zip(self.convs, self.norms):
            h = self.drop(torch.relu(norm(h + conv(h, data.edge_index, data.edge_attr))))
        src, dst = data.edge_index
        q = data.q[data.batch[src]] if hasattr(data, "batch") and data.batch is not None else data.q.expand(src.size(0), -1)
        z = torch.cat([h[src], h[dst], self.edge_proj(data.edge_attr), self.q_proj(q)], dim=-1)
        return self.head(z).squeeze(-1)


def _synthetic_graphs(n_graphs=32, nodes=70, edges=415, gen=None):
    out = []
    for _ in range(n_graphs):
        out.append(
            Data(
                x=torch.randn(nodes, NODE_DIM, generator=gen),
                edge_index=torch.randint(0, nodes, (2, edges), generator=gen),
                edge_attr=torch.randn(edges, EDGE_DIM, generator=gen),
                y_edge=torch.randint(0, 2, (edges,), generator=gen).float(),
                q=torch.randn(1, Q_DIM, generator=gen),
            )
        )
    return out


def bench(device: str, n_graphs=96, batch=32, repeats=2) -> dict:
    """One training epoch (forward + backward + step) over n_graphs."""
    gen = torch.Generator().manual_seed(1234)
    graphs = _synthetic_graphs(n_graphs, gen=gen)
    loader = DataLoader(graphs, batch_size=batch, shuffle=False)

    model = _BenchRouter().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    lossf = nn.BCEWithLogitsLoss()

    # warm-up epoch (kernel compilation / lazy init), not timed
    times = []
    for r in range(repeats + 1):
        t0 = time.perf_counter()
        for b in loader:
            b = b.to(device)
            opt.zero_grad()
            loss = lossf(model(b), b.y_edge)
            loss.backward()
            opt.step()
        if device == "mps":
            torch.mps.synchronize()
        dt = time.perf_counter() - t0
        if r > 0:
            times.append(dt)

    n_params = sum(p.numel() for p in model.parameters())
    return {
        "device": device,
        "epoch_s": min(times),
        "epoch_s_all": times,
        "n_graphs": n_graphs,
        "batch_graphs": batch,
        "n_params": n_params,
        "params_under_1m": n_params < 1_000_000,
    }


def main() -> int:
    set_seeds()
    cfg = load_config()
    results = {}

    print(f"torch {torch.__version__} | mps available: {torch.backends.mps.is_available()}")
    print(f"PYTORCH_ENABLE_MPS_FALLBACK={__import__('os').environ.get('PYTORCH_ENABLE_MPS_FALLBACK')}")

    r = bench("cpu")
    results["cpu"] = r
    print(f"cpu : {r['epoch_s']:.3f} s/epoch  ({r['n_params']:,} params)")

    if torch.backends.mps.is_available():
        try:
            r = bench("mps")
            results["mps"] = r
            print(f"mps : {r['epoch_s']:.3f} s/epoch")
        except Exception as exc:
            results["mps"] = {"device": "mps", "error": repr(exc), "traceback": traceback.format_exc()}
            print(f"mps : FAILED (unsupported op even with fallback) -> {exc!r}")
            print("      recording failure and using cpu (Section 6.4)")
    else:
        results["mps"] = {"device": "mps", "error": "mps not available"}
        print("mps : not available")

    ok = [v for v in results.values() if "epoch_s" in v]
    chosen = min(ok, key=lambda v: v["epoch_s"])["device"]
    results["chosen"] = chosen
    speedup = None
    if all("epoch_s" in results.get(d, {}) for d in ("cpu", "mps")):
        speedup = results["cpu"]["epoch_s"] / results["mps"]["epoch_s"]
        results["mps_speedup_over_cpu"] = speedup
        print(f"mps/cpu speedup: {speedup:.2f}x")

    set_config_value("device", chosen)   # surgical: keeps the config's comments
    print(f"\nDEVICE -> {chosen}  (written to configs/default.yaml)")

    out = ROOT / "results" / "device_benchmark.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"benchmark -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
