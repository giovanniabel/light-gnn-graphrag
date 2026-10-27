"""M6 stage A: train the main EdgeRouter per dataset, 3 seeds (agent.md Section 11).

Stage A is the main model only -- L=2, GATv2Conv. No ablations.

Discipline enforced here:
  * trained and evaluated on TRAIN and DEV only; the validation split is never
    loaded, let alone scored;
  * loss over extracted (kind == triple) edges only, pos_weight from training
    triple edges;
  * early stopping on dev edge-level average precision, with dev AllGold@4 logged
    per epoch for information;
  * Ollama must not hold a model (Section 6.1) -- asserted, not assumed.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src.config import ROOT, load_config  # noqa: E402
from src.data import carve_dev, load_examples, pool_size_for  # noqa: E402
from src.gnn_router import (  # noqa: E402
    EdgeRouter,
    compute_pos_weight,
    evidence_mask,
    route,
    train_router,
)
from src.ollama_client import unload_all  # noqa: E402
from src.routing import rank_paragraphs  # noqa: E402
from scripts.m1_data import load_pools  # noqa: E402


def load_graphs(cfg, ds: str, split: str) -> dict:
    if split.startswith("validation"):        # "validation_ext" included (D27)
        from src.validation_guard import require

        require(f"load_graphs({ds!r}, {split!r})")
    hops = int(cfg["graph"]["hops"])
    path = cfg.path("graphs") / f"{ds}_{split}_H{hops}.pt"
    datas = torch.load(path, map_location="cpu", weights_only=False)
    return {d.qid: d for d in datas}


def split_train_dev(cfg, ds: str):
    """Same dev carve as everywhere else: 10% of train, by the global seed."""
    ps = pool_size_for(cfg, ds)
    pooled = load_pools(ds, "train", ps)
    tr_ex, dev_ex = carve_dev(pooled, float(cfg["data"]["dev_frac"]), int(cfg["seed"]))
    return tr_ex, dev_ex


def allgold_on_dev(model, dev_ex, dev_graphs, cfg, orig_gold) -> dict:
    """Dev AllGold@4, unconditional and D4-conditional. Information only."""
    from evaluation.metrics import all_gold_conditional

    k = int(cfg["routing"]["k"])
    routed = {}
    for ex in dev_ex:
        d = dev_graphs.get(ex.qid)
        if d is None:
            continue
        routed[ex.qid] = route(
            model, d, k=k, pool_pids=[p.pid for p in ex.paragraphs],
            mode=str(cfg["routing"]["paragraph_score"]), top2=int(cfg["routing"]["top2"]),
        )
    return all_gold_conditional(routed, dev_ex, orig_gold, k=k)


def overfit_test(cfg, graphs: list, log=print) -> dict:
    """Section 11: overfit 8 graphs to near-zero loss, as a wiring check."""
    import torch.nn as nn

    log("\n--- overfit test: 8 graphs, 300 steps, dropout off ---")
    torch.manual_seed(int(cfg["seed"]))
    sub = graphs[:8]
    m_cfg = cfg["model"]
    model = EdgeRouter(
        hidden=int(m_cfg["hidden"]), heads=int(m_cfg["heads"]), layers=int(m_cfg["layers"]),
        dropout=0.0, aggregator=str(m_cfg["aggregator"]),
        query_conditioning=bool(m_cfg["query_conditioning"]),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    lossf = nn.BCEWithLogitsLoss()
    from torch_geometric.loader import DataLoader

    batch = next(iter(DataLoader(sub, batch_size=8, shuffle=False)))
    m = evidence_mask(batch)
    first = None
    model.train()
    for step in range(1, 301):
        opt.zero_grad()
        loss = lossf(model(batch)[m], batch.y_edge[m])
        loss.backward()
        opt.step()
        if first is None:
            first = float(loss.detach())
        if step % 60 == 0:
            log(f"    step {step:>3}  loss {float(loss.detach()):.6f}")
    final = float(loss.detach())
    model.eval()
    with torch.no_grad():
        logits = model(batch)[m]
        pred = (torch.sigmoid(logits) > 0.5).float()
        acc = float((pred == batch.y_edge[m]).float().mean())
    ok = final < 0.05
    log(f"    first loss {first:.4f} -> final {final:.6f}  (edge accuracy {acc:.4f})")
    log(f"    reaches near-zero loss: {ok}")
    return {"first_loss": first, "final_loss": final, "edge_accuracy": acc,
            "near_zero": ok, "n_graphs": len(sub), "steps": 300,
            "n_evidence_edges": int(m.sum())}


def run(cfg, args) -> int:
    print("--- M6 stage A: main model (L=2, GATv2Conv), 3 seeds per dataset ---")
    unloaded = unload_all()
    import subprocess

    ps_out = subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout
    rows = [l for l in ps_out.strip().splitlines()[1:] if l.strip()]
    print(f"  ollama unloaded : {unloaded or 'nothing was loaded'}")
    print(f"  `ollama ps`     : {len(rows)} model(s) loaded -> "
          f"{'EMPTY, as Section 6.1 requires' if not rows else 'NOT EMPTY: ' + str(rows)}")
    if rows:
        print("  refusing to train while Ollama holds a model (Section 6.1)")
        return 1

    device = cfg.get("device") or "cpu"
    seeds = [int(s) for s in cfg["seeds"]]
    datasets = [args.dataset] if args.dataset else list(cfg["data"]["datasets"])
    print(f"  device {device} | seeds {seeds} | L={cfg['model']['layers']} "
          f"| aggregator {cfg['model']['aggregator']}")
    print(f"  validation split: NOT LOADED (dev only)")

    report = {"stage": "A", "device": device, "seeds": seeds, "datasets": {}}

    for ds in datasets:
        tr_ex, dev_ex = split_train_dev(cfg, ds)
        g_tr = load_graphs(cfg, ds, "train")
        orig_gold = {
            e.qid: e.gold_pids
            for e in load_examples(ds, "train", int(cfg["data"]["target_n"]), int(cfg["seed"]))
        }

        # Section 4.2: drop training questions with no positive edge. Eval keeps them.
        train_g, dropped = [], []
        for ex in tr_ex:
            d = g_tr.get(ex.qid)
            if d is None:
                continue
            m = evidence_mask(d)
            if float(d.y_edge[m].sum()) == 0:
                dropped.append(ex.qid)
            else:
                train_g.append(d)
        dev_g_list = [g_tr[ex.qid] for ex in dev_ex if ex.qid in g_tr]
        dev_graphs = {d.qid: d for d in dev_g_list}

        pw = compute_pos_weight(train_g, evidence_only=True)
        pw_all = compute_pos_weight(train_g, evidence_only=False)
        print(f"\n{'='*94}\n{ds.upper()}\n{'='*94}")
        print(f"  train graphs {len(train_g)} (dropped {len(dropped)} with no positive "
              f"evidence edge), dev graphs {len(dev_g_list)}")
        print(f"  pos_weight (evidence edges) {pw:.3f}   [all edges would be {pw_all:.2f}]")

        if ds == datasets[0]:
            report["overfit_test"] = overfit_test(cfg, train_g)

        runs = []
        for seed in seeds:
            print(f"\n  -- {ds} seed {seed} --")
            ck = train_router(
                train_g, dev_g_list, cfg, seed, dataset=ds,
                out_dir=ROOT / "results" / "checkpoints",
                dev_allgold_fn=lambda mdl: (
                    allgold_on_dev(mdl, dev_ex, dev_graphs, cfg, orig_gold)["unconditional"]
                ),
            )
            hist = json.loads((ck.parent / f"{ck.stem}_history.json").read_text())
            from src.gnn_router import load_router

            best = load_router(ck, cfg, device)
            ag = allgold_on_dev(best, dev_ex, dev_graphs, cfg, orig_gold)
            runs.append({
                "seed": seed, "checkpoint": str(ck.relative_to(ROOT)),
                "wall_s": hist["wall_s"], "epochs_run": hist["epochs_run"],
                "best_epoch": hist["best_epoch"], "best_dev_ap": hist["best_dev_ap"],
                "stopped_early": hist["stopped_early"], "n_params": hist["n_params"],
                "pos_weight": hist["pos_weight"], "dev_allgold4": ag,
            })
            print(f"    -> {hist['epochs_run']} epochs in {hist['wall_s']:.1f}s, "
                  f"best epoch {hist['best_epoch']}, dev AP {hist['best_dev_ap']:.4f}, "
                  f"dev AllGold@4 {ag['unconditional']:.3f} (cond {ag['conditional']} "
                  f"on {ag['n_conditional']}/{ag['n_all']})")

        report["datasets"][ds] = {
            "n_train": len(train_g), "n_dropped": len(dropped), "dropped_qids": dropped,
            "n_dev": len(dev_g_list), "pos_weight_evidence": pw, "pos_weight_all": pw_all,
            "runs": runs,
        }

    print(f"\n{'='*94}\nSTAGE A SUMMARY\n{'='*94}")
    print(f"  {'dataset':<9} {'seed':>5} {'epochs':>7} {'best_ep':>8} {'wall_s':>8} "
          f"{'dev_AP':>8} {'dev_AllGold@4':>14} {'checkpoint':<44}")
    import statistics

    for ds, r in report["datasets"].items():
        for run_ in r["runs"]:
            print(f"  {ds:<9} {run_['seed']:>5} {run_['epochs_run']:>7} "
                  f"{run_['best_epoch']:>8} {run_['wall_s']:>8.1f} "
                  f"{run_['best_dev_ap']:>8.4f} {run_['dev_allgold4']['unconditional']:>14.3f} "
                  f"{run_['checkpoint']:<44}")
        aps = [x["best_dev_ap"] for x in r["runs"]]
        ags = [x["dev_allgold4"]["unconditional"] for x in r["runs"]]
        print(f"  {ds:<9} {'mean':>5} {'':>7} {'':>8} "
              f"{statistics.fmean(x['wall_s'] for x in r['runs']):>8.1f} "
              f"{statistics.fmean(aps):>8.4f} {statistics.fmean(ags):>14.3f}"
              f"   +- AP {statistics.pstdev(aps):.4f} / AllGold {statistics.pstdev(ags):.3f}")

    checks = {
        "overfit test reaches near-zero loss": report["overfit_test"]["near_zero"],
        "every run produced a checkpoint": all(
            (ROOT / x["checkpoint"]).exists()
            for r in report["datasets"].values() for x in r["runs"]
        ),
        "each run <= 5 min": all(
            x["wall_s"] <= 300 for r in report["datasets"].values() for x in r["runs"]
        ),
        "3 seeds per dataset": all(len(r["runs"]) == 3 for r in report["datasets"].values()),
        "model under 1M parameters": all(
            x["n_params"] < 1_000_000
            for r in report["datasets"].values() for x in r["runs"]
        ),
        "pos_weight from evidence edges (~4.4-4.5)": all(
            4.0 <= r["pos_weight_evidence"] <= 5.0 for r in report["datasets"].values()
        ),
    }
    print("\n=== M6 STAGE A ACCEPTANCE ===")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    report["acceptance"] = checks
    report["all_passed"] = all(checks.values())

    out = ROOT / "results" / "m6_stageA.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nreport -> {out.relative_to(ROOT)}")
    print(f"M6 stage A: {'PASS' if report['all_passed'] else 'FAIL'}")
    print("Ablations NOT run (stage B awaits approval).")
    return 0 if report["all_passed"] else 1
