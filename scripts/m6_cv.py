"""M6 under D17 cross-validation: references, grid, ablations, learning curve, final.

Usage: python scripts/m6_cv.py <stage>
  refs       non-learned references (no training, no Ollama)
  grid       hyperparameter search on CV, pick by OOF edge AP
  compare    GNN vs references, RRF pairs, paired-bootstrap CIs
  ablations  Section 5 ablations as paired differences vs the full model
  curve      learning curve at 25/50/75/100% of each fold's training questions
  final      freeze the config, train on the FULL training set, 3 seeds, save

Validation is never loaded by any stage.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from evaluation.baselines import (  # noqa: E402
    edge_scores_for_ap,
    edge_sim,
    ppr_rag,
    rrf_pair,
    vector_rag,
)
from evaluation.cv import ABLATION_MASKS, fold_summary, make_folds  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    all_gold,
    all_gold_conditional,
    average_precision,
    paired_bootstrap,
    sp_recall,
)
from evaluation.oof import override_cfg, run_oof  # noqa: E402
from src.config import ROOT, env, load_config  # noqa: E402
from src.data import load_examples, pool_size_for  # noqa: E402
from src.embedder import EmbeddingCache, Embedder  # noqa: E402
from src.extractor import TripleCache, effective_prompt_version  # noqa: E402
from src.graph_builder import build_graph  # noqa: E402
from src.ollama_client import unload_all  # noqa: E402
from scripts.m1_data import load_pools  # noqa: E402
from scripts.m6_train import load_graphs  # noqa: E402

RES = ROOT / "results"
DATASETS = ("musique", "hotpotqa")


def assert_no_ollama() -> bool:
    unload_all()
    out = subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout
    rows = [l for l in out.strip().splitlines()[1:] if l.strip()]
    print(f"  `ollama ps` -> {len(rows)} model(s) loaded "
          f"{'(EMPTY, as Section 6.1 requires)' if not rows else '(NOT EMPTY: ' + str(rows) + ')'}")
    return not rows


def ctx(cfg, ds):
    exs = load_pools(ds, "train", pool_size_for(cfg, ds))
    graphs = load_graphs(cfg, ds, "train")
    folds = make_folds(exs, 5, int(cfg["seed"]))
    orig = {e.qid: e.gold_pids
            for e in load_examples(ds, "train", int(cfg["data"]["target_n"]), int(cfg["seed"]))}
    return exs, graphs, folds, orig


def query_graphs(cfg, ds, exs):
    """Rebuild QueryGraphs for the train split -- PPR and Edge-Sim need node names
    and relation phrases, which the serialized PyG Data does not carry."""
    g = cfg["graph"]
    tc = TripleCache(cfg.path("triples"), env("SLM_TAG"), effective_prompt_version(cfg))
    ec = EmbeddingCache(cfg.path("embeddings"), env("EMBED_MODEL"))
    warm = Embedder(env("EMBED_MODEL"), device=cfg.get("device") or "cpu", cache=ec)
    out = {}
    for ex in exs:
        tri = []
        for p in ex.paragraphs:
            tri.extend(tc.get(p.text) or [])
        out[ex.qid] = build_graph(
            ex, tri, warm, hops=int(g["hops"]), max_nodes=int(g["max_nodes"]),
            max_edges=int(g["max_edges"]),
            merge_near_duplicates=bool(g["merge_near_duplicates"]),
            merge_cosine=float(g["merge_cosine"]), max_seeds=int(g["max_seeds"]),
            seed_fallback_topk=int(g["seed_fallback_topk"]),
            connectivity=dict(g.get("connectivity") or {}),
        )
    return out, warm, ec


def metric_rows(name, per_q, allgold, edge_ap_pooled=None):
    return {
        "method": name,
        "edge_ap_mean": statistics.fmean([v["edge_ap"] for v in per_q.values()])
        if per_q and "edge_ap" in next(iter(per_q.values())) else None,
        "edge_ap_pooled": edge_ap_pooled,
        "sp_recall4": statistics.fmean([v["sp_recall4"] for v in per_q.values()]),
        "allgold4_uncond": allgold["unconditional"],
        "allgold4_cond": allgold["conditional"],
        "n_cond": allgold["n_conditional"],
        "n_all": allgold["n_all"],
    }


# ------------------------------------------------------------------ refs
def stage_refs(cfg):
    print("--- Stage: non-learned references (no training, no Ollama) ---")
    assert_no_ollama()
    k = int(cfg["routing"]["k"])
    out = {}
    for ds in DATASETS:
        exs, graphs, folds, orig = ctx(cfg, ds)
        qgs, emb, ec = query_graphs(cfg, ds, exs)
        print(f"\n  {ds}: {len(exs)} train questions")
        # pre-cache edge texts for Edge-Sim
        texts = set()
        for qg in qgs.values():
            for (u, v), rel in zip(qg.edges, qg.edge_rel):
                texts.add(f"{qg.nodes[u]} {rel} {qg.nodes[v]}")
        todo = [t for t in sorted(texts) if ec.get(t, "edges") is None]
        if todo:
            print(f"    caching {len(todo):,} edge texts for Edge-Sim ...", flush=True)
            for i in range(0, len(todo), 4096):
                emb.encode(todo[i:i + 4096], kind="edges")
            ec.flush()
        res = {}
        for name in ("vector_rag", "ppr", "edge_sim", "ppr_plus_vector"):
            per_q, routed = {}, {}
            for ex in exs:
                qg = qgs[ex.qid]
                if name == "vector_rag":
                    r = vector_rag(ex, emb, k=k)
                elif name == "ppr":
                    r = ppr_rag(qg, ex, k=k)
                elif name == "edge_sim":
                    r = edge_sim(qg, emb, ex, k=k)
                else:
                    r = rrf_pair(ppr_rag(qg, ex, k=k).paragraph_ids, ex, emb, k=k)
                routed[ex.qid] = r
                row = {"sp_recall4": sp_recall(r, ex, k=k), "allgold4": all_gold(r, ex, k=k)}
                if r.edges:
                    s, y = edge_scores_for_ap(qg, [0.0] * len(qg.edge_pid))
                    sc = [0.0] * len(qg.edge_pid)
                    for i, v in r.edges:
                        sc[i] = v
                    s, y = edge_scores_for_ap(qg, sc)
                    row["edge_ap"] = average_precision(s, y)
                per_q[ex.qid] = row
            ag = all_gold_conditional(routed, exs, orig, k=k)
            pooled = None
            if "edge_ap" in next(iter(per_q.values())):
                ss, yy = [], []
                for ex in exs:
                    qg = qgs[ex.qid]
                    sc = [0.0] * len(qg.edge_pid)
                    for i, v in routed[ex.qid].edges:
                        sc[i] = v
                    s, y = edge_scores_for_ap(qg, sc)
                    ss += s
                    yy += y
                pooled = average_precision(ss, yy)
            res[name] = {**metric_rows(name, per_q, ag, pooled), "per_question": per_q}
            r0 = res[name]
            ap = f"{r0['edge_ap_mean']:.4f}" if r0["edge_ap_mean"] is not None else "n/a"
            print(f"    {name:<16} edgeAP {ap:>7}  SP@4 {r0['sp_recall4']:.4f}  "
                  f"AllGold@4 {r0['allgold4_uncond']:.3f} (cond {r0['allgold4_cond']:.3f} "
                  f"on {r0['n_cond']}/{r0['n_all']})")
        out[ds] = res
        emb.free()
    (RES / "m6_refs.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n-> results/m6_refs.json")


# ------------------------------------------------------------------ grid
GRID = [
    {"lr": lr, "dropout": dr, "weight_decay": wd}
    for lr in (1e-3, 3e-4) for dr in (0.1, 0.3) for wd in (1e-2, 1e-1)
]


def stage_grid(cfg):
    print(f"--- Stage: hyperparameter grid, {len(GRID)} configs x 2 datasets ---")
    assert_no_ollama()
    print(f"  estimated {len(GRID) * 114 / 60:.1f} min at the measured 114 s/config\n")
    out = {"grid": [], "folds": {}}
    t0 = time.perf_counter()
    for i, hp in enumerate(GRID, 1):
        label = f"lr{hp['lr']:g}_dr{hp['dropout']}_wd{hp['weight_decay']:g}"
        c = override_cfg(cfg, model={"dropout": hp["dropout"]},
                         train={"lr": hp["lr"], "weight_decay": hp["weight_decay"]})
        row = {"label": label, **hp, "datasets": {}}
        for ds in DATASETS:
            exs, graphs, folds, orig = ctx(cfg, ds)
            out["folds"][ds] = fold_summary(folds, exs)
            r = run_oof(c, ds, graphs, exs, folds, [int(s) for s in cfg["seeds"]], orig,
                        label=label)
            row["datasets"][ds] = {
                "edge_ap_mean": r.mean("edge_ap"), "edge_ap_pooled": r.edge_ap_pooled,
                "sp_recall4": r.mean("sp_recall4"), "allgold": r.allgold,
                "fold_ap": r.fold_ap, "seed_ap": r.seed_ap, "wall_s": r.wall_s,
                "epochs_mean": statistics.fmean(r.epochs) if r.epochs else 0,
            }
        row["score"] = statistics.fmean(
            [row["datasets"][d]["edge_ap_mean"] for d in DATASETS])
        out["grid"].append(row)
        print(f"  [{i}/{len(GRID)}] {label:<28} "
              f"mus AP {row['datasets']['musique']['edge_ap_mean']:.4f}  "
              f"hot AP {row['datasets']['hotpotqa']['edge_ap_mean']:.4f}  "
              f"mean {row['score']:.4f}   ({time.perf_counter()-t0:.0f}s elapsed)", flush=True)
    best = max(out["grid"], key=lambda r: r["score"])
    out["best"] = {k: best[k] for k in ("label", "lr", "dropout", "weight_decay", "score")}
    print(f"\n  BEST by mean OOF per-question edge AP: {best['label']} "
          f"(score {best['score']:.4f})")
    (RES / "m6_grid.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"-> results/m6_grid.json")




# ------------------------------------------------------------------ compare
def _best_cfg(cfg):
    g = json.loads((RES / "m6_grid.json").read_text())
    b = g["best"]
    return b, override_cfg(cfg, model={"dropout": b["dropout"]},
                           train={"lr": b["lr"], "weight_decay": b["weight_decay"]})


def stage_compare(cfg):
    """GNN at the chosen config vs every reference, with paired-bootstrap CIs."""
    print("--- Stage: GNN vs references, paired-bootstrap CIs ---")
    assert_no_ollama()
    b, c = _best_cfg(cfg)
    refs = json.loads((RES / "m6_refs.json").read_text())
    k = int(cfg["routing"]["k"])
    out = {"config": b, "datasets": {}}

    for ds in DATASETS:
        exs, graphs, folds, orig = ctx(cfg, ds)
        qgs, emb, _ec = query_graphs(cfg, ds, exs)
        r = run_oof(c, ds, graphs, exs, folds, [int(s) for s in cfg["seeds"]], orig,
                    label="chosen")
        # GNN+Vector RRF on the OOF rankings
        gnn_rrf, per_q_rrf = {}, {}
        for ex in exs:
            if ex.qid not in r.rankings:
                continue
            fused = rrf_pair(r.rankings[ex.qid], ex, emb, k=k)
            gnn_rrf[ex.qid] = fused
            per_q_rrf[ex.qid] = {"sp_recall4": sp_recall(fused, ex, k=k),
                                 "allgold4": all_gold(fused, ex, k=k)}
        ag_rrf = all_gold_conditional(gnn_rrf, exs, orig, k=k)

        methods = {
            "gnn_graph_only": {
                "per_question": {q: {"sp_recall4": v["sp_recall4"], "allgold4": v["allgold4"],
                                     "edge_ap": v["edge_ap"]}
                                 for q, v in r.per_question.items()},
                **metric_rows("gnn_graph_only", r.per_question, r.allgold, r.edge_ap_pooled),
            },
            "gnn_plus_vector": {"per_question": per_q_rrf,
                                **metric_rows("gnn_plus_vector", per_q_rrf, ag_rrf, None)},
        }
        for name, row in refs[ds].items():
            methods[name] = row

        qids = sorted(set.intersection(*[set(m["per_question"]) for m in methods.values()]))
        print(f"\n  {ds}  (n={len(qids)} questions common to all methods)")
        print(f"    {'method':<18} {'edgeAP':>8} {'SP@4':>8} {'AllGold@4':>10} "
              f"{'cond':>8} {'n_cond':>8}")
        for name, m in methods.items():
            ap = m.get("edge_ap_mean")
            print(f"    {name:<18} {(f'{ap:.4f}' if ap is not None else 'n/a'):>8} "
                  f"{m['sp_recall4']:>8.4f} {m['allgold4_uncond']:>10.3f} "
                  f"{m['allgold4_cond']:>8.3f} {str(m['n_cond']) + '/' + str(m['n_all']):>8}")

        # paired bootstrap: each method vs Vector RAG, and the matched RRF pair
        pairs = [("gnn_graph_only", "vector_rag"), ("gnn_graph_only", "ppr"),
                 ("gnn_graph_only", "edge_sim"), ("gnn_plus_vector", "ppr_plus_vector"),
                 ("gnn_plus_vector", "vector_rag")]
        cis = {}
        print(f"\n    paired bootstrap, AllGold@4 difference (10,000 resamples)")
        for a, bn in pairs:
            va = [methods[a]["per_question"][q]["allgold4"] for q in qids]
            vb = [methods[bn]["per_question"][q]["allgold4"] for q in qids]
            ci = paired_bootstrap(va, vb, n=10_000)
            cis[f"allgold4|{a}-vs-{bn}"] = ci.as_dict()
            verdict = "excludes 0" if ci.excludes_zero else "INCLUDES 0 (within noise)"
            print(f"      {a:<16} - {bn:<16} {ci.point:+.4f}  "
                  f"[{ci.lo:+.4f}, {ci.hi:+.4f}]  {verdict}")

        # EDGE AP: the C2 comparison -- only the edge-scoring methods have it
        print(f"\n    paired bootstrap, EDGE AP difference (the C2 evidence)")
        for a, bn in (("gnn_graph_only", "ppr"), ("gnn_graph_only", "edge_sim")):
            if "edge_ap" not in methods[bn]["per_question"][qids[0]]:
                continue
            va = [methods[a]["per_question"][q]["edge_ap"] for q in qids]
            vb = [methods[bn]["per_question"][q]["edge_ap"] for q in qids]
            ci = paired_bootstrap(va, vb, n=10_000)
            cis[f"edge_ap|{a}-vs-{bn}"] = ci.as_dict()
            verdict = "excludes 0" if ci.excludes_zero else "INCLUDES 0 (within noise)"
            print(f"      {a:<16} - {bn:<16} {ci.point:+.4f}  "
                  f"[{ci.lo:+.4f}, {ci.hi:+.4f}]  {verdict}")
        out["datasets"][ds] = {
            "methods": {n: {kk: vv for kk, vv in m.items() if kk != "per_question"}
                        for n, m in methods.items()},
            "paired_bootstrap": cis,
            "fold_ap": r.fold_ap, "seed_ap": r.seed_ap,
            "fold_ap_sd": statistics.pstdev(r.fold_ap), "seed_ap_sd": statistics.pstdev(r.seed_ap),
        }
        print(f"    fold-to-fold AP sd {statistics.pstdev(r.fold_ap):.4f}  "
              f"seed-to-seed AP sd {statistics.pstdev(r.seed_ap):.4f}")
        emb.free()
    (RES / "m6_compare.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n-> results/m6_compare.json")


# ------------------------------------------------------------------ ablations
ABLATIONS = [
    ("full", {}, {}),
    ("L0", {"layers": 0}, {}),
    ("L1", {"layers": 1}, {}),
    ("L3", {"layers": 3}, {}),
    ("sage", {"aggregator": "sage"}, {}),
    ("no_query_cond", {"query_conditioning": False}, {}),
    ("H2", {}, {"max_hop": 2}),
    ("no_para", {}, {"drop_kinds": ("para",)}),
    ("no_same_as", {}, {"drop_kinds": ("same_as",)}),
    ("no_title", {}, {"drop_kinds": ("title",)}),
    ("triples_only", {}, {"drop_kinds": ("title", "same_as", "para")}),
]


def stage_ablations(cfg):
    print(f"--- Stage: Section 5 ablations, {len(ABLATIONS)} configs x 2 datasets ---")
    assert_no_ollama()
    print(f"  estimated {len(ABLATIONS) * 114 / 60:.1f} min at the measured 114 s/config\n")
    b, c = _best_cfg(cfg)
    print(f"  chosen config: {b['label']}\n")
    out = {"config": b, "datasets": {}}
    t0 = time.perf_counter()
    for ds in DATASETS:
        exs, graphs, folds, orig = ctx(cfg, ds)
        rows, vecs = {}, {}
        for label, mdl, mask in ABLATIONS:
            cc = override_cfg(cfg, model={"dropout": b["dropout"], **mdl},
                              train={"lr": b["lr"], "weight_decay": b["weight_decay"]})
            r = run_oof(cc, ds, graphs, exs, folds, [int(s) for s in cfg["seeds"]], orig,
                        label=label, mask=mask or None)
            rows[label] = {"edge_ap_mean": r.mean("edge_ap"), "sp_recall4": r.mean("sp_recall4"),
                           "allgold4_uncond": r.allgold["unconditional"],
                           "allgold4_cond": r.allgold["conditional"],
                           "n_cond": r.allgold["n_conditional"], "n_all": r.allgold["n_all"],
                           "fold_ap_sd": statistics.pstdev(r.fold_ap),
                           "seed_ap_sd": statistics.pstdev(r.seed_ap)}
            vecs[label] = r
            print(f"    {ds} {label:<14} edgeAP {r.mean('edge_ap'):.4f}  "
                  f"AllGold@4 {r.allgold['unconditional']:.3f}   "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
        qids = sorted(vecs["full"].per_question)
        cis = {}
        for label in rows:
            if label == "full":
                continue
            for metric in ("edge_ap", "allgold4"):
                a = [vecs[label].per_question[q][metric] for q in qids]
                f = [vecs["full"].per_question[q][metric] for q in qids]
                ci = paired_bootstrap(a, f, n=10_000)
                cis[f"{label}|{metric}"] = ci.as_dict()
        out["datasets"][ds] = {"rows": rows, "paired_vs_full": cis}
        print(f"\n  {ds}: paired difference vs FULL (negative = ablation is worse)")
        print(f"    {'ablation':<14} {'d edgeAP':>10} {'95% CI':>22} {'verdict':<24} "
              f"{'d AllGold@4':>12} {'95% CI':>22} {'verdict'}")
        for label in rows:
            if label == "full":
                continue
            e = cis[f"{label}|edge_ap"]
            g = cis[f"{label}|allgold4"]
            ve = "significant" if e["excludes_zero"] else "within noise"
            vg = "significant" if g["excludes_zero"] else "within noise"
            print(f"    {label:<14} {e['point']:>+10.4f} "
                  f"[{e['lo']:+.4f}, {e['hi']:+.4f}]".rjust(23)
                  + f" {ve:<24} {g['point']:>+12.4f} "
                  + f"[{g['lo']:+.4f}, {g['hi']:+.4f}]".rjust(23) + f" {vg}")
    (RES / "m6_ablations.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n-> results/m6_ablations.json")


# ------------------------------------------------------------------ curve
def stage_curve(cfg):
    print("--- Stage: learning curve at 25/50/75/100% of each fold's training set ---")
    assert_no_ollama()
    b, c = _best_cfg(cfg)
    sizes = (0.25, 0.50, 0.75, 1.00)
    print(f"  estimated {len(sizes) * 114 / 60:.1f} min\n")
    out = {"config": b, "datasets": {}}
    for ds in DATASETS:
        exs, graphs, folds, orig = ctx(cfg, ds)
        rows = {}
        for frac in sizes:
            r = run_oof(c, ds, graphs, exs, folds, [int(s) for s in cfg["seeds"]], orig,
                        label=f"size{int(frac*100)}", size_frac=frac)
            rows[f"{int(frac*100)}%"] = {
                "edge_ap_mean": r.mean("edge_ap"), "sp_recall4": r.mean("sp_recall4"),
                "allgold4_uncond": r.allgold["unconditional"],
                "allgold4_cond": r.allgold["conditional"],
                "n_train_inner": max(8, int(round(frac * (len(folds[0]["train"]) * 0.9)))),
            }
            print(f"    {ds} {int(frac*100):>3}%  edgeAP {r.mean('edge_ap'):.4f}  "
                  f"AllGold@4 {r.allgold['unconditional']:.3f}", flush=True)
        out["datasets"][ds] = rows
    (RES / "m6_curve.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n-> results/m6_curve.json")


# ------------------------------------------------------------------ final
def stage_final(cfg):
    """Train on the FULL training set (the former dev questions included), 3 seeds."""
    print("--- Stage: final models on the full training set ---")
    assert_no_ollama()
    b, c = _best_cfg(cfg)
    from evaluation.cv import inner_holdout
    from src.gnn_router import evidence_mask, train_router

    out = {"config": b, "datasets": {}}
    for ds in DATASETS:
        exs, graphs, folds, orig = ctx(cfg, ds)
        all_q = [e.qid for e in exs]
        runs = []
        for seed in [int(s) for s in cfg["seeds"]]:
            inner_tr, inner_ho = inner_holdout(all_q, float(cfg["data"]["dev_frac"]), seed)
            tr = [graphs[q] for q in inner_tr
                  if q in graphs and float(graphs[q].y_edge[evidence_mask(graphs[q])].sum()) > 0]
            ho = [graphs[q] for q in inner_ho if q in graphs]
            ck = train_router(tr, ho, c, seed, dataset=f"{ds}_final",
                              out_dir=RES / "checkpoints_final", log=lambda *a: None)
            h = json.loads((ck.parent / f"{ck.stem}_history.json").read_text())
            runs.append({"seed": seed, "checkpoint": str(ck.relative_to(ROOT)),
                         "n_train": len(tr), "n_holdout": len(ho),
                         "epochs": h["epochs_run"], "best_epoch": h["best_epoch"],
                         "holdout_ap": h["best_dev_ap"], "wall_s": h["wall_s"],
                         "pos_weight": h["pos_weight"]})
            print(f"    {ds} seed {seed}: {len(tr)} train + {len(ho)} stop-holdout, "
                  f"{h['epochs_run']} epochs, holdout AP {h['best_dev_ap']:.4f} "
                  f"-> {ck.name}")
        out["datasets"][ds] = {"n_questions": len(exs), "runs": runs}
    (RES / "m6_final.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n-> results/m6_final.json")
    print("  NOTE: validation was not evaluated. That is M7.")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "refs"
    cfg = load_config()
    {
        "refs": stage_refs, "grid": stage_grid, "compare": stage_compare,
        "ablations": stage_ablations, "curve": stage_curve, "final": stage_final,
    }[stage](cfg)
