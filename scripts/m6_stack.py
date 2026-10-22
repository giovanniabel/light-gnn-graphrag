"""D19: stacked hybrid -- logistic regression over paragraph rows.

Features per (question, pool paragraph): bge cosine to the question, the GNN's
OOF paragraph score, and the PPR paragraph score. Fit under a SECOND CV over
questions, with the same fixed stratified folds, so the combiner never scores a
question it was fit on. The GNN feature is already out-of-fold, so no label or
score from a question ever reaches its own prediction.

Train split only. The validation guard will refuse a validation read.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from evaluation.baselines import ppr_rag, vector_cosines  # noqa: E402
from evaluation.cv import make_folds  # noqa: E402
from evaluation.metrics import all_gold, all_gold_conditional, paired_bootstrap, sp_recall  # noqa: E402
from evaluation.oof import override_cfg, run_oof  # noqa: E402
from src.config import ROOT, load_config  # noqa: E402
from src.ollama_client import unload_all  # noqa: E402
from src.routing import paragraph_scores  # noqa: E402
from scripts.m6_cv import DATASETS, assert_no_ollama, ctx, query_graphs  # noqa: E402

FEATURES = ("vector_cos", "gnn_score", "ppr_score")


def _norm(d: dict, pool: list[str]) -> dict[str, float]:
    """Min-max normalize within a question; missing graph scores sit at the floor.

    Per-question normalization is what makes rows comparable across questions,
    and 0.0 for a paragraph the graph cannot score is the honest encoding: the
    graph has no opinion, which is itself the signal.
    """
    vals = [d[p] for p in pool if p in d]
    if not vals:
        return {p: 0.0 for p in pool}
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    return {p: ((d[p] - lo) / rng if p in d else 0.0) for p in pool}


def build_rows(cfg, ds, exs, qgs, emb, gnn_scores):
    """(qid -> {pid -> feature vector}, qid -> {pid -> label})."""
    X, Y, pools = {}, {}, {}
    for ex in exs:
        qg = qgs[ex.qid]
        pool = [p.pid for p in ex.paragraphs]
        pools[ex.qid] = pool
        vec = _norm(vector_cosines(ex, emb), pool)
        gnn = _norm(paragraph_scores(qg, gnn_scores[ex.qid],
                                     mode=str(cfg["routing"]["paragraph_score"]),
                                     top2=int(cfg["routing"]["top2"]))
                    if ex.qid in gnn_scores else {}, pool)
        pr = ppr_rag(qg, ex, k=int(cfg["routing"]["k"]))
        pr_sc = [0.0] * len(qg.edge_pid)
        for i, v in pr.edges:
            pr_sc[i] = v
        ppr = _norm(paragraph_scores(qg, pr_sc), pool)
        X[ex.qid] = {p: [vec[p], gnn[p], ppr[p]] for p in pool}
        Y[ex.qid] = {p.pid: float(p.is_gold) for p in ex.paragraphs}
    return X, Y, pools


def run(cfg) -> dict:
    from sklearn.linear_model import LogisticRegression

    print("--- D19: stacked hybrid (logistic regression over paragraph rows) ---")
    assert_no_ollama()
    k = int(cfg["routing"]["k"])
    refs = json.loads((RES := ROOT / "results") / "m6_refs.json" and
                      (ROOT / "results" / "m6_refs.json").read_text())
    grid = json.loads((ROOT / "results" / "m6_grid.json").read_text())
    b = grid["best"]
    c = override_cfg(cfg, model={"dropout": b["dropout"]},
                     train={"lr": b["lr"], "weight_decay": b["weight_decay"]})
    out = {"features": list(FEATURES), "config": b, "datasets": {}}

    for ds in DATASETS:
        exs, graphs, folds, orig = ctx(cfg, ds)
        qgs, emb, _ec = query_graphs(cfg, ds, exs)
        print(f"\n  {ds}: regenerating OOF GNN scores for the stack feature ...", flush=True)
        r = run_oof(c, ds, graphs, exs, folds, [int(s) for s in cfg["seeds"]], orig,
                    label="stack_feat")
        X, Y, pools = build_rows(cfg, ds, exs, qgs, emb, r.edge_scores)

        # nested CV: fit on 4/5 of the QUESTIONS, predict the held-out 1/5
        preds, coefs = {}, []
        for f in folds:
            tr_q = [q for q in f["train"] if q in X]
            te_q = [q for q in f["test"] if q in X]
            Xtr = np.array([X[q][p] for q in tr_q for p in pools[q]])
            ytr = np.array([Y[q][p] for q in tr_q for p in pools[q]])
            lr = LogisticRegression(max_iter=2000, class_weight="balanced")
            lr.fit(Xtr, ytr)
            coefs.append(lr.coef_[0].tolist())
            for q in te_q:
                sc = lr.predict_proba(np.array([X[q][p] for p in pools[q]]))[:, 1]
                order = sorted(range(len(pools[q])), key=lambda i: -sc[i])
                preds[q] = [pools[q][i] for i in order]

        per_q = {q: {"sp_recall4": sp_recall(preds[q], e, k=k),
                     "allgold4": all_gold(preds[q], e, k=k)}
                 for e in exs if (q := e.qid) in preds}
        ag = all_gold_conditional(preds, exs, orig, k=k)
        mean_coef = [statistics.fmean(c_[i] for c_ in coefs) for i in range(len(FEATURES))]
        sd_coef = [statistics.pstdev([c_[i] for c_ in coefs]) for i in range(len(FEATURES))]

        print(f"    stacked      SP@4 {statistics.fmean(v['sp_recall4'] for v in per_q.values()):.4f}  "
              f"AllGold@4 {ag['unconditional']:.3f} (cond {ag['conditional']:.3f} "
              f"on {ag['n_conditional']}/{ag['n_all']})")
        print(f"    coefficients (mean over 5 folds, features min-max normalized per question):")
        for name, m, sd in zip(FEATURES, mean_coef, sd_coef):
            print(f"      {name:<12} {m:+.4f}  (sd {sd:.4f})")

        # paired bootstrap vs Vector RAG and vs the RRF pair
        cis = {}
        for other in ("vector_rag", "ppr_plus_vector"):
            o = refs[ds][other]["per_question"]
            qids = sorted(set(per_q) & set(o))
            for metric in ("allgold4", "sp_recall4"):
                a = [per_q[q][metric] for q in qids]
                bb = [o[q][metric] for q in qids]
                ci = paired_bootstrap(a, bb, n=10_000)
                cis[f"{metric}|stacked-vs-{other}"] = ci.as_dict()
                v = "excludes 0" if ci.excludes_zero else "INCLUDES 0 (within noise)"
                print(f"    {metric:<12} stacked - {other:<16} {ci.point:+.4f} "
                      f"[{ci.lo:+.4f}, {ci.hi:+.4f}]  {v}")
        out["datasets"][ds] = {
            "sp_recall4": statistics.fmean(v["sp_recall4"] for v in per_q.values()),
            "allgold4_uncond": ag["unconditional"], "allgold4_cond": ag["conditional"],
            "n_cond": ag["n_conditional"], "n_all": ag["n_all"],
            "coef_mean": dict(zip(FEATURES, mean_coef)),
            "coef_sd": dict(zip(FEATURES, sd_coef)),
            "paired_bootstrap": cis,
            "per_question": per_q,
        }
        emb.free()

    (ROOT / "results" / "m6_stack.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n-> results/m6_stack.json")

    # the D20 gate
    print(f"\n=== D20 GATE ===")
    gains = {}
    for ds in DATASETS:
        ci = out["datasets"][ds]["paired_bootstrap"]["allgold4|stacked-vs-vector_rag"]
        gains[ds] = ci
        print(f"  {ds}: stacked - vector_rag AllGold@4 {ci['point']:+.4f} "
              f"[{ci['lo']:+.4f}, {ci['hi']:+.4f}]  "
              f"{'excludes 0' if ci['excludes_zero'] else 'includes 0'}")
    any_sig = any(g["excludes_zero"] and g["point"] > 0 for g in gains.values())
    both_pos = all(g["point"] > 0 for g in gains.values())
    opened = any_sig or both_pos
    print(f"  gate: CI excludes 0 with a gain on >=1 dataset -> {any_sig}")
    print(f"  gate: clearly positive point estimate on both   -> {both_pos}")
    print(f"  => D20 {'OPENS' if opened else 'is SKIPPED'}")
    out["d20_gate"] = {"any_significant_gain": any_sig, "both_positive": both_pos,
                       "opened": opened}
    (ROOT / "results" / "m6_stack.json").write_text(json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    run(load_config())
