"""Pareto analysis: cost vs quality (agent.md Section 11, M10).

Written as a jupytext-percent script so it is diffable and runnable headless:
    .venv/bin/python notebooks/pareto_analysis.py
    .venv/bin/jupytext --to notebook notebooks/pareto_analysis.py   # if a .ipynb is wanted

Three cost axes (routing latency, query-time LLM tokens, peak process-tree RSS)
against two quality axes (AllGold@4 and reader EM), with indexing and query cost
shown separately and combined. Vector RAG is included throughout. Every point
whose underlying stage failed §7.3 is drawn hollow and labelled.
"""

# %%
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = ROOT / "results"
DS = ("musique", "hotpotqa")


def j(p):
    return json.loads((RES / p).read_text())


# %% [markdown]
# ## Assembling the table
#
# Latency is the FULL §7.3 routing stage from M7a (assembly + forward), never the
# forward-only M7c figure. LLM-Trace latency is indicative (D24): three of its four
# measurements tripped the swap rule. Indexing cost is shared by every graph method
# and is amortised per question.

# %%
idx = j("m7_indexing_cost.json")
m7a = j("m7a_report.json")
emf1 = j("m8_emf1.json")

# indexing, amortised per question. Vector RAG needs paragraph embeddings but no
# extraction; graph methods pay extraction + entity/relation embedding.
POOL = {"musique": 14, "hotpotqa": 10}
T_P = idx["t_p_mean_s"]
EMB_PER_Q = idx["embedding"]["seconds_per_question"]
TOK_PER_PARA = idx["llm_tokens_per_paragraph_mean"]

METHODS = [
    ("vector",   "Vector RAG",       False),
    ("ppr",      "PPR",              True),
    ("edge_sim", "Edge-Sim",         True),
    ("gnn",      "GNN",              True),
    ("gnn_rrf",  "GNN+Vector RRF",   True),
    ("ppr_rrf",  "PPR+Vector RRF",   True),
    ("stacked",  "Stacked (D19)",    True),
]
READER_KEY = {"vector": "vector", "ppr": "ppr", "gnn": "gnn",
              "gnn_rrf": "gnn_rrf", "stacked": "stacked"}

rows = []
for ds in DS:
    pool = POOL[ds]
    index_s = pool * T_P + EMB_PER_Q          # per question, graph methods
    index_tok = pool * TOK_PER_PARA
    for key, label, is_graph in METHODS:
        mm = m7a["datasets"][ds]["methods"][key]["ALL"]
        eff = m7a["datasets"][ds]["efficiency"][key]
        rows.append({
            "dataset": ds, "method": key, "label": label, "is_graph": is_graph,
            "allgold4": mm["allgold4"],
            "allgold4_cond": mm["allgold4_cond"],
            "ceiling": m7a["datasets"][ds]["context"]["ALL"]["allgold4_ceiling"],
            "em": emf1["datasets"][ds]["rows"].get(READER_KEY.get(key, ""), {}).get("em"),
            "routing_ms": eff["routing"]["median_s"] * 1000,
            "query_tokens": 0,
            "peak_rss_mb": eff["routing"]["peak_tree_rss_mb"],
            "index_s_per_q": index_s if is_graph else EMB_PER_Q,
            "index_tokens_per_q": index_tok if is_graph else 0,
            "valid": not eff["routing"]["invalidated_by_swap"],
            "indicative": False,
        })
    # LLM-Trace: query-time LLM calls, indicative efficiency
    tr = j(f"m7c_llm_trace_{ds}_full.json")
    trm = {r["qid"]: r for r in (json.loads(l) for l in
           (RES / f"val_llm_trace_{ds}.jsonl").read_text().splitlines() if l.strip())}
    rows.append({
        "dataset": ds, "method": "llm_trace", "label": "LLM-Trace (1.5B)",
        "is_graph": True,
        "allgold4": sum(r["allgold4"] for r in trm.values()) / len(trm),
        "allgold4_cond": None,
        "ceiling": m7a["datasets"][ds]["context"]["ALL"]["allgold4_ceiling"],
        "em": emf1["datasets"][ds]["rows"]["llm_trace"]["em"],
        "routing_ms": tr["median_s"] * 1000,
        "query_tokens": tr["tokens_per_question_mean"],
        "peak_rss_mb": tr["peak_tree_rss_mb"],
        "index_s_per_q": POOL[ds] * T_P + EMB_PER_Q,
        "index_tokens_per_q": POOL[ds] * TOK_PER_PARA,
        "valid": not tr["invalidated_by_swap"],
        "indicative": True,
    })
    rows.append({
        "dataset": ds, "method": "oracle", "label": "Gold oracle",
        "is_graph": False, "allgold4": 1.0, "allgold4_cond": 1.0,
        "ceiling": 1.0, "em": emf1["datasets"][ds]["rows"]["oracle"]["em"],
        "routing_ms": None, "query_tokens": 0, "peak_rss_mb": None,
        "index_s_per_q": 0, "index_tokens_per_q": 0, "valid": True, "indicative": False,
    })

# %%
def fmt(v, spec=".3f", dash="-"):
    return format(v, spec) if v is not None else dash


print(f"{'dataset':<9} {'method':<18} {'AllGold@4':>10} {'EM':>7} {'route ms':>9} "
      f"{'q tokens':>9} {'peakRSS':>8} {'index s/q':>10} flag")
for r in rows:
    flag = "" if r["valid"] else "SWAP-INVALID"
    if r["indicative"]:
        flag = (flag + " INDICATIVE").strip()
    print(f"{r['dataset']:<9} {r['label']:<18} {r['allgold4']:>10.3f} "
          f"{fmt(r['em']):>7} {fmt(r['routing_ms'], '.1f'):>9} "
          f"{r['query_tokens']:>9.0f} {fmt(r['peak_rss_mb'], '.0f'):>8} "
          f"{r['index_s_per_q']:>10.2f} {flag}")

# %% [markdown]
# ## The plots
#
# Cost on a log x-axis (the methods span three orders of magnitude). Hollow markers
# are stages that failed §7.3; a dashed horizontal line is the graph ceiling, which
# bounds every graph method but not Vector RAG — that gap is the central finding.

# %%
def panel(ax, rows, xkey, ykey, xlabel, ylabel, logx=True):
    for r in rows:
        if r.get(xkey) is None or r.get(ykey) is None:
            continue
        filled = r["valid"] and not r["indicative"]
        ax.scatter(r[xkey], r[ykey], s=130,
                   facecolors=("tab:blue" if r["is_graph"] else "tab:orange") if filled else "none",
                   edgecolors="tab:blue" if r["is_graph"] else "tab:orange",
                   linewidths=1.8, zorder=3)
        ax.annotate(r["label"], (r[xkey], r[ykey]), textcoords="offset points",
                    xytext=(7, 4), fontsize=7)
    ceil = next((r["ceiling"] for r in rows if r["is_graph"]), None)
    if ceil and ykey == "allgold4":
        ax.axhline(ceil, ls="--", lw=1, color="tab:red")
        ax.annotate(f"graph ceiling {ceil:.3f}", (ax.get_xlim()[0], ceil),
                    textcoords="offset points", xytext=(4, 4), fontsize=7, color="tab:red")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3, zorder=0)


for ds in DS:
    rs = [r for r in rows if r["dataset"] == ds]
    for r in rs:
        r["total_s_per_q"] = (r["index_s_per_q"] +
                              (r["routing_ms"] or 0) / 1000)
        r["total_tokens"] = r["index_tokens_per_q"] + r["query_tokens"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Pareto: cost vs quality — {ds} validation (n=150)\n"
                 f"hollow = stage failed §7.3 swap rule or is indicative (D24/D26)",
                 fontsize=10)
    panel(axes[0][0], rs, "routing_ms", "allgold4", "routing latency, ms (query only)", "AllGold@4")
    panel(axes[0][1], rs, "total_tokens", "allgold4", "LLM tokens/question (indexing + query)", "AllGold@4")
    panel(axes[0][2], rs, "peak_rss_mb", "allgold4", "peak process-tree RSS, MB", "AllGold@4", logx=False)
    panel(axes[1][0], rs, "routing_ms", "em", "routing latency, ms (query only)", "reader EM")
    panel(axes[1][1], rs, "total_s_per_q", "em", "seconds/question (indexing + query)", "reader EM")
    panel(axes[1][2], rs, "query_tokens", "em", "query-time LLM tokens/question", "reader EM", logx=False)
    fig.tight_layout()
    p = RES / f"pareto_{ds}.png"
    fig.savefig(p, dpi=150)
    print(f"  wrote {p.relative_to(ROOT)}")

# %%
(RES / "pareto_table.json").write_text(json.dumps(rows, indent=2, default=str))
print("  wrote results/pareto_table.json")
print("""
Reading the panels:
  * Vector RAG sits at the far left of every latency panel (0.2-0.5 ms) and pays NO
    extraction, so its indexing cost is embeddings only. It is also at or above every
    graph method on AllGold@4. That is C1's negative result in one picture.
  * The red dashed line is the graph ceiling. On MuSiQue it is 0.380 and Vector RAG
    scores 0.267 BELOW it while still beating the graph methods -- the ceiling binds
    the graph arms, not the baseline.
  * LLM-Trace is three orders of magnitude slower and is the only method paying
    query-time tokens; it is drawn hollow because its efficiency is indicative only.
  * Combining indexing with query cost (bottom middle) is where graph methods look
    worst: ~22 s/question of extraction on MuSiQue against Vector RAG's ~0.02 s.
""")
