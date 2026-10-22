"""Non-learned retrieval baselines (agent.md Sections 5, 10).

All of these are query-time LLM-free. `llm_trace` is deliberately absent here: it
needs Ollama loaded and belongs to M7, which runs as its own phase (§6.1).
"""

from __future__ import annotations

import time

import networkx as nx
import torch

from src.graph_builder import KIND_TRIPLE
from src.routing import fuse_with_vector, rank_paragraphs
from src.schemas import Routed, StageStats


def _routed(ranked, edges, t0) -> Routed:
    return Routed(paragraph_ids=list(ranked), edges=list(edges),
                  stats=StageStats(latency_s=time.perf_counter() - t0, stage="routing"))


# ---------------------------------------------------------------- Vector RAG
def vector_rag(ex, embedder, k: int = 4) -> Routed:
    """Rank pool paragraphs by bge cosine to the question; top-k (Section 5).

    The primary baseline, and the one with no graph ceiling: it can rank every
    paragraph in the pool.
    """
    t0 = time.perf_counter()
    q = embedder.encode_query(ex.question)
    texts = [f"{p.title}. {p.text}" for p in ex.paragraphs]
    v = embedder.encode(texts, kind="paragraphs")
    sims = (torch.nn.functional.normalize(v, dim=-1)
            @ torch.nn.functional.normalize(q, dim=-1)).tolist()
    order = sorted(range(len(sims)), key=lambda i: -sims[i])
    return _routed([ex.paragraphs[i].pid for i in order], [], t0)


def vector_cosines(ex, embedder) -> dict[str, float]:
    """pid -> cosine, for the hybrid variants and RRF fusion."""
    q = embedder.encode_query(ex.question)
    v = embedder.encode([f"{p.title}. {p.text}" for p in ex.paragraphs], kind="paragraphs")
    sims = (torch.nn.functional.normalize(v, dim=-1)
            @ torch.nn.functional.normalize(q, dim=-1)).tolist()
    return {p.pid: s for p, s in zip(ex.paragraphs, sims)}


# ---------------------------------------------------------------- PPR
def ppr_rag(qg, ex=None, k: int = 4, alpha: float = 0.85, cfg=None) -> Routed:
    """Personalized PageRank from the seed nodes (HippoRAG-style, Section 5).

    A paragraph's score is the summed PPR mass of its edges' endpoints, over
    extracted edges only -- scaffolding carries no provenance, exactly as for the
    GNN (Section 4.1.1), so the two methods are scored on the same footing.
    """
    t0 = time.perf_counter()
    g = nx.Graph()
    g.add_nodes_from(range(len(qg.nodes)))
    g.add_edges_from(qg.edges)
    if g.number_of_nodes() == 0:
        return _routed([p.pid for p in ex.paragraphs] if ex else [], [], t0)

    seeds = [s for s in qg.seeds if s in g]
    pers = {n: 0.0 for n in g.nodes}
    for s in (seeds or list(g.nodes)[:1]):
        pers[s] = 1.0 / max(len(seeds or [1]), 1)
    try:
        pr = nx.pagerank(g, alpha=alpha, personalization=pers)
    except Exception:
        pr = {n: 1.0 / g.number_of_nodes() for n in g.nodes}

    edge_scores = []
    for i, (u, v) in enumerate(qg.edges):
        edge_scores.append(pr.get(u, 0.0) + pr.get(v, 0.0))

    pool = [p.pid for p in ex.paragraphs] if ex else list(dict.fromkeys(qg.edge_pid))

    class _V:
        edge_pid = list(qg.edge_pid)
        edge_kind = list(qg.edge_kind)

    ranked = rank_paragraphs(_V(), edge_scores, pool, k=k)
    return _routed(ranked, sorted(enumerate(edge_scores), key=lambda t: -t[1]), t0)


# ---------------------------------------------------------------- Edge-Sim
def edge_sim(qg, embedder, ex=None, k: int = 4) -> Routed:
    """Rank edges by cosine of their [head; rel; tail] text to the question.

    Isolates "graph vs no graph": it uses the extracted triples but no graph
    structure at all.
    """
    t0 = time.perf_counter()
    if not qg.edges:
        return _routed([p.pid for p in ex.paragraphs] if ex else [], [], t0)
    q = torch.nn.functional.normalize(embedder.encode_query(qg.question), dim=-1)
    texts = [
        f"{qg.nodes[u]} {rel} {qg.nodes[v]}"
        for (u, v), rel in zip(qg.edges, qg.edge_rel)
    ]
    v = torch.nn.functional.normalize(embedder.encode(texts, kind="edges"), dim=-1)
    scores = (v @ q).tolist()
    pool = [p.pid for p in ex.paragraphs] if ex else list(dict.fromkeys(qg.edge_pid))

    class _V:
        edge_pid = list(qg.edge_pid)
        edge_kind = list(qg.edge_kind)

    ranked = rank_paragraphs(_V(), scores, pool, k=k)
    return _routed(ranked, sorted(enumerate(scores), key=lambda t: -t[1]), t0)


# ---------------------------------------------------------------- RRF pairs
def rrf_pair(graph_ranking: list[str], ex, embedder, k: int = 4) -> Routed:
    """D15's matched hybrid: fuse a graph ranking with Vector RAG by RRF, k=60."""
    t0 = time.perf_counter()
    pool = [p.pid for p in ex.paragraphs]
    vec = vector_rag(ex, embedder, k=k).paragraph_ids
    return _routed(fuse_with_vector(graph_ranking, vec, pool), [], t0)


# ---------------------------------------------------------------- LLM-Trace
TRACE_SCHEMA = {
    "type": "object",
    "properties": {
        "edges": {"type": "array", "minItems": 0, "maxItems": 3,
                  "items": {"type": "integer"}}
    },
    "required": ["edges"],
}

TRACE_SYSTEM = ("You trace a path through a knowledge graph to answer a question. "
                "Reply with JSON only.")


def trace_candidates(qg, frontier: set[int], used: set[int], limit: int = 20) -> list[int]:
    """Candidate edge indices for one LLM-Trace step.

    THIS IS THE TRIPLE-ONLY ENFORCEMENT POINT. An edge is offered to the model
    only if its kind is `triple`, so the Section 4.1.1 scaffolding (title,
    same_as, para) is never a traversal option: LLM-Trace walks extracted
    evidence only, which keeps it comparable to the other graph methods whose
    paragraph scoring is likewise evidence-only.
    """
    kinds = qg.edge_kind or [KIND_TRIPLE] * len(qg.edges)
    deg: dict[int, int] = {}
    for u, v in qg.edges:
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1
    cand = []
    for i, (u, v) in enumerate(qg.edges):
        if i in used:
            continue
        if kinds[i] != KIND_TRIPLE:          # <-- scaffolding can never be traversed
            continue
        if u in frontier or v in frontier:
            other = v if u in frontier else u
            cand.append((deg.get(other, 0), i))
    cand.sort(key=lambda t: (-t[0], t[1]))   # Section 5: chosen by degree
    return [i for _d, i in cand[:limit]]


def llm_trace(qg, ex, model_tag: str, k: int = 4, steps: int = 3,
              max_candidates: int = 20, num_predict: int = 40, client=None) -> Routed:
    """Bounded iterative traversal with the SLM (agent.md Section 5, Section 10).

    Up to `steps` calls; each offers <= max_candidates outgoing TRIPLE edges chosen
    by degree and asks for <= 3 edge indices as JSON.
    """
    import json as _json

    from src.ollama_client import OllamaClient

    t0 = time.perf_counter()
    cli = client or OllamaClient(model_tag, num_ctx=2048, temperature=0.0)
    pool = [p.pid for p in ex.paragraphs]
    if not qg.edges or not qg.seeds:
        return _routed(pool, [], t0)

    frontier: set[int] = set(qg.seeds)
    used: set[int] = set()
    chosen: list[int] = []
    prompt_tokens = eval_tokens = 0
    n_calls = 0

    for _step in range(steps):
        cand = trace_candidates(qg, frontier, used, max_candidates)
        if not cand:
            break
        lines = [f"{j}: {qg.nodes[qg.edges[i][0]]} -[{qg.edge_rel[i]}]-> "
                 f"{qg.nodes[qg.edges[i][1]]}" for j, i in enumerate(cand)]
        front = ", ".join(sorted({qg.nodes[n] for n in frontier})[:10])
        prompt = (f"Question: {qg.question}\n"
                  f"Current entities: {front}\n"
                  f"Candidate edges:\n" + "\n".join(lines) +
                  "\n\nChoose up to 3 edge numbers that help answer the question.\n"
                  'Reply as JSON: {"edges": [0, 1]}\nJSON:')
        try:
            out, usage, _raw = cli.generate(prompt, system=TRACE_SYSTEM,
                                            fmt=TRACE_SCHEMA, num_predict=num_predict)
            n_calls += 1
            prompt_tokens += usage.prompt_tokens
            eval_tokens += usage.completion_tokens
            picked = _json.loads(out).get("edges", [])
        except Exception:
            break
        step_new = []
        for j in picked:
            if isinstance(j, int) and 0 <= j < len(cand):
                i = cand[j]
                if i not in used:
                    used.add(i)
                    chosen.append(i)
                    step_new.append(i)
        if not step_new:
            break
        for i in step_new:
            u, v = qg.edges[i]
            frontier.add(u)
            frontier.add(v)

    # collected edges -> paragraphs, earliest-selected first, then pool order
    seen, ranked = set(), []
    for i in chosen:
        pid = qg.edge_pid[i]
        if pid not in seen and pid in pool:
            seen.add(pid)
            ranked.append(pid)
    ranked += [p for p in pool if p not in seen]

    st = StageStats(latency_s=time.perf_counter() - t0,
                    llm_tokens=prompt_tokens + eval_tokens, stage="routing")
    r = Routed(paragraph_ids=ranked,
               edges=[(i, float(len(chosen) - n)) for n, i in enumerate(chosen)],
               stats=st)
    r.n_calls = n_calls
    r.prompt_tokens = prompt_tokens
    r.eval_tokens = eval_tokens
    return r


def edge_scores_for_ap(qg, scores: list[float]) -> tuple[list[float], list[float]]:
    """Evidence-edge (score, label) pairs, so edge AP is comparable across methods."""
    kinds = qg.edge_kind or [KIND_TRIPLE] * len(qg.edge_pid)
    s, y = [], []
    for i, kind in enumerate(kinds):
        if kind == KIND_TRIPLE:
            s.append(scores[i])
            y.append(float(qg.edge_y[i]))
    return s, y
