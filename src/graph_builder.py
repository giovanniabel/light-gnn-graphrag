"""Per-question graph construction (agent.md Section 4.1), labels (4.2) and the
PyG conversion with the Section 4.3 feature layout.

Pipeline per question:
  nodes   normalized entity strings, exact matches merged, optionally near-dupes
          merged at embedding cosine >= merge_cosine
  edges   one directed edge per triple plus a reverse edge carrying a direction
          bit; every edge keeps its source paragraph pid
  seeds   nodes whose normalized name is a substring of the normalized question;
          if none, the top-3 nodes by cosine to the question; capped at 5
  ego     undirected BFS from the seeds to H hops, capped at max_nodes /
          max_edges in BFS order
  labels  y=1 iff the edge's source paragraph is a gold supporting paragraph
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import networkx as nx
import torch

from .normalize import normalize_answer, normalize_entity
from .schemas import Example, Triple

H_DEFAULT = 3

# Section 4.3 feature layout.
#   x        = [e_v ; e_v*q ; cos(e_v,q) ; is_seed ; onehot(hop 0..H) ; log1p(deg)
#               ; node-kind indicators]
#   edge_attr= [r_e ; cos(r_e,q) ; direction_bit ; edge-kind one-hot]
# Node-kind indicators are MULTI-hot, not one-hot: a title node that merges with an
# identically named entity is genuinely both, and collapsing that to one slot would
# hide the merge that the connectivity fix depends on.
NODE_KINDS = ("entity", "title", "para")          # 3 indicator dims
EDGE_KINDS = ("triple", "title", "same_as", "para")  # 4 one-hot dims
NODE_FEAT_DIM = 384 * 2 + 1 + 1 + (H_DEFAULT + 1) + 1 + len(NODE_KINDS)   # 778
EDGE_FEAT_DIM = 384 + 1 + 1 + len(EDGE_KINDS)                             # 390

SEED_SUBSTRING, SEED_FALLBACK, SEED_NONE = "substring", "embedding_fallback", "none"

# Edge kinds. "triple" is extracted evidence; the rest are connectivity scaffolding
# added by Section 4.1.1 and never read a gold label.
KIND_TRIPLE, KIND_TITLE, KIND_SAME_AS, KIND_PARA = "triple", "title", "same_as", "para"

# Synthetic relation phrases (embedded like any other relation for edge_attr).
REL_TITLE = "mentions"
REL_SAME_AS = "same as"
REL_PARA = "appears in paragraph"
SYNTHETIC_RELATIONS = (REL_TITLE, REL_SAME_AS, REL_PARA)

# same_as joins two surface forms, so it has no single source paragraph. The
# sentinel pid is never a real pid, so these edges are always y=0 and are skipped
# by paragraph scoring: they add reachability, never evidence.
SAME_AS_PID = "__same_as__"


def _tokens(name: str) -> frozenset[str]:
    return frozenset(name.split())


def _token_match(a: str, b: str, jaccard: float) -> bool:
    """Containment either way, or token Jaccard at/above the threshold."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    if ta <= tb or tb <= ta:
        return True
    inter = len(ta & tb)
    return inter > 0 and inter / len(ta | tb) >= jaccard


DEFAULT_CONNECTIVITY = {
    "title_hubs": False,
    "title_fuzzy": True,
    "title_token_jaccard": 0.5,
    "same_as_edges": False,
    "same_as_cosine": 0.85,
    "same_as_token_match": True,
    "same_as_max_per_node": 3,
    "paragraph_nodes": False,
}


@dataclass
class QueryGraph:
    """One question's ego-graph, with everything the PyG conversion needs."""

    qid: str
    dataset: str
    question: str
    answer: str
    n_hops: int | None

    nodes: list[str]                       # normalized names, index = node id
    edges: list[tuple[int, int]]           # (src, dst) into nodes
    edge_rel: list[str]                    # relation phrase per edge
    edge_pid: list[str]                    # source paragraph per edge
    edge_dir: list[int]                    # 1 = original triple, 0 = reverse
    edge_y: list[int]                      # provenance label (Section 4.2)
    seeds: list[int]
    seed_mode: str
    hop_dist: list[int]                    # BFS hop distance from the nearest seed
    edge_kind: list[str] = field(default_factory=list)   # triple | title | same_as | para
    node_pids: list[set] = field(default_factory=list)   # which paragraphs a node came from
    node_kind: list[frozenset] = field(default_factory=list)  # subset of NODE_KINDS
    # Text to embed for each node. Identical to the node name for entity and title
    # nodes; for a pid-keyed paragraph node it is that paragraph's title, so the
    # node carries a real bge vector instead of zeros. Without this, every
    # paragraph hub would enter the model with a 768-dim zero block, and the L=0
    # ablation would differ from the full model in its inputs as well as its depth.
    node_text: list[str] = field(default_factory=list)

    gold_pids: set[str] = field(default_factory=set)
    pool_pids: list = field(default_factory=list)   # every paragraph in this question's pool
    answer_aliases: list[str] = field(default_factory=list)
    # diagnostics
    n_nodes_full: int = 0
    n_edges_full: int = 0
    n_components_full: int = 0
    node_cap_bound: bool = False
    edge_cap_bound: bool = False
    merged_near_duplicates: int = 0
    # Section 4.1.1 diagnostics. cross_paragraph_links records, for every added
    # connectivity link, the pairs of distinct paragraphs it joins. Used for a
    # TRAIN-only quality check; never read as a feature.
    added_links: dict = field(default_factory=dict)
    cross_paragraph_links: list = field(default_factory=list)

    @property
    def evidence_edges(self) -> int:
        """Edges that are extracted evidence, i.e. kind == triple.

        The Section 4.1.1 scaffolding (title, same_as, para) adds reachability and
        nothing else: it is synthesized by us, not extracted from the paragraph, so
        it never counts as evidence, never carries y=1, and never makes a paragraph
        scorable on its own.
        """
        return sum(1 for k in self.edge_kind if k == KIND_TRIPLE)

    def covered_pids(self) -> set:
        """Paragraphs a graph method can actually rank: those with at least one
        EXTRACTED (kind == triple) edge in the ego-graph. Scaffolding edges are
        excluded, so a paragraph reached only through its own hub is not counted."""
        return {
            pid for pid, kind in zip(self.edge_pid, self.edge_kind)
            if kind == KIND_TRIPLE and pid != SAME_AS_PID
        }

    def covered_pids_any_edge(self) -> set:
        """Diagnostic only: paragraphs with any edge, scaffolding included."""
        return {p for p in self.edge_pid if p != SAME_AS_PID}

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def positive_edges(self) -> int:
        return sum(self.edge_y)

    def answer_node_ids(self, use_aliases: bool = False) -> list[int]:
        """Node ids whose normalized name equals the answer (or an alias)."""
        cands = {normalize_answer(self.answer)}
        if use_aliases:
            cands |= {normalize_answer(a) for a in self.answer_aliases if a}
        cands.discard("")
        return [i for i, n in enumerate(self.nodes) if n in cands]

    def answer_in_graph(self, use_aliases: bool = False) -> bool:
        return bool(self.answer_node_ids(use_aliases))

    def path_from_seed_to_answer(self, use_aliases: bool = False) -> bool:
        """Is an answer node connected to a seed within this ego-graph?

        The ego-graph is itself a BFS ball around the seeds, so membership
        implies reachability; this checks it explicitly rather than assuming it,
        because the node/edge caps can truncate the ball and orphan a node.
        """
        targets = set(self.answer_node_ids(use_aliases))
        if not targets or not self.seeds:
            return False
        g = nx.Graph()
        g.add_nodes_from(range(len(self.nodes)))
        g.add_edges_from(self.edges)
        seen = set(self.seeds)
        dq = deque(self.seeds)
        while dq:
            v = dq.popleft()
            if v in targets:
                return True
            for w in g.neighbors(v):
                if w not in seen:
                    seen.add(w)
                    dq.append(w)
        return False


# ------------------------------------------------------------------ nodes
def _merge_near_duplicates(names: list[str], embedder, threshold: float) -> tuple[dict[str, str], int]:
    """Map each name to a canonical name, merging pairs with cosine >= threshold.

    Greedy single pass in descending length order, so the longer (more specific)
    surface form becomes canonical.
    """
    if len(names) < 2 or embedder is None:
        return {n: n for n in names}, 0
    order = sorted(names, key=lambda n: (-len(n), n))
    vecs = embedder.encode(order, kind="entities")
    vecs = torch.nn.functional.normalize(vecs, dim=-1)
    canon: dict[str, str] = {}
    kept_idx: list[int] = []
    merged = 0
    for i, name in enumerate(order):
        if kept_idx:
            sims = vecs[kept_idx] @ vecs[i]
            best = int(torch.argmax(sims))
            if float(sims[best]) >= threshold:
                canon[name] = order[kept_idx[best]]
                merged += 1
                continue
        canon[name] = name
        kept_idx.append(i)
    return canon, merged


# ------------------------------------------------------------------ seeds
def find_seeds(
    question: str,
    nodes: list[str],
    embedder=None,
    max_seeds: int = 5,
    fallback_topk: int = 3,
) -> tuple[list[int], str]:
    """Section 4.1 step 4. No LLM.

    Nodes whose normalized name is a substring of the normalized question; if
    fewer than one is found, fall back to the top-k nodes by cosine to the
    question embedding. Capped at max_seeds.
    """
    if not nodes:
        return [], SEED_NONE
    nq = normalize_entity(question)
    hits = [i for i, n in enumerate(nodes) if n and n in nq]
    if hits:
        # longest names first: the most specific match is the most useful seed
        hits.sort(key=lambda i: (-len(nodes[i]), i))
        return hits[:max_seeds], SEED_SUBSTRING
    if embedder is None:
        return [], SEED_NONE
    q = embedder.encode_query(question)
    v = embedder.encode(nodes, kind="entities")
    sims = (torch.nn.functional.normalize(v, dim=-1) @ torch.nn.functional.normalize(q, dim=-1))
    k = min(fallback_topk, max_seeds, len(nodes))
    top = torch.topk(sims, k).indices.tolist()
    return top, SEED_FALLBACK


# ------------------------------------------------------------------ build
def build_graph(
    ex: Example,
    triples: list[Triple],
    embedder,
    hops: int = 3,
    max_nodes: int = 200,
    max_edges: int = 400,
    merge_near_duplicates: bool = True,
    merge_cosine: float = 0.92,
    max_seeds: int = 5,
    seed_fallback_topk: int = 3,
    connectivity: dict | None = None,
) -> QueryGraph:
    """Build one question's ego-graph (agent.md Section 10 signature).

    `connectivity` switches on the Section 4.1.1 cross-paragraph mechanisms. All
    of them read only pid/title/text; none reads is_gold. Gold flags are consulted
    exactly once, at the very end, to assign edge labels (Section 4.2).
    """
    conn = {**DEFAULT_CONNECTIVITY, **(connectivity or {})}
    gold_pids = ex.gold_pids

    # --- nodes -----------------------------------------------------------
    raw_names: list[str] = []
    seen_names: set[str] = set()
    norm_triples: list[tuple[str, str, str, str]] = []
    prov: dict[str, set] = {}          # normalized name -> paragraphs it came from
    kinds: dict[str, set] = {}         # normalized name -> subset of NODE_KINDS
    for t in triples:
        h, ta = normalize_entity(t.head), normalize_entity(t.tail)
        rel = " ".join(str(t.rel).split())
        if not h or not ta or h == ta or not rel:
            continue
        norm_triples.append((h, rel, ta, t.pid))
        for n in (h, ta):
            prov.setdefault(n, set()).add(t.pid)
            kinds.setdefault(n, set()).add("entity")
            if n not in seen_names:
                seen_names.add(n)
                raw_names.append(n)

    # (a) title hub nodes: one per pool paragraph, named by its normalized title.
    # Added to the node set BEFORE merging so a title merges with an identically
    # named entity from another paragraph -- that merge is what bridges components.
    titles: dict[str, str] = {}        # pid -> normalized title
    if conn["title_hubs"]:
        for para in ex.paragraphs:
            tnorm = normalize_entity(para.title)
            if not tnorm:
                continue
            titles[para.pid] = tnorm
            prov.setdefault(tnorm, set()).add(para.pid)
            kinds.setdefault(tnorm, set()).add("title")
            if tnorm not in seen_names:
                seen_names.add(tnorm)
                raw_names.append(tnorm)

    # (c) one pid-keyed node per paragraph. These never merge across paragraphs,
    # so they guarantee a paragraph is representable even if nothing links it.
    para_nodes: dict[str, str] = {}
    para_text: dict[str, str] = {}
    if conn["paragraph_nodes"]:
        for para in ex.paragraphs:
            pn = f"__para__{para.pid}"
            para_nodes[para.pid] = pn
            para_text[pn] = normalize_entity(para.title) or para.title
            prov.setdefault(pn, set()).add(para.pid)
            kinds.setdefault(pn, set()).add("para")
            if pn not in seen_names:
                seen_names.add(pn)
                raw_names.append(pn)

    canon, n_merged = (
        _merge_near_duplicates(raw_names, embedder, merge_cosine)
        if merge_near_duplicates else ({n: n for n in raw_names}, 0)
    )
    nodes_full: list[str] = []
    node_id: dict[str, int] = {}
    for n in raw_names:
        c = canon.get(n, n)
        if c not in node_id:
            node_id[c] = len(nodes_full)
            nodes_full.append(c)
    # provenance follows the merge: a canonical node inherits every pid of every
    # surface form folded into it
    node_pids_full: list[set] = [set() for _ in nodes_full]
    node_kind_full: list[set] = [set() for _ in nodes_full]
    for n in raw_names:
        i = node_id[canon.get(n, n)]
        node_pids_full[i] |= prov.get(n, set())
        node_kind_full[i] |= kinds.get(n, set())

    # --- edges (one per triple + a flagged reverse) -----------------------
    full_edges: list[tuple[int, int, str, str, int, str]] = []

    def _add(u: int, v: int, rel: str, pid: str, kind: str) -> None:
        if u == v:
            return
        full_edges.append((u, v, rel, pid, 1, kind))
        full_edges.append((v, u, rel, pid, 0, kind))

    for h, rel, ta, pid in norm_triples:
        _add(node_id[canon.get(h, h)], node_id[canon.get(ta, ta)], rel, pid, KIND_TRIPLE)

    added = {KIND_TITLE: 0, KIND_SAME_AS: 0, KIND_PARA: 0}

    # entities per paragraph, for the hub links
    ents_by_pid: dict[str, set[int]] = {}
    for h, _rel, ta, pid in norm_triples:
        for name in (h, ta):
            ents_by_pid.setdefault(pid, set()).add(node_id[canon.get(name, name)])

    # (a) title hub -> every entity of that paragraph, plus fuzzy title matches
    if conn["title_hubs"]:
        for pid, tnorm in titles.items():
            tid = node_id[canon.get(tnorm, tnorm)]
            for eid in ents_by_pid.get(pid, set()):
                _add(tid, eid, REL_TITLE, pid, KIND_TITLE)
                added[KIND_TITLE] += 1
            if conn["title_fuzzy"]:
                jac = float(conn["title_token_jaccard"])
                for eid, ename in enumerate(nodes_full):
                    if eid == tid or ename.startswith("__para__"):
                        continue
                    if _token_match(nodes_full[tid], ename, jac):
                        _add(tid, eid, REL_TITLE, pid, KIND_TITLE)
                        added[KIND_TITLE] += 1

    # (b) same_as between near-identical entity surface forms
    if conn["same_as_edges"]:
        cand = [i for i, n in enumerate(nodes_full) if not n.startswith("__para__")]
        if len(cand) > 1 and embedder is not None:
            vecs = torch.nn.functional.normalize(
                embedder.encode([nodes_full[i] for i in cand], kind="entities"), dim=-1
            )
            sim = vecs @ vecs.T
            thr = float(conn["same_as_cosine"])
            cap = int(conn["same_as_max_per_node"])
            for a in range(len(cand)):
                order = torch.argsort(sim[a], descending=True).tolist()
                linked = 0
                for b in order:
                    if b <= a or linked >= cap:
                        continue
                    ia, ib = cand[a], cand[b]
                    close = float(sim[a, b]) >= thr
                    if not close and conn["same_as_token_match"]:
                        close = _token_match(nodes_full[ia], nodes_full[ib],
                                             float(conn["title_token_jaccard"]))
                    if close and node_pids_full[ia] != node_pids_full[ib]:
                        _add(ia, ib, REL_SAME_AS, SAME_AS_PID, KIND_SAME_AS)
                        added[KIND_SAME_AS] += 1
                        linked += 1

    # (c) entity -> pid-keyed paragraph node
    if conn["paragraph_nodes"]:
        for pid, pn in para_nodes.items():
            pnid = node_id[canon.get(pn, pn)]
            for eid in ents_by_pid.get(pid, set()):
                _add(pnid, eid, REL_PARA, pid, KIND_PARA)
                added[KIND_PARA] += 1

    # --- seeds ------------------------------------------------------------
    seeds_full, seed_mode = find_seeds(
        ex.question, nodes_full, embedder, max_seeds, seed_fallback_topk
    )

    # --- ego-graph: undirected BFS from seeds to `hops`, caps in BFS order -
    adj: dict[int, list[int]] = {}
    for u, v, *_rest in full_edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    dist: dict[int, int] = {s: 0 for s in seeds_full}
    bfs_order: list[int] = list(seeds_full)
    dq = deque(seeds_full)
    while dq:
        u = dq.popleft()
        if dist[u] >= hops:
            continue
        for w in adj.get(u, []):
            if w not in dist:
                dist[w] = dist[u] + 1
                bfs_order.append(w)
                dq.append(w)

    # connected components of the FULL graph, before the ego restriction
    comp_parent = list(range(len(nodes_full)))

    def _find(a: int) -> int:
        while comp_parent[a] != a:
            comp_parent[a] = comp_parent[comp_parent[a]]
            a = comp_parent[a]
        return a

    for u, v, *_rest in full_edges:
        ru, rv = _find(u), _find(v)
        if ru != rv:
            comp_parent[ru] = rv
    n_components_full = len({_find(i) for i in range(len(nodes_full))}) if nodes_full else 0

    node_cap_bound = len(bfs_order) > max_nodes
    kept_nodes = bfs_order[:max_nodes]
    keep = {n: i for i, n in enumerate(kept_nodes)}

    kept_edges = [e for e in full_edges if e[0] in keep and e[1] in keep]
    edge_cap_bound = len(kept_edges) > max_edges
    kept_edges = kept_edges[:max_edges]

    return QueryGraph(
        qid=ex.qid, dataset=ex.dataset, question=ex.question, answer=ex.answer,
        n_hops=ex.n_hops,
        nodes=[nodes_full[n] for n in kept_nodes],
        edges=[(keep[u], keep[v]) for u, v, *_ in kept_edges],
        edge_rel=[e[2] for e in kept_edges],
        edge_pid=[e[3] for e in kept_edges],
        edge_dir=[e[4] for e in kept_edges],
        # Section 4.2: y=1 iff the edge's source paragraph is gold. This is the
        # ONLY place a gold flag is read. Restricted to kind == triple, so no
        # scaffolding edge can ever be a positive: same_as carries a sentinel pid,
        # and title/para edges carry a real pid but are not extracted evidence.
        edge_y=[
            1 if (e[5] == KIND_TRIPLE and e[3] in gold_pids) else 0
            for e in kept_edges
        ],
        edge_kind=[e[5] for e in kept_edges],
        node_pids=[set(node_pids_full[n]) for n in kept_nodes],
        node_kind=[frozenset(node_kind_full[n]) for n in kept_nodes],
        node_text=[para_text.get(nodes_full[n], nodes_full[n]) for n in kept_nodes],
        seeds=[keep[s] for s in seeds_full if s in keep],
        seed_mode=seed_mode,
        hop_dist=[min(dist.get(n, hops), hops) for n in kept_nodes],
        added_links=dict(added),
        cross_paragraph_links=[
            (e[5], tuple(sorted(node_pids_full[e[0]])), tuple(sorted(node_pids_full[e[1]])))
            for e in kept_edges
            if e[5] != KIND_TRIPLE and e[4] == 1
        ],
        gold_pids=set(gold_pids),
        pool_pids=[p.pid for p in ex.paragraphs],
        answer_aliases=list(ex.meta.get("answer_aliases") or []),
        n_nodes_full=len(nodes_full), n_edges_full=len(full_edges),
        n_components_full=n_components_full,
        node_cap_bound=node_cap_bound, edge_cap_bound=edge_cap_bound,
        merged_near_duplicates=n_merged,
    )


# ------------------------------------------------------------------ PyG
def to_pyg(qg: QueryGraph, emb_cache, hops: int = 3):
    """Section 4.3 feature layout.

    x        = [e_v ; e_v * q ; cos(e_v,q) ; is_seed ; onehot(hop 0..H) ; log1p(deg)]
    edge_attr= [r_e ; cos(r_e,q) ; direction_bit]
    """
    from torch_geometric.data import Data

    # No feature is ever zero-filled. A cache miss does not mean "this text has
    # no embedding", it means the embedding pass was not run for this split, and
    # a zero vector is not a neutral stand-in: it silently removes exactly the
    # query conditioning this model is built on.
    #   * question missing -> e_v*q, cos(e_v,q) and cos(r_e,q) all collapse to 0
    #     and the router scores edges query-blind (D29, the M11 HotpotQA pools).
    #   * relation missing -> r_e = 0 and cos(r_e,q) = 0, so that edge's entire
    #     edge_attr carries no signal (D30: 10% of the M11 extension's edges).
    # Both produced plausible-looking scores and neither announced itself, so
    # every kind now fails loudly. Published train/validation splits are 0% miss,
    # so this raises only where a pass is genuinely absent.
    def _emb(text: str, kind: str) -> torch.Tensor:
        v = emb_cache.get(text, kind)
        if v is None:
            raise KeyError(
                f"no cached {kind} embedding for {text[:60]!r} "
                f"(qid {qg.qid!r}, {qg.dataset}). to_pyg must not zero-fill "
                "features: that silently degrades query conditioning. Run the "
                "embedding pass for this split first (scripts/m3_embed.py, or "
                "scripts/ext_embed.py for an extension split)."
            )
        return v

    q = _emb(qg.question, "questions")
    qn = torch.nn.functional.normalize(q, dim=-1)

    n = len(qg.nodes)
    e = len(qg.edges)

    if n == 0:
        return Data(
            x=torch.zeros(0, NODE_FEAT_DIM), edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_attr=torch.zeros(0, EDGE_FEAT_DIM), y_edge=torch.zeros(0),
            q=q.unsqueeze(0), seed_mask=torch.zeros(0, dtype=torch.bool),
            edge_pid=[], qid=qg.qid, dataset=qg.dataset, n_hops=qg.n_hops,
        )

    texts = qg.node_text or qg.nodes
    ev = torch.stack([_emb(t, "entities") for t in texts])
    evn = torch.nn.functional.normalize(ev, dim=-1)
    cos_vq = (evn @ qn).unsqueeze(-1)

    seed_mask = torch.zeros(n, dtype=torch.bool)
    seed_mask[qg.seeds] = True

    onehot = torch.zeros(n, hops + 1)
    for i, d in enumerate(qg.hop_dist):
        onehot[i, min(max(d, 0), hops)] = 1.0

    # node-kind indicators (multi-hot: a merged title/entity node is both)
    nkind = torch.zeros(n, len(NODE_KINDS))
    for i, ks in enumerate(qg.node_kind or [frozenset({"entity"})] * n):
        for j, name in enumerate(NODE_KINDS):
            if name in ks:
                nkind[i, j] = 1.0

    deg = torch.zeros(n)
    for u, v in qg.edges:
        deg[u] += 1
        deg[v] += 1

    x = torch.cat(
        [ev, ev * q.unsqueeze(0), cos_vq, seed_mask.float().unsqueeze(-1),
         onehot, torch.log1p(deg).unsqueeze(-1), nkind],
        dim=-1,
    )

    edge_index = torch.tensor(qg.edges, dtype=torch.long).t().contiguous()
    if e:
        re_ = torch.stack([_emb(r, "relations") for r in qg.edge_rel])
        cos_rq = (torch.nn.functional.normalize(re_, dim=-1) @ qn).unsqueeze(-1)
        dirbit = torch.tensor(qg.edge_dir, dtype=torch.float32).unsqueeze(-1)
        ekind = torch.zeros(e, len(EDGE_KINDS))
        for i, k in enumerate(qg.edge_kind or [KIND_TRIPLE] * e):
            ekind[i, EDGE_KINDS.index(k)] = 1.0
        edge_attr = torch.cat([re_, cos_rq, dirbit, ekind], dim=-1)
        y_edge = torch.tensor(qg.edge_y, dtype=torch.float32)
    else:
        edge_attr = torch.zeros(0, EDGE_FEAT_DIM)
        y_edge = torch.zeros(0)

    return Data(
        x=x, edge_index=edge_index, edge_attr=edge_attr, y_edge=y_edge,
        q=q.unsqueeze(0), seed_mask=seed_mask, edge_pid=list(qg.edge_pid),
        edge_kind=list(qg.edge_kind),
        # True only for extracted triples: the edges that carry evidence and are
        # the only ones paragraph scoring may use (Section 4.4 / 4.1.1).
        evidence_mask=torch.tensor(
            [k == KIND_TRIPLE for k in qg.edge_kind], dtype=torch.bool
        ) if e else torch.zeros(0, dtype=torch.bool),
        qid=qg.qid, dataset=qg.dataset, n_hops=qg.n_hops,
    )


def is_yes_no(answer: str) -> bool:
    """Yes/no answers are excluded from answer-in-graph and Path-Recall ceilings:
    'yes' is never an entity in the graph, so counting it as a miss understates
    the ceiling (HotpotQA comparison questions are mostly yes/no)."""
    return normalize_answer(answer) in {"yes", "no"}
