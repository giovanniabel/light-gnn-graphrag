"""M11 step 5: generate the "Validation extension" section and append it to report.md.

Idempotent: the section lives between two HTML-comment markers, so re-running
replaces only that block. Earlier sections are never edited; the one change
outside the block is a single pointer line in Section 2, inserted once.

Every table cites the results/ files it came from, as in make_report_tables.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RES = ROOT / "results"
DS = ("musique", "hotpotqa")
SETS = ("orig150", "new350", "pooled500")
SETLAB = {"orig150": "original 150", "new350": "**new 350**", "pooled500": "pooled 500"}
START = "<!-- M11-VALIDATION-EXTENSION:START -->"
END = "<!-- M11-VALIDATION-EXTENSION:END -->"
POINTER = ("> **Extension (M11, D27).** These verdicts are the pre-registered "
           "150-question ones and are *not* revised. Section 10 reports an "
           "independent replication on 350 **new** validation questions per "
           "dataset and says where the two disagree.\n")

RET_ORDER = ["vector", "ppr", "edge_sim", "gnn", "gnn_rrf", "ppr_rrf", "stacked"]
RET_LAB = {"vector": "Vector RAG", "ppr": "PPR", "edge_sim": "Edge-Sim",
           "gnn": "GNN router (graph-only)", "gnn_rrf": "GNN+Vector (RRF k=60)",
           "ppr_rrf": "PPR+Vector (RRF k=60)", "stacked": "Stacked combiner (D19)"}


def j(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def f3(v, dash="—"):
    return dash if v is None else f"{v:.3f}"


def f4(v, dash="—"):
    return dash if v is None else f"{v:.4f}"


def ci(d):
    return "—" if d is None else f"{d['point']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]"


def src(*names):
    return "*Source: " + ", ".join(f"`results/{n}`" for n in names) + "*"


def build() -> str:
    ext = j("ext_m7a_report.json")
    orig = j("m7a_report.json")
    pool = j("ext_pooled_report.json")
    xr = j("ext_extraction.json")
    samp = {d: j(f"ext_sample_{d}.json") for d in DS}
    L = [START, "", "## 10. Validation extension (M11)", ""]

    # ---------------- design -------------------------------------------
    L += [
        "Pre-registered as **D27** and committed before a single new validation "
        "question was read. The original 150-question results in Sections 2–9 "
        "stand unrevised; this section is an independent replication on questions "
        "no part of the project had seen, plus a secondary pooled analysis.",
        "",
        "The **primary, confirmatory** column throughout is the **new 350 alone**. "
        "The pooled 500 is secondary and is reported because it tightens the "
        "intervals, not because it settles a disagreement — where the two columns "
        "disagree, both are shown and the disagreement is stated.",
        "",
        "### 10.1 Sampling", "",
        src(*[f"ext_sample_{d}.json" for d in DS]), "",
        "| | eligible | available after exclusion | drawn | strata | sha256 of qid list |",
        "|---|---:|---:|---:|---|---|",
    ]
    for d in DS:
        s = samp[d]
        strata = (", ".join(f"{k}-hop {v}" for k, v in sorted(s["hop_distribution"].items()))
                  if d == "musique" else
                  ", ".join(f"{k} {v}" for k, v in sorted(s["type_distribution"].items())))
        L.append(f"| {d} | {s['eligible_total']:,} | {s['available_after_exclusion']:,} "
                 f"| {s['n']} | {strata} | `{s['sha256_of_qid_list'][:16]}…` |")
    L += ["",
          f"Seed **{samp['musique']['seed']}**, fixed in D27. Zero overlap with the "
          "original 150 and zero with the training questions, asserted at sampling "
          "time. Pool construction is the frozen, gold-blind one (MuSiQue top-14 of "
          "20 by bge cosine, HotpotQA all 10), and every pool came out exactly full.",
          "",
          "Of MuSiQue's 355 available 4-hop validation questions this draw takes 116, "
          "about a third of the stock, so a further extension has little 4-hop "
          "headroom left.", ""]

    # ---------------- extraction ---------------------------------------
    if xr:
        cov = xr["coverage"]
        L += ["### 10.2 Extraction on the new pools", "",
              src("ext_extraction.json"), "",
              "| | new paragraphs | t_p mean | JSON failures | gold coverage | pool coverage |",
              "|---|---:|---:|---:|---:|---:|"]
        L.append(f"| both datasets | {xr['paragraphs_extracted']:,} | "
                 f"{xr['t_p_mean_s']:.3f} s | {xr['json_failures']} "
                 f"({xr['json_failure_rate']:.4%}) | — | — |")
        for d in DS:
            c = cov[d]
            L.append(f"| {d} | {c['paragraph_slots']:,} slots | — | — | "
                     f"{c['extraction_coverage']:.4f} "
                     f"({c['gold_with_triples']}/{c['gold_paragraphs']}) | "
                     f"{c['paragraph_coverage']:.4f} |")
        L += ["",
              "Neither stop condition fired. Gold-paragraph coverage is **1.0000** on "
              "both datasets against 0.999 / 1.000 originally, and the JSON failure "
              "rate is **0.0000%** against 0.02%, so the new questions are not harder "
              "to extract from. Per-paragraph behaviour matches the original run "
              "closely (salvage 4.88% vs 5.35%, truncation 4.88% vs 5.39%, 423.6 vs "
              "426.4 LLM tokens, 5.52 vs 5.54 triples).",
              "",
              f"The run took **{xr['wall_h']:.2f} h** against the 3.08 h the step-2 "
              "timing check projected, because t_p came out at "
              f"{xr['t_p_mean_s']:.3f} s rather than 1.767 s: the machine held "
              "~1.2–1.9 GB of swap throughout. Still inside D27's 12 h gate. "
              f"**{len(xr['invalid_timing_chunks'])} of {len(xr['chunks'])} chunks are "
              "§7.3 timing-invalid**; the triples are unaffected, since decoding is "
              "greedy and each paragraph is extracted independently. The accounting for "
              "those chunks was lost to a retry-loop bug and rebuilt from the cache — "
              "see **D28**.", ""]

    # ---------------- measurement corrections --------------------------
    L += [
        "### 10.3 Three measurement faults found and corrected", "",
        "The extension's first scoring sweep was wrong twice over, in ways that "
        "produced entirely plausible numbers; a third fault hit the extraction "
        "accounting. All three are recorded because published figures changed, "
        "and because the second was only found by going looking for it after the "
        "first.", "",
        "**D29 — the router ran query-blind on HotpotQA.** `to_pyg` zero-filled any "
        "embedding-cache miss. For the *question* that is not a degraded feature but "
        "a silent amputation: `q = 0` zeroes `e_v * q`, `cos(e_v, q)` and "
        "`cos(r_e, q)`, so every query-conditioned feature vanishes. The M3 embedding "
        "pass had never been run for the new pools, and `build_graph` does not cache "
        "the question, so HotpotQA's 350 questions had no embedding at all. MuSiQue "
        "escaped only because its top-14-of-20 pool selection embeds the question "
        "anyway. Re-measured: HotpotQA GNN AllGold@4 0.571 → 0.583, stacked "
        "0.777 → 0.786, with MuSiQue and every non-graph method byte-identical.", "",
        "**D30 — 10% of edges were scored with no relation feature.** D29's fix made "
        "only the *question* lookup strict. An audit of the remaining kinds then "
        "measured relation cache misses of **10.2%** (MuSiQue) and **9.1%** "
        "(HotpotQA) on the new pools, against **0.0%** on the published splits: a "
        "missed relation leaves that edge's whole `edge_attr` signal-free. The "
        "trigger was a latency anomaly, not a metric one — the new questions' graphs "
        "are *smaller* (median 270 vs 380 edges on MuSiQue) yet assembly cost 2.4× "
        "more per edge, because uncached entity names were being encoded by bge-small "
        "inside the timed routing region. `_emb` now raises on a miss for every kind, "
        "and the embedding pass caches relation and entity texts. Re-measured, the "
        "effect on AllGold@4 is small but runs in both directions — MuSiQue GNN "
        "0.217 → 0.211, GNN+Vector 0.271 → 0.274; HotpotQA GNN+Vector 0.717 → "
        "0.723, stacked 0.786 → 0.780 — which is what a 10% feature corruption "
        "looks like when it is noise rather than bias. All GNN-family metrics and "
        "all routing timings in this section are from after the fix; the superseded "
        "runs are kept at `results/ext_cold_snapshot/` and "
        "`results/ext_pre_d30_snapshot/`.", "",
        "**D28 — the extraction chunk accounting was lost to a retry-loop bug** and "
        "rebuilt from the triple cache; the triples themselves were never in doubt.", "",
        "The common thread is that a zero vector is not a neutral default. Three "
        "times now (paragraph nodes, then the question, then relations) a cache miss "
        "has been silently absorbed into a feature and the model has gone on "
        "returning plausible scores. Nothing in the pipeline announced any of them.", "",
    ]

    # ---------------- comparability ------------------------------------
    if pool and pool.get("comparability"):
        L += ["### 10.4 The two question sets are not equally hard", "",
              src("ext_pooled_report.json"), "",
              "AllGold@4 is bounded above by whether the ego-graph reaches every gold "
              "paragraph, so if that bound differs between the two sets, absolute "
              "AllGold@4 cannot be compared across them. It does differ, and the "
              "interval is unpaired because the two sets are disjoint samples.", "",
              "| dataset | quantity | original 150 | new 350 | difference (95% CI) |",
              "|---|---|---:|---:|---|"]
        LABQ = {"allgold4_ceiling": "graph ceiling (AllGold@4)",
                "graph_pool_coverage": "graph pool coverage"}
        harder = []
        for d in DS:
            for k, x in pool["comparability"][d].items():
                L.append(f"| {d} | {LABQ.get(k, k)} | {x['mean_b']:.3f} | "
                         f"{x['mean_a']:.3f} | {ci(x)} |")
                if x.get("harder_on_new350"):
                    harder.append(f"{d} {LABQ.get(k, k)}")
        L += ["",
              "**MuSiQue's new draw is materially harder for every graph method**: the "
              "ceiling falls 0.380 → 0.257 and graph pool coverage 0.791 → 0.674, both "
              "intervals excluding zero. This is not a stratification artefact — both "
              "sets are uniform thirds by hop count, pool recall is unchanged at 0.897 "
              "(so the gold paragraphs are present), and the ceiling drops inside "
              "*every* hop bucket (2-hop 0.720 → 0.590, 3-hop 0.200 → 0.128, 4-hop "
              "0.220 → 0.052). HotpotQA is unchanged, as expected when the ceiling is "
              "already 0.98+.", "",
              "Two consequences, which govern how the rest of this section may be "
              "read. First, **only the within-set paired contrasts replicate across "
              "the two sets**; an absolute AllGold@4 from one set must not be compared "
              "to the other's. Second, the coverage bottleneck that Sections 5–8 "
              "identify as this project's central limitation is **worse on fresh "
              "questions than the original 150 suggested**, so that finding "
              "strengthens rather than softens on replication.", ""]

    # ---------------- retrieval ----------------------------------------
    if ext and orig:
        L += ["### 10.5 Retrieval, original 150 vs new 350", ""]
        for d in DS:
            eo = orig["datasets"][d]
            en = ext["datasets"][d]
            L += [f"**{d}** — AllGold@4 unconditional, with the graph ceiling.", "",
                  src("m7a_report.json", "ext_m7a_report.json"), "",
                  "| method | SP-Recall@4 (150) | SP-Recall@4 (350) | AllGold@4 (150) "
                  "| AllGold@4 (350) | Δ |", "|---|---:|---:|---:|---:|---:|"]
            for m in RET_ORDER:
                a = eo["methods"].get(m, {}).get("ALL")
                b = en["methods"].get(m, {}).get("ALL")
                if not a or not b:
                    continue
                dd = b["allgold4"] - a["allgold4"]
                L.append(f"| {RET_LAB[m]} | {f4(a['sp_recall4'])} | {f4(b['sp_recall4'])} "
                         f"| {f3(a['allgold4'])} | {f3(b['allgold4'])} | {dd:+.3f} |")
            L.append(f"| *graph ceiling* | — | — | {f3(eo['context']['ALL']['allgold4_ceiling'])} "
                     f"| {f3(en['context']['ALL']['allgold4_ceiling'])} | "
                     f"{en['context']['ALL']['allgold4_ceiling'] - eo['context']['ALL']['allgold4_ceiling']:+.3f} |")
            L += ["",
                  "The Δ column is **descriptive only and carries no interval**: by "
                  "§10.4 the two sets are not equally hard, so a cross-set Δ mixes a "
                  "method difference with a sample difference and cannot be read as a "
                  "contrast." + (" On MuSiQue the ceiling itself moved by −0.123, "
                  "which is larger than every method Δ in the table, so the per-method "
                  "numbers here say more about the draw than about the methods."
                  if d == "musique" else ""),
                  "",
                  "The replicating quantities are the **within-set paired contrasts** "
                  "in §10.7, each computed inside one question set.", ""]

    # ---------------- reader -------------------------------------------
    if pool and pool.get("reader"):
        RORD = ["vector", "ppr", "llm_trace", "gnn", "gnn_rrf", "stacked", "oracle"]
        RLAB = {**RET_LAB, "llm_trace": "LLM-Trace", "oracle": "Oracle (gold only)"}
        L += ["### 10.6 Reader answer quality", "",
              src("ext_pooled_report.json"), "",
              "EM/F1 from the frozen 1.5B reader over each method's top-4 paragraphs. "
              "Paired bootstrap on the new 350, F1.", ""]
        for d in DS:
            r = pool["reader"][d]
            L += [f"**{d}**", "",
                  "| method | EM (150) | F1 (150) | EM (350) | F1 (350) | F1 vs Vector RAG (350) |",
                  "|---|---:|---:|---:|---:|---|"]
            for m in RORD:
                a = r["table"]["orig150"].get(m)
                b = r["table"]["new350"].get(m)
                if not b:
                    continue
                vs = r["paired_bootstrap"]["new350"].get(f"f1|{m}-vs-vector")
                L.append(f"| {RLAB.get(m, m)} | {f4(a['em']) if a else '—'} "
                         f"| {f4(a['f1']) if a else '—'} | {f4(b['em'])} | {f4(b['f1'])} "
                         f"| {ci(vs) if vs else '—'} |")
            L.append("")
        L += ["Both datasets reproduce the original ordering: no graph method's F1 "
              "beats Vector RAG, the GNN+Vector and stacked hybrids are "
              "indistinguishable from it, and the oracle gap is large on both "
              "(MuSiQue 0.442 F1 vs 0.236 for Vector RAG), so retrieval, not the "
              "reader, is the binding constraint. On HotpotQA the GNN router does "
              "beat LLM-Trace on F1 (+0.0579 [+0.0139, +0.1019]), matching C3's "
              "quality half there.", ""]

    # ---------------- verdicts -----------------------------------------
    if pool:
        L += ["### 10.7 Verdicts on three question sets", "",
              src("ext_pooled_report.json"), "",
              "The primary column is **new 350**. A row marked "
              "*disagreement* has a verdict that is not the same across all three.", ""]
        for key, cl in pool["claims"].items():
            if key in ("C3c", "C4") or "datasets" not in cl:
                continue
            L += [f"**{key}** — {cl['label']}", "",
                  "| dataset | set | n | difference (95% CI) | verdict |",
                  "|---|---|---:|---|---|"]
            for d in DS:
                per = cl["datasets"][d]
                for s in SETS:
                    x = per.get(s)
                    if not x:
                        continue
                    L.append(f"| {d} | {SETLAB[s]} | {x['n']} | {ci(x)} | {x['verdict']} |")
                if not per.get("agrees", True):
                    L.append(f"| {d} | *disagreement* | | | {per['verdicts']} |")
            L.append("")
        c3c, c4 = pool["claims"].get("C3c"), pool["claims"].get("C4")
        if c3c:
            L += ["**C3, cost half** — LLM-Trace median latency over GNN routing median.", "",
                  "| dataset | set | GNN routing | LLM-Trace | ratio | verdict |",
                  "|---|---|---:|---:|---:|---|"]
            for d in DS:
                for s, x in c3c["datasets"][d].items():
                    if not x:
                        continue
                    flag = " ⚠" if (x["trace_stage_swap_invalid"]
                                    or x["gnn_stage_swap_invalid"]) else ""
                    L.append(f"| {d} | {SETLAB.get(s, s)} | {x['gnn_median_ms']:.1f} ms "
                             f"| {x['trace_median_ms']:.0f} ms | {x['ratio']:.1f}× "
                             f"| {x['verdict']}{flag} |")
            L += ["", "⚠ = at least one of the two stages is §7.3-invalid on swap, so "
                  "the ratio is indicative.", "",
                  "**The GNN side of this ratio is not comparable to Section 7's, "
                  "and Section 7's is the inflated one.** Counting bge-small calls "
                  "per graph showed the published validation split was doing 8.2 "
                  "(MuSiQue) and 6.5 (HotpotQA) encodes *inside* the timed region, "
                  "because the M3/M5 passes never persisted every text "
                  "`build_graph` produces; the extension, after D30, does zero. With "
                  "the graph texts of both splits cached, GNN routing medians are "
                  "**29.0 / 20.9 ms** on the published 150 and **26.5 / 19.9 ms** on "
                  "the new 350 — the two question sets agree, and the extension is "
                  "not faster than the original. Section 7's 43.8 / 35.6 ms carry "
                  "the artefact; they are left as published and corrected here "
                  "rather than restated there. The published 150's metrics "
                  "reproduced exactly across all three timings, which is what makes "
                  "this a measurement fix and not a rerun (D30, same fault class as "
                  "D23's Edge-Sim 583.5 → 10.6 ms). The ratio's numerator is "
                  "§7.3-invalid on swap in every run (D24), so read it as an order "
                  "of magnitude: the margin is ~36–180× against a threshold of "
                  "10.", ""]
        if c4:
            L += [f"**C4** — peak process-tree RSS over the extension's stages: "
                  f"**{c4['peak_rss_mb']:.0f} MB** "
                  f"({c4['peak_rss_mb']/1024:.2f} GB) → **{c4['verdict']}** against the "
                  "6 GB budget. The same caveat as Section 7 applies: RSS under-reports "
                  "Ollama's GPU-wired memory.", ""]

    # ---------------- multiplicity -------------------------------------
    if pool:
        # Count every interval the extension actually computed, not just the
        # claim table's: under-counting multiplicity is the direction that
        # flatters, so the reader table, the comparability intervals and the
        # aggregation's own bootstraps are all in here. orig150 columns are
        # recomputations of already-published comparisons and are counted
        # separately rather than folded in.
        n_claim = sum(1 for cl in pool["claims"].values() if "datasets" in cl
                      for d in DS for s in ("new350", "pooled500")
                      if isinstance(cl["datasets"].get(d), dict) and cl["datasets"][d].get(s))
        n_recomp = sum(1 for cl in pool["claims"].values() if "datasets" in cl
                       for d in DS
                       if isinstance(cl["datasets"].get(d), dict)
                       and cl["datasets"][d].get("orig150"))
        n_reader = sum(1 for d in DS
                       for k, v in pool.get("reader", {}).get(d, {})
                       .get("paired_bootstrap", {}).get("new350", {}).items() if v)
        n_comp = sum(len(v) for v in pool.get("comparability", {}).values())
        n_agg = sum(len(ext["datasets"][d].get("paired_bootstrap", {})) for d in DS) if ext else 0
        n_new = n_claim + n_reader + n_comp + n_agg
        L += ["### 10.8 Multiplicity and stage validity", "",
              f"The extension computes **{n_new}** new confidence intervals: "
              f"{n_claim} in the claim table (new 350 and pooled 500), {n_agg} in the "
              f"aggregation's own method contrasts, {n_reader} on reader F1/EM and "
              f"{n_comp} on set comparability. A further {n_recomp} recompute "
              "already-published original-150 comparisons through this code path and "
              "are not new evidence.", "",
              "They remain **uncorrected**, as everywhere else in this report. At this "
              f"count, roughly {round(n_new * 0.05)} intervals would be expected to "
              "exclude zero by chance alone, so **no single borderline interval here "
              "is evidence of anything**.", ""]
        # Name the disagreements from the data rather than by hand, and say how
        # far each flipped interval sits from its threshold, so the reader can
        # judge whether a verdict change is a real one or a coin landing.
        flips = [(k, d) for k, cl in pool["claims"].items() if "datasets" in cl
                 for d in DS
                 if isinstance(cl["datasets"].get(d), dict)
                 and not cl["datasets"][d].get("agrees", True)]
        if flips:
            # C1a/C1a_uncond and C2/C2_cond are the conditional and unconditional
            # readings of one comparison, so the row count overstates how many
            # independent things changed.
            distinct = len({(k.replace("_cond", "").replace("_uncond", ""), d)
                            for k, d in flips})
            L += [f"The verdict changes in §10.7 are not borderline. They are "
                  f"**{distinct}** distinct comparisons ({len(flips)} rows below, "
                  "since two appear in both their conditional and unconditional "
                  "readings). Each flipped interval and its distance from the "
                  "decision threshold:", "",
                  "| claim | dataset | original 150 | new 350 | new-350 interval vs threshold |",
                  "|---|---|---|---|---|"]
            for k, d in flips:
                per = pool["claims"][k]["datasets"][d]
                n = per["new350"]
                thr = "−0.03" if k == "C3q" else "0"
                edge = n["hi"] if n["point"] < 0 else n["lo"]
                ref = -0.03 if k == "C3q" else 0.0
                L.append(f"| {k} | {d} | {per['verdicts']['orig150']} | "
                         f"**{per['verdicts']['new350']}** | "
                         f"{ci(n)}, nearer bound {edge:+.4f} vs {thr} "
                         f"({abs(edge - ref):.4f} clear) |")
            L += ["",
                  "Every one is a case where the original 150 could not resolve the "
                  "comparison and 350 questions could — the direction of the point "
                  "estimate did not reverse in any of them.", ""]
        L += [
              "**Stage validity under §7.3.** The extension's own measured stages:", "",
              "| stage group | measured | §7.3-invalid on swap |",
              "|---|---:|---:|",
              "| retrieval routing (§10.5, §10.7) | 18 | **0** |",
              "| extraction chunks (§10.2) | 13 | 12 |",
              "| LLM-Trace routing | 2 | 2 |",
              "| reader (§10.6) | 14 | 8 |", "",
              "Retrieval is fully valid this time, which it was not before D30 — "
              "removing bge-small from the timed region cut the sweep from 3,462 s to "
              "768 s and with it the memory pressure. The 22 invalid stages are all "
              "Ollama-loaded or extraction stages, the same pattern as D26, and they "
              "void **only** latency, token-rate and RSS figures: every EM/F1 and "
              "retrieval metric is unaffected, because the reader and the traversal "
              "decode greedily at `temperature=0`, so identical context gives "
              "identical output however the machine was paging.", ""]
    L += [END]
    return "\n".join(L)


def main() -> int:
    section = build()
    (RES / "ext_report_section.md").write_text(section + "\n")
    rp = ROOT / "report.md"
    text = rp.read_text()

    replaced = START in text and END in text
    if replaced:
        a, b = text.index(START), text.index(END) + len(END)
        text = text[:a] + section + text[b:]
    else:
        # Section 10 belongs after Section 9 and BEFORE the appendix, not at the
        # end of the file.
        appendix = "\n# Appendix\n"
        if appendix in text:
            i = text.index(appendix)
            text = text[:i] + "\n" + section + "\n" + text[i:]
        else:
            text = text.rstrip("\n") + "\n\n" + section + "\n"

    if POINTER not in text:
        anchor = "## 2. Pre-registered claims and verdicts\n\n"
        assert anchor in text, "Section 2 heading not found; pointer not inserted"
        text = text.replace(anchor, anchor + POINTER + "\n", 1)

    rp.write_text(text)
    print(f"-> results/ext_report_section.md ({len(section.splitlines())} lines)")
    print(f"-> report.md section 10 {'replaced' if replaced else 'inserted'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
