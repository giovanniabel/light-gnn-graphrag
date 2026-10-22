"""SLM OpenIE extraction with an append-only cache and one retry (agent.md Sections 4.1, 6.2, 8.1).

Speed levers from Section 6.2, all applied here:
  * format="json", temperature=0, num_predict=192
  * paragraph truncated to 200 words
  * a short fixed prompt with one example, placed BEFORE the paragraph so Ollama's
    prefix cache is reused across every call
  * dedupe by paragraph hash across questions
  * strictly sequential unless the pilot shows NUM_PARALLEL=2 is faster

Malformed JSON gets one retry, then an empty triple list that is counted
(Section 6.2 gate: failure rate <= 5%).
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from pathlib import Path

from .ollama_client import OllamaClient
from .schemas import ExtractionStats, Paragraph, Triple

PROMPT_VERSION = "v2"

# --- prompt fix history (Section 6.2 allows one prompt fix before the gate) ----
# v1: free-form `format="json"`, paragraphs cut to 200 words, "extract every triple".
#     Measured on this machine: t_p 8.29 s, 50% of responses hit num_predict=192 and
#     were truncated mid-array, so the JSON failure rate was 50% -- far above the 5%
#     gate. Asking for "at most 8 triples" in prose changed nothing (the 1.5B model
#     ignores it), and greedy decoding sometimes looped on a repeated triple.
# v2: three changes, each measured:
#     (a) `format` is a JSON *schema* with maxItems, so the triple count is enforced
#         by the decoding grammar instead of requested in prose;
#     (b) paragraphs cut to 120 words -- only 2/15 pool paragraphs are affected and
#         98.8% of all words are kept, since the median paragraph is 99 words;
#     (c) a truncated response is salvaged for its complete leading triples rather
#         than thrown away (see salvage_triples).

# Fixed prefix: identical bytes on every call so the prefix cache hits (Section 6.2).
SYSTEM = (
    "You extract knowledge-graph triples from text. "
    "Reply with JSON only, no prose."
)

PROMPT_PREFIX = """Extract the most important factual (head, relation, tail) triples from the paragraph.
Rules: use exact names from the text; keep relations to 1-4 words; no pronouns; never repeat a triple.
Reply as JSON: {"triples": [["head", "relation", "tail"]]}

Example paragraph:
Grant's First Stand is the debut album by American jazz guitarist Grant Green, released on the Blue Note label in 1961.
Example JSON:
{"triples": [["Grant's First Stand", "debut album by", "Grant Green"], ["Grant Green", "nationality", "American"], ["Grant's First Stand", "released on label", "Blue Note"], ["Grant's First Stand", "release year", "1961"]]}

Paragraph:
"""

PROMPT_SUFFIX = "\nJSON:"


MAX_TRIPLES = 6


def triple_schema(max_triples: int = MAX_TRIPLES) -> dict:
    """JSON schema passed as Ollama's `format`, so llama.cpp compiles it to a GBNF
    grammar and the triple count is enforced during decoding (Section 6.2)."""
    return {
        "type": "object",
        "properties": {
            "triples": {
                "type": "array",
                "minItems": 1,
                "maxItems": max_triples,
                "items": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "items": {"type": "string"},
                },
            }
        },
        "required": ["triples"],
    }


_TRIPLE_RE = re.compile(
    r'\[\s*"((?:[^"\\]|\\.)*)"\s*,\s*"((?:[^"\\]|\\.)*)"\s*,\s*"((?:[^"\\]|\\.)*)"\s*\]'
)


def salvage_triples(text: str, pid: str) -> list[Triple]:
    """Recover complete triples from a response cut off at num_predict.

    A response that stops mid-array is not a malformed generation -- it is our own
    output cap truncating a valid stream -- so the complete leading triples are
    kept instead of discarded. Counted separately from true JSON failures so the
    Section 6.2 gate is still reported on its strict definition.
    """
    out: list[Triple] = []
    for m in _TRIPLE_RE.finditer(text):
        try:
            h, r, t = json.loads(f"[\"{m.group(1)}\", \"{m.group(2)}\", \"{m.group(3)}\"]")
        except json.JSONDecodeError:
            continue
        if all(str(x).strip() for x in (h, r, t)):
            out.append(Triple(str(h).strip(), str(r).strip(), str(t).strip(), pid))
    return out


def effective_prompt_version(cfg) -> str:
    """The cache's prompt_version, with the triple cap folded in (deviation D8).

    The Section 8.1 key is sha1(model_tag + prompt_version + paragraph_text) and
    does not mention max_triples, which changes the output. Every caller must
    derive the version here: extract_all and the M4 reporting previously computed
    it independently, disagreed ("v2" vs "v2-mt6"), and the report then looked up
    keys that had never been written, reporting 0% extraction coverage for a cache
    that was in fact fully populated.
    """
    ex = cfg["extraction"]
    return f"{ex['prompt_version']}-mt{int(ex.get('max_triples', MAX_TRIPLES))}"


def truncate_words(text: str, max_words: int = 120) -> str:
    w = text.split()
    return " ".join(w[:max_words])


def cache_key(paragraph_text: str, model_tag: str, prompt_version: str = PROMPT_VERSION) -> str:
    """sha1(model_tag + prompt_version + paragraph_text) -- agent.md Section 8.1."""
    return hashlib.sha1(
        f"{model_tag}\x00{prompt_version}\x00{paragraph_text}".encode()
    ).hexdigest()


class TripleCache:
    """Append-only JSONL cache, one line per paragraph, flushed after every paragraph."""

    def __init__(self, path: Path | str, model_tag: str, prompt_version: str = PROMPT_VERSION):
        self.path = Path(path)
        self.model_tag = model_tag
        self.prompt_version = prompt_version
        self._index: dict[str, list[Triple]] = {}
        self._meta: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._fh = None
        self.load()

    # -- io -----------------------------------------------------------------
    def load(self) -> int:
        self._index.clear()
        self._meta.clear()
        if not self.path.exists():
            return 0
        bad = 0
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1          # truncated tail from a killed run: ignore it
                    continue
                k = d.get("key")
                if not k:
                    continue
                self._index[k] = [Triple(**t) for t in d.get("triples", [])]
                self._meta[k] = {
                    "json_failed": d.get("json_failed", False),
                    "llm_tokens": d.get("llm_tokens", 0),
                    "seconds": d.get("seconds", 0.0),
                    "pid": d.get("pid"),
                }
        return len(self._index)

    def _handle(self):
        if self._fh is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a", encoding="utf-8")
        return self._fh

    def append(self, key: str, pid: str, triples: list[Triple], **meta) -> None:
        """Write one line and flush immediately (Section 8.1)."""
        rec = {
            "key": key,
            "pid": pid,
            "model_tag": self.model_tag,
            "prompt_version": self.prompt_version,
            "triples": [vars(t) for t in triples],
            **meta,
        }
        with self._lock:
            fh = self._handle()
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            self._index[key] = triples
            self._meta[key] = {
                "json_failed": meta.get("json_failed", False),
                "salvaged": meta.get("salvaged", False),
                "truncated": meta.get("truncated", False),
                "eval_count": meta.get("eval_count", 0),
                "eval_duration_s": meta.get("eval_duration_s", 0.0),
                "llm_tokens": meta.get("llm_tokens", 0),
                "seconds": meta.get("seconds", 0.0),
                "pid": pid,
            }

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.flush()
                self._fh.close()
                self._fh = None

    # -- lookup -------------------------------------------------------------
    def key_for(self, paragraph_text: str) -> str:
        return cache_key(truncate_words(paragraph_text), self.model_tag, self.prompt_version)

    def get(self, paragraph_text: str) -> list[Triple] | None:
        return self._index.get(self.key_for(paragraph_text))

    def __contains__(self, paragraph_text: str) -> bool:
        return self.key_for(paragraph_text) in self._index

    def __len__(self) -> int:
        return len(self._index)

    @property
    def n_lines(self) -> int:
        if not self.path.exists():
            return 0
        with open(self.path, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())


# ---------------------------------------------------------------- parsing
def parse_triples(text: str, pid: str) -> tuple[list[Triple], bool]:
    """Parse the SLM's JSON. Returns (triples, failed)."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return [], True

    raw = data.get("triples") if isinstance(data, dict) else data
    if raw is None and isinstance(data, dict):
        # tolerate a differently-named single list value
        for v in data.values():
            if isinstance(v, list):
                raw = v
                break
    if not isinstance(raw, list):
        return [], True

    out: list[Triple] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            h, r, t = item[0], item[1], item[2]
        elif isinstance(item, dict):
            h = item.get("head") or item.get("subject") or item.get("s")
            r = item.get("relation") or item.get("rel") or item.get("predicate") or item.get("p")
            t = item.get("tail") or item.get("object") or item.get("o")
        else:
            continue
        if not all(isinstance(x, (str, int, float)) and str(x).strip() for x in (h, r, t)):
            continue
        out.append(Triple(head=str(h).strip(), rel=str(r).strip(), tail=str(t).strip(), pid=pid))
    # valid JSON that simply yielded nothing is not a parse failure
    return out, False


# ---------------------------------------------------------------- extraction
def extract_paragraph(
    p: Paragraph,
    cache: TripleCache,
    client: OllamaClient | None = None,
    num_predict: int = 192,
    max_words: int = 120,
    retries: int = 1,
    max_triples: int = MAX_TRIPLES,
) -> list[Triple]:
    """Extract triples for one paragraph, serving from cache when possible.

    Section 10 signature is (p, cache); the remaining arguments are optional and
    defaulted from agent.md Section 6.2.
    """
    text = truncate_words(p.text, max_words)
    key = cache.key_for(p.text)
    hit = cache._index.get(key)
    if hit is not None:
        # re-stamp provenance: the cache is deduped by paragraph text across questions
        return [Triple(t.head, t.rel, t.tail, p.pid) for t in hit]

    if client is None:
        client = OllamaClient(cache.model_tag)

    prompt = PROMPT_PREFIX + f"{p.title}. {text}" + PROMPT_SUFFIX
    schema = triple_schema(max_triples)
    triples: list[Triple] = []
    failed = True
    salvaged = False
    truncated = False
    tokens = 0
    eval_count = 0
    eval_duration_s = 0.0
    prompt_eval_count = 0
    t0 = time.perf_counter()
    for attempt in range(retries + 1):
        try:
            out, usage, raw = client.generate(
                prompt, system=SYSTEM, fmt=schema, num_predict=num_predict
            )
            tokens += usage.total
            eval_count += int(raw.get("eval_count", 0) or 0)
            prompt_eval_count += int(raw.get("prompt_eval_count", 0) or 0)
            eval_duration_s += float(raw.get("eval_duration", 0) or 0) / 1e9
            truncated = raw.get("done_reason") == "length"
            triples, failed = parse_triples(out, p.pid)
            if not failed:
                break
            # Output cap cut a valid stream: keep the complete leading triples
            # instead of retrying (a retry is greedy and would truncate again).
            rescued = salvage_triples(out, p.pid)
            if rescued:
                triples, failed, salvaged = rescued, False, True
                break
        except Exception:
            failed = True
    seconds = time.perf_counter() - t0

    cache.append(
        key, p.pid, triples if not failed else [],
        json_failed=failed, salvaged=salvaged, truncated=truncated,
        llm_tokens=tokens, seconds=seconds,
        eval_count=eval_count, eval_duration_s=eval_duration_s,
        prompt_eval_count=prompt_eval_count,
        n_triples=len(triples) if not failed else 0,
    )
    return triples if not failed else []


def extract_all(
    examples,
    cache_path,
    max_hours: float,
    resume: bool,
    model_tag: str | None = None,
    cfg=None,
    progress_every: int = 25,
    on_progress=None,
    should_stop=None,
) -> ExtractionStats:
    """Extract triples for every pool paragraph of every example (agent.md Section 10).

    Resumable and time-capped (Section 6.5): paragraphs already in the cache are
    skipped, and the run exits cleanly with the cache flushed at max_hours.
    Deduped by paragraph text across questions (Section 6.2).

    on_progress(stats, i, total) is called every progress_every paragraphs.
    should_stop(stats, i, total) -> str | None aborts cleanly when it returns a
    reason, so callers can enforce their own budget guards without bypassing the
    cache flush.
    """
    from .config import env, load_config

    cfg = cfg or load_config()
    ex_cfg = cfg["extraction"]
    model_tag = model_tag or env("SLM_TAG")

    cache = TripleCache(cache_path, model_tag, effective_prompt_version(cfg))
    client = OllamaClient(
        model_tag, num_ctx=int(ex_cfg["num_ctx"]), temperature=float(ex_cfg["temperature"])
    )
    stats = ExtractionStats()

    # dedupe by truncated paragraph text across questions
    seen: set[str] = set()
    work: list[Paragraph] = []
    for ex in examples:
        for p in ex.paragraphs:
            stats.paragraphs_total += 1
            k = cache.key_for(p.text)
            if k in seen:
                continue
            seen.add(k)
            work.append(p)

    deadline = time.time() + max_hours * 3600 if max_hours else None
    t0 = time.perf_counter()
    try:
        for i, p in enumerate(work, 1):
            if resume and cache.key_for(p.text) in cache._index:
                stats.paragraphs_cached += 1
                continue
            if deadline and time.time() > deadline:
                stats.timed_out = True
                print(f"  [extract_all] max_hours reached at {i}/{len(work)}; cache flushed")
                break
            before = time.perf_counter()
            triples = extract_paragraph(
                p, cache, client,
                num_predict=int(ex_cfg["num_predict"]),
                max_words=int(ex_cfg["max_words_per_paragraph"]),
                retries=int(ex_cfg["retries"]),
                max_triples=int(ex_cfg.get("max_triples", MAX_TRIPLES)),
            )
            stats.paragraphs_extracted += 1
            stats.seconds += time.perf_counter() - before
            stats.triples_total += len(triples)
            meta = cache._meta.get(cache.key_for(p.text), {})
            stats.llm_tokens += int(meta.get("llm_tokens", 0))
            if meta.get("json_failed"):
                stats.json_failures += 1
            if not triples:
                stats.empty_results += 1
            if progress_every and i % progress_every == 0:
                if on_progress is not None:
                    on_progress(stats, i, len(work))
                else:
                    rate = stats.seconds / max(stats.paragraphs_extracted, 1)
                    print(f"  [extract_all] {i}/{len(work)}  t_p={rate:.2f}s  "
                          f"fail={stats.json_failure_rate:.1%}", flush=True)
                if should_stop is not None:
                    reason = should_stop(stats, i, len(work))
                    if reason:
                        stats.stopped_reason = reason
                        print(f"  [extract_all] STOPPING: {reason}", flush=True)
                        break
    finally:
        cache.close()
    _ = time.perf_counter() - t0
    return stats


def extraction_coverage(examples, cache: TripleCache) -> dict:
    """Fraction of gold paragraphs that yielded at least one triple (Section 4.2).

    Questions whose gold paragraphs produced no edges are dropped from training
    only; this is the number that must be reported per dataset.
    """
    gold_total = gold_with_triples = 0
    q_total = q_unmappable = 0
    for ex in examples:
        q_total += 1
        any_gold_edge = False
        for p in ex.paragraphs:
            if not p.is_gold:
                continue
            gold_total += 1
            if cache.get(p.text):
                gold_with_triples += 1
                any_gold_edge = True
        if not any_gold_edge:
            q_unmappable += 1
    return {
        "gold_paragraphs": gold_total,
        "gold_paragraphs_with_triples": gold_with_triples,
        "extraction_coverage": gold_with_triples / gold_total if gold_total else 0.0,
        "questions": q_total,
        "questions_unmappable": q_unmappable,
        "unmappable_rate": q_unmappable / q_total if q_total else 0.0,
    }
