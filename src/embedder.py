"""Frozen bge-small-en-v1.5 wrapper with a disk cache (agent.md Sections 2, 8.1).

The encoder is loaded only in embedding phases and freed afterwards (Section 2);
`free()` drops the model so later phases hold cached tensors only.

Cache layout (Section 8.1 "keyed by text hash"): the key is sha1(model_tag + text),
but entries are sharded into `embeddings/<kind>/<key[:2]>.pt` dicts rather than one
file per text -- same keying, ~256 files instead of ~100k inodes.
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import numpy as np
import torch

KINDS = ("questions", "entities", "relations", "paragraphs", "misc")


def text_key(text: str, model_tag: str) -> str:
    return hashlib.sha1(f"{model_tag}\x00{text}".encode()).hexdigest()


class EmbeddingCache:
    """Sharded on-disk store of normalized float32 embeddings."""

    def __init__(self, cache_dir: Path | str, model_tag: str, dim: int = 384):
        self.dir = Path(cache_dir)
        self.model_tag = model_tag
        self.dim = dim
        self._shards: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
        self._dirty: set[tuple[str, str]] = set()
        self._lock = threading.Lock()

    def _shard_path(self, kind: str, prefix: str) -> Path:
        return self.dir / kind / f"{prefix}.pt"

    def _shard(self, kind: str, prefix: str) -> dict[str, torch.Tensor]:
        key = (kind, prefix)
        if key not in self._shards:
            p = self._shard_path(kind, prefix)
            if p.exists():
                self._shards[key] = torch.load(p, map_location="cpu")
            else:
                self._shards[key] = {}
        return self._shards[key]

    def get(self, text: str, kind: str = "misc") -> torch.Tensor | None:
        k = text_key(text, self.model_tag)
        return self._shard(kind, k[:2]).get(k)

    def put(self, text: str, vec: torch.Tensor, kind: str = "misc") -> None:
        k = text_key(text, self.model_tag)
        with self._lock:
            # .clone() is load-bearing: `vec` is normally one row of an encode()
            # batch, i.e. a VIEW whose underlying storage is the whole [B, 384]
            # tensor. torch.save serializes the entire storage behind a view, so
            # storing the view wrote ~4096x more bytes than the row needs and grew
            # this cache to 59 GB instead of ~200 MB. clone() detaches the row
            # onto its own storage.
            self._shards[(kind, k[:2])] = self._shard(kind, k[:2])
            self._shards[(kind, k[:2])][k] = vec.detach().to(torch.float32).cpu().clone()
            self._dirty.add((kind, k[:2]))

    def flush(self) -> int:
        with self._lock:
            n = 0
            for kind, prefix in sorted(self._dirty):
                p = self._shard_path(kind, prefix)
                p.parent.mkdir(parents=True, exist_ok=True)
                tmp = p.with_suffix(".pt.tmp")
                torch.save(self._shards[(kind, prefix)], tmp)
                tmp.replace(p)
                n += 1
            self._dirty.clear()
            return n

    def stats(self) -> dict:
        counts = {}
        for kind in KINDS:
            d = self.dir / kind
            counts[kind] = sum(
                len(torch.load(p, map_location="cpu")) for p in sorted(d.glob("*.pt"))
            ) if d.exists() else 0
        return counts


class Embedder:
    """Frozen, normalized, 384-dim. Never fine-tuned (agent.md Section 2)."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "cpu",
        cache: EmbeddingCache | None = None,
        cache_dir: Path | str | None = None,
        batch_size: int = 64,
        query_instruction: str | None = None,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        # bge's documented asymmetric-retrieval prefix. agent.md Section 4.3 uses one
        # shared question embedding q everywhere, so this defaults to None (off).
        # Turning it on is a pre-eval config change and must be logged as a deviation.
        self.query_instruction = query_instruction
        self.dim = 384
        self._model = None
        self.cache = cache or (
            EmbeddingCache(cache_dir, model_name) if cache_dir is not None else None
        )
        self.n_encoded = 0
        self.n_cache_hits = 0

    # -- model lifecycle ----------------------------------------------------
    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._model.eval()
            for p in self._model.parameters():
                p.requires_grad_(False)
        return self._model

    def free(self) -> None:
        """Drop the encoder from memory (Section 2: loaded only in embedding phases)."""
        if self._model is not None:
            del self._model
            self._model = None
        import gc

        gc.collect()
        if torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    def __enter__(self) -> Embedder:
        return self

    def __exit__(self, *exc) -> bool:
        if self.cache is not None:
            self.cache.flush()
        self.free()
        return False

    # -- encoding -----------------------------------------------------------
    def _encode_raw(self, texts: list[str]) -> torch.Tensor:
        with torch.no_grad():
            v = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        self.n_encoded += len(texts)
        return torch.from_numpy(np.asarray(v, dtype=np.float32))

    def encode(self, texts: list[str] | str, kind: str = "misc") -> torch.Tensor:
        """Normalized embeddings [N, 384], served from the disk cache where possible."""
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        if not items:
            return torch.zeros(0, self.dim)

        out: list[torch.Tensor | None] = [None] * len(items)
        todo: dict[str, list[int]] = {}
        for i, t in enumerate(items):
            if self.cache is not None:
                hit = self.cache.get(t, kind)
                if hit is not None:
                    out[i] = hit
                    self.n_cache_hits += 1
                    continue
            todo.setdefault(t, []).append(i)

        if todo:
            uniq = list(todo)
            vecs = self._encode_raw(uniq)
            for t, v in zip(uniq, vecs):
                if self.cache is not None:
                    self.cache.put(t, v, kind)
                for i in todo[t]:
                    out[i] = v

        res = torch.stack([o for o in out])  # type: ignore[arg-type]
        return res[0] if single else res

    def encode_query(self, question: str | list[str]) -> torch.Tensor:
        """Question embedding. Applies query_instruction only if configured."""
        qs = [question] if isinstance(question, str) else list(question)
        if self.query_instruction:
            texts = [self.query_instruction + q for q in qs]
        else:
            texts = qs
        v = self.encode(texts, kind="questions")
        return v[0] if isinstance(question, str) else v

    @staticmethod
    def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Cosine similarity for already-normalized vectors (= dot product)."""
        a = a if a.dim() > 1 else a.unsqueeze(0)
        b = b if b.dim() > 1 else b.unsqueeze(0)
        a = torch.nn.functional.normalize(a, dim=-1)
        b = torch.nn.functional.normalize(b, dim=-1)
        return a @ b.T
