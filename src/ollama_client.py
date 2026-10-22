"""Thin Ollama HTTP wrapper with greedy decoding and token accounting.

agent.md Section 2: greedy decoding (temperature=0), num_ctx=2048 per request.
Section 6.1: the model must be unloaded (`ollama stop <tag>`) before any GNN phase.
"""

from __future__ import annotations

import os
import subprocess
import time

import requests

from .schemas import Usage


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(
        self,
        model_tag: str,
        host: str | None = None,
        num_ctx: int = 2048,
        temperature: float = 0.0,
        keep_alive: str | None = None,
        timeout_s: float = 300.0,
    ):
        self.model_tag = model_tag
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.keep_alive = keep_alive or os.environ.get("OLLAMA_KEEP_ALIVE", "30m")
        self.timeout_s = timeout_s
        self._session = requests.Session()

    # -- availability -------------------------------------------------------
    def version(self) -> str:
        r = self._session.get(f"{self.host}/api/version", timeout=5)
        r.raise_for_status()
        return r.json().get("version", "?")

    def list_models(self) -> list[str]:
        r = self._session.get(f"{self.host}/api/tags", timeout=10)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    def assert_model_present(self) -> None:
        names = self.list_models()
        if self.model_tag not in names:
            raise OllamaError(
                f"model tag {self.model_tag!r} not found in `ollama list`: {names}"
            )

    # -- generation ---------------------------------------------------------
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        fmt: str | None = None,
        num_predict: int | None = None,
        stop: list[str] | None = None,
    ) -> tuple[str, Usage, dict]:
        """Greedy single-turn generation. Returns (text, Usage, raw_response)."""
        options = {
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
            "top_p": 1.0,
            "seed": int(os.environ.get("SEED", "1234")),
        }
        if num_predict is not None:
            options["num_predict"] = num_predict
        if stop:
            options["stop"] = stop

        payload = {
            "model": self.model_tag,
            "prompt": prompt,
            "stream": False,
            "options": options,
            "keep_alive": self.keep_alive,
        }
        if system is not None:
            payload["system"] = system
        if fmt is not None:
            payload["format"] = fmt

        r = self._session.post(f"{self.host}/api/generate", json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        usage = Usage(
            prompt_tokens=int(data.get("prompt_eval_count", 0) or 0),
            completion_tokens=int(data.get("eval_count", 0) or 0),
        )
        return data.get("response", ""), usage, data

    # -- lifecycle ----------------------------------------------------------
    def unload(self) -> bool:
        """`ollama stop <tag>` -- call before any GNN training phase (Section 6.1)."""
        try:
            subprocess.run(
                ["ollama", "stop", self.model_tag],
                check=False, capture_output=True, timeout=60,
            )
            return True
        except (OSError, subprocess.TimeoutExpired):
            return False

    def loaded_models(self) -> list[dict]:
        r = self._session.get(f"{self.host}/api/ps", timeout=10)
        r.raise_for_status()
        return r.json().get("models", [])


def unload_all(timeout_s: float = 30.0) -> list[str]:
    """Stop every loaded Ollama model. Returns the tags that were unloaded."""
    host = (os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
    try:
        r = requests.get(f"{host}/api/ps", timeout=10)
        r.raise_for_status()
        tags = [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []
    for t in tags:
        subprocess.run(["ollama", "stop", t], check=False, capture_output=True, timeout=60)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = requests.get(f"{host}/api/ps", timeout=5)
            if not r.json().get("models"):
                break
        except Exception:
            break
        time.sleep(1.0)
    return tags
