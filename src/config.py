"""Config + environment loading. Import this BEFORE torch anywhere in the project:
it sets PYTORCH_ENABLE_MPS_FALLBACK from .env (agent.md Section 2).
"""

from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"

# Load .env first so PYTORCH_ENABLE_MPS_FALLBACK is in os.environ before torch import.
load_dotenv(ROOT / ".env")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


class Config(dict):
    """Dict with attribute access on nested dicts, loaded from configs/default.yaml."""

    def __getattr__(self, name: str) -> Any:
        try:
            v = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return Config(v) if isinstance(v, dict) else v

    def path(self, key: str) -> Path:
        return ROOT / self["paths"][key]


def load_config(path: Path | str = DEFAULT_CONFIG) -> Config:
    with open(path) as f:
        return Config(yaml.safe_load(f))


def save_config(cfg: dict, path: Path | str = DEFAULT_CONFIG) -> None:
    """Full rewrite. Destroys comments -- prefer set_config_value for one key."""
    with open(path, "w") as f:
        yaml.safe_dump(dict(cfg), f, sort_keys=False, default_flow_style=False)


def set_config_value(key: str, value, path: Path | str = DEFAULT_CONFIG) -> None:
    """Update one top-level scalar in place, preserving comments and layout.

    A full yaml.safe_dump round-trip silently strips every comment in the file,
    which lost the whole annotated config once already when the M0 benchmark
    wrote DEVICE.
    """
    path = Path(path)
    lines = path.read_text().splitlines(keepends=True)
    pat = re.compile(rf"^({re.escape(key)}\s*:\s*)([^#\n]*)(.*)$")
    for i, line in enumerate(lines):
        m = pat.match(line)
        if m:
            comment = m.group(3).rstrip("\n")
            lines[i] = f"{m.group(1)}{value}{(' ' + comment) if comment.strip() else ''}\n"
            path.write_text("".join(lines))
            return
    with open(path, "a") as f:
        f.write(f"{key}: {value}\n")


def env(key: str, default: str | None = None) -> str:
    v = os.environ.get(key, default)
    if v is None:
        raise KeyError(f"{key} not set in .env")
    return v


def set_seeds(seed: int | None = None) -> int:
    """Seed python, numpy and torch (agent.md Section 2)."""
    if seed is None:
        seed = int(env("SEED", "1234"))
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except ImportError:
        pass
    return seed


def get_device(cfg: Config | None = None) -> str:
    """Device from config (set by the M0 benchmark). Falls back to cpu if unset."""
    cfg = cfg or load_config()
    d = cfg.get("device")
    if d:
        return d
    return "cpu"
